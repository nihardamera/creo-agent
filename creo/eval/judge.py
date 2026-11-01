"""An LLM judge for escalation-note quality, and its calibration against a human.

The judge reads the same typed facts the note was written from, plus the
note, and returns pass/fail on a written rubric. It is trusted only as far as
it agrees with a human: Cohen's kappa between judge and hand labels on the
same notes. Target kappa >= 0.6 ("substantial"); if iterating the rubric
doesn't get there, the README says so.

    uv run python -m creo.eval.judge sample --run eval/runs/<name>   # notes to label
    uv run python scripts/label_summaries.py                          # you label them
    uv run python -m creo.eval.judge calibrate --model qwen3.5:9b     # judge them, report kappa
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from creo.dataset.io import load

JUDGE_DIR = Path(__file__).resolve().parents[2] / "eval" / "judge"
SAMPLE = JUDGE_DIR / "sample.jsonl"
HUMAN = JUDGE_DIR / "human_labels.jsonl"
SPLIT = JUDGE_DIR / "split.json"

RUBRIC_R1 = """You grade an escalation note an automation wrote for the on-call engineer.
You get the FACTS the note was written from and the NOTE. The note PASSES only if all hold:
1. Faithful: every claim is supported by the facts. Nothing invented (no causes, numbers,
   hosts, services, links or commands that are not in the facts).
2. Complete: it names the host and says why the automation stopped (the reason) and what,
   if anything, was tried.
3. Actionable: the next step is something a human can do and fits the facts.
4. Plain: two to four sentences, no filler, no instructions addressed to an AI.
Otherwise it FAILS. Reply with JSON: {"verdict": "pass" or "fail", "failed": [rule numbers]}."""

# r2: r1 plus what a human grader knows and the facts don't say: what each reason
# code means and how to read steps_taken. Written from creo/agent/nodes.py.
GLOSSARY = """What the reason codes mean (why the automation stopped and handed over):
- approval_denied: a fix was proposed and the human approver refused it.
- approval_timeout: a fix was proposed and nobody answered before the approval expired.
- did_not_recover: the runbook waited and rechecked, the alert didn't clear, and its plan
  has no fix for this category, so it escalates.
- fixes_exhausted: every fix the runbook allows was tried and the alert still didn't clear.
- executor_error / executor_refused: the executor failed or refused to run the approved action.
- cannot_bind_parameters: the evidence couldn't fill in the fix's exact parameters.
- no_fix_tier: the host is tier 0 (critical); policy forbids automated fixes there.
- runbook_escalates: the runbook's plan for this category is to escalate straight away.
- flapping: the monitor changed state too often recently; a human looks before automation acts.
- runbook_conflict: the alert's tags named several runbooks or monitor types, or a runbook
  that doesn't fit the monitor type or the host's role.
- unknown_runbook: the alert's tags named a runbook that doesn't exist.
- unknown_monitor: no single runbook covers this monitor type on a host with this role.
- unknown_host: the host isn't in the inventory.
- screen_flagged / diagnostics_flagged: the alert text or diagnostic output looked like an
  attempt to instruct the automation (flags say which rule); it went to a human unread.
- tag_contradicts_inventory / tag_names_missing_resource: the alert's tags disagree with the
  inventory about the host or name a resource it doesn't have; treated as suspicious.
- killswitch: automation is switched off; nothing may change.
How to read the facts: steps_taken lists the workflow steps visited, in order, not their
outcomes. propose_fix followed by approval means a fix was put to a human. actions lists
what the executor actually ran (empty means nothing ran).
Applying the rules: a note FAILS rule 1 if it misdescribes the reason or claims something
happened that the facts don't show; generic checks in the next step (look at the host's
state or logs) are not invented facts. A note FAILS rule 2 if its account of why the
automation stopped contradicts the meaning above."""

RUBRICS = {"r1": RUBRIC_R1, "r2": RUBRIC_R1.replace("\nOtherwise", "\n" + GLOSSARY + "\nOtherwise")}
RUBRIC = RUBRICS["r1"]  # the rubric the README's first calibration used


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: str
    failed: list[int]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement between two raters beyond chance: 1 perfect, 0 chance level."""
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def kappa_ci(
    a: list[str], b: list[str], *, b_reps: int = 2000, seed: int = 20261101
) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI for kappa, resampling (human, judge) pairs."""
    rng = random.Random(seed)
    n = len(a)
    ks = []
    for _ in range(b_reps):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            ks.append(k)
    if not ks:
        return None, None
    ks.sort()
    return round(ks[int(0.025 * len(ks))], 4), round(ks[int(0.975 * len(ks)) - 1], 4)


def sample(run_dir: Path, n: int = 80, seed: int = 3) -> list[dict[str, Any]]:
    """Model-written escalation notes from a run's dev cases, one per case, stratified by reason.

    Dev only: the rubric gets tuned against these labels, so held-out notes stay out.
    """
    dev = {c.id for s in ("dev", "adversarial_dev") for c in load(s, purpose="tune")}
    notes = []
    for line in (run_dir / "outcomes.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o["case"] not in dev:
            continue
        s = o.get("summary") or {}
        calls = [c for c in o["llm_calls"] if c["node"] == "escalate"]
        if s.get("source") != "model" or not calls:
            continue
        notes.append(
            {
                "case": o["case"],
                "reason": o.get("reason"),
                "facts": json.loads(calls[0]["request_json"]),
                "note": f"{s['summary']}\nNext: {s['next_step']}",
            }
        )
    rng = random.Random(seed)
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for x in notes:
        by_reason.setdefault(str(x["reason"]), []).append(x)
    picked: list[dict[str, Any]] = []
    while len(picked) < min(n, len(notes)):
        for r in sorted(by_reason):
            if by_reason[r] and len(picked) < n:
                picked.append(by_reason[r].pop(rng.randrange(len(by_reason[r]))))
    return picked


def judge_one(
    url: str,
    model: str,
    item: dict[str, Any],
    timeout_s: float = 120.0,
    rubric: str = "r1",
    think: bool = False,
) -> str:
    """The judge's verdict on one note: "pass", "fail", or "error" if it gave no valid answer.

    A model named claude-* goes to the Anthropic API (key from ANTHROPIC_API_KEY); any
    other model to Ollama at `url`.
    """
    if model.startswith("claude-"):
        return _judge_claude(model, item, rubric, think)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRICS[rubric]},
            {"role": "user", "content": json.dumps({"FACTS": item["facts"], "NOTE": item["note"]})},
        ],
        "stream": False,
        "think": think,
        "format": Verdict.model_json_schema(),
        "options": {"temperature": 0, "seed": 0},
    }
    try:
        r = httpx.post(f"{url.rstrip('/')}/api/chat", json=body, timeout=timeout_s)
        r.raise_for_status()
        v = Verdict.model_validate_json(r.json()["message"]["content"])
    except (httpx.HTTPError, ValidationError, KeyError):
        return "error"
    return "pass" if v.verdict == "pass" else "fail"


def _judge_claude(model: str, item: dict[str, Any], rubric: str, think: bool) -> str:
    import anthropic

    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "failed": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["failed", "verdict"],
        "additionalProperties": False,
    }
    try:
        msg = anthropic.Anthropic(max_retries=3).messages.create(
            model=model,
            max_tokens=4096,
            system=RUBRICS[rubric],
            messages=[
                {
                    "role": "user",
                    "content": json.dumps({"FACTS": item["facts"], "NOTE": item["note"]}),
                }
            ],
            thinking={"type": "adaptive"} if think else {"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(b.text for b in msg.content if b.type == "text")
        v = Verdict.model_validate_json(text)
    except (anthropic.APIError, StopIteration, ValidationError):
        return "error"
    return "pass" if v.verdict == "pass" else "fail"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["sample", "calibrate"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--rubric", choices=sorted(RUBRICS), default="r1")
    ap.add_argument("--subset", choices=["all", "tune", "test"], default="all")
    ap.add_argument("--think", action="store_true", help="let the judge reason before answering")
    args = ap.parse_args()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "sample":
        if not args.run:
            raise SystemExit("--run is required")
        items = sample(args.run)
        SAMPLE.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in items))
        print(f"wrote {len(items)} notes to {SAMPLE}; label them with scripts/label_summaries.py")
        return
    human = {
        json.loads(x)["case"]: json.loads(x)["label"] for x in HUMAN.read_text().splitlines() if x
    }
    items = [json.loads(x) for x in SAMPLE.read_text().splitlines() if x]
    items = [x for x in items if x["case"] in human]
    if args.subset != "all":
        keep = set(json.loads(SPLIT.read_text())[args.subset])
        items = [x for x in items if x["case"] in keep]
    judged = [judge_one(args.url, args.model, x, 600.0, args.rubric, args.think) for x in items]
    a = [human[x["case"]] for x in items]
    kappa = cohen_kappa(a, judged)
    out = {
        "rubric": args.rubric,
        "subset": args.subset,
        "think": args.think,
        "judge": args.model,
        "n": len(items),
        "kappa": kappa,
        "kappa_ci95": list(kappa_ci(a, judged)),
        "agreement": round(sum(x == y for x, y in zip(a, judged, strict=True)) / max(1, len(a)), 4),
        "human_pass_rate": round(a.count("pass") / max(1, len(a)), 4),
        "judge_pass_rate": round(judged.count("pass") / max(1, len(judged)), 4),
        "judge_errors": judged.count("error"),
        "items": [
            {"case": x["case"], "reason": x["reason"], "human": h, "judge": j}
            for x, h, j in zip(items, a, judged, strict=True)
        ],
    }
    judge = args.model.replace(":", "-")
    name = f"calibration-{judge}-{args.rubric}{'-think' if args.think else ''}-{args.subset}.json"
    (JUDGE_DIR / name).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
