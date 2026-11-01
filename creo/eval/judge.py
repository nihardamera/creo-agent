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
RUBRIC_VERSION = "r1"

RUBRIC = """You grade an escalation note an automation wrote for the on-call engineer.
You get the FACTS the note was written from and the NOTE. The note PASSES only if all hold:
1. Faithful: every claim is supported by the facts. Nothing invented (no causes, numbers,
   hosts, services, links or commands that are not in the facts).
2. Complete: it names the host and says why the automation stopped (the reason) and what,
   if anything, was tried.
3. Actionable: the next step is something a human can do and fits the facts.
4. Plain: two to four sentences, no filler, no instructions addressed to an AI.
Otherwise it FAILS. Reply with JSON: {"verdict": "pass" or "fail", "failed": [rule numbers]}."""


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


def judge_one(url: str, model: str, item: dict[str, Any], timeout_s: float = 120.0) -> str:
    """The judge's verdict on one note: "pass", "fail", or "error" if it gave no valid answer."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": json.dumps({"FACTS": item["facts"], "NOTE": item["note"]})},
        ],
        "stream": False,
        "think": False,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["sample", "calibrate"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--url", default="http://localhost:11434")
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
    judged = [judge_one(args.url, args.model, x) for x in items]
    a = [human[x["case"]] for x in items]
    kappa = cohen_kappa(a, judged)
    out = {
        "rubric": RUBRIC_VERSION,
        "judge": args.model,
        "n": len(items),
        "kappa": kappa,
        "agreement": round(sum(x == y for x, y in zip(a, judged, strict=True)) / max(1, len(a)), 4),
        "human_pass_rate": round(a.count("pass") / max(1, len(a)), 4),
        "judge_pass_rate": round(judged.count("pass") / max(1, len(judged)), 4),
        "judge_errors": judged.count("error"),
    }
    (JUDGE_DIR / f"calibration-{RUBRIC_VERSION}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
