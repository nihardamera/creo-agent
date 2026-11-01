"""Run dataset cases through the agent and write a report.

    uv run python -m creo.eval.run --name full-qwen --llm ollama --model qwen3.5:4b \\
        --splits test adversarial_test --record cassettes/default.jsonl
    uv run python -m creo.eval.run --name ci --llm mock --cases-file eval/ci_subset.json

Writes eval/runs/<name>/outcomes.jsonl (one observed trajectory per case),
report.json (headline metrics with 95% CIs, per split, per class) and
report.html. With --record, every model response is appended to a cassette
so mock mode can replay the run with no model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from creo.agent.llm import (
    PROMPT_VERSION,
    AnthropicLLM,
    CassetteLLM,
    LLMClient,
    OllamaLLM,
    RecordingLLM,
    RouterLLM,
)
from creo.dataset.io import DATASET_DIR, load
from creo.dataset.schema import SPLITS, Case
from creo.eval import baseline, html
from creo.eval.metrics import by_class, per_case, summarise
from creo.harness import run_case

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def git_sha() -> str:
    try:

        def git(*args: str) -> str:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

        dirty = git("status", "--porcelain", "--untracked-files=no").strip()
        return git("rev-parse", "--short", "HEAD").strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_digest(url: str, model: str) -> str | None:
    """The exact model build behind a tag, so a result can be tied to the weights."""
    try:
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=5).json()["models"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    return next((m["digest"] for m in tags if m["name"] == model), None)


def client(spec: str, url: str) -> LLMClient:
    """`ollama:<model>` or `anthropic:<model>`."""
    kind, _, model = spec.partition(":")
    return AnthropicLLM(model) if kind == "anthropic" else OllamaLLM(url, model)


def make_llm(
    mode: str,
    model: str,
    url: str,
    record: Path | None,
    cassette: Path,
    routes: dict[str, str] | None = None,
) -> LLMClient:
    if mode == "mock":
        return CassetteLLM(cassette)
    llm: LLMClient
    if mode == "router":
        llm = RouterLLM(
            {t: [client(x, url) for x in spec.split(",")] for t, spec in (routes or {}).items()}
        )
    elif mode == "anthropic":
        llm = AnthropicLLM(model)
    else:
        llm = OllamaLLM(url, model)
    return RecordingLLM(llm, record) if record else llm


def evaluate(
    cases: list[Case], llm: LLMClient, *, progress: bool = False, workers: int = 1
) -> list[tuple[Case, dict[str, Any]]]:
    """Run every case; each one is isolated, so they can run in parallel. Order is kept."""
    t0, done = time.time(), 0

    def one(c: Case) -> dict[str, Any]:
        return dataclasses.asdict(run_case(c, llm))

    out: list[tuple[Case, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c, o in zip(cases, pool.map(one, cases), strict=True):
            out.append((c, o))
            done += 1
            if progress and (done % 25 == 0 or done == len(cases)):
                print(f"  {done}/{len(cases)} cases, {time.time() - t0:.0f}s", flush=True)
    return out


def build_report(pairs: list[tuple[Case, dict[str, Any]]], meta: dict[str, Any]) -> dict[str, Any]:
    rows = [per_case(c, o) for c, o in pairs]
    splits = sorted({r["split"] for r in rows}, key=SPLITS.index)
    failures = [
        {k: r[k] for k in ("case", "split", "class", "final_state", "reason", "executed")}
        | {"expected": str(c.labels.expected_final_state), "path_exact": r["path_exact"]}
        for (c, _), r in zip(pairs, rows, strict=True)
        if not (r["final_ok"] and r["path_exact"])
    ]
    return {
        "meta": meta,
        "summary": summarise(rows),
        "by_split": {s: summarise([r for r in rows if r["split"] == s]) for s in splits},
        "by_class": by_class(rows),
        "classification_baseline": baseline.compare(pairs),
        "failures": failures,
    }


def write_report(d: Path, report: dict[str, Any]) -> None:
    (d / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (d / "report.html").write_text(html.render(report).rstrip("\n") + "\n", encoding="utf-8")


def rescore(d: Path) -> None:
    """Recompute a run's metrics from its saved outcomes, with no model calls."""
    meta = json.loads((d / "report.json").read_text())["meta"]
    outs = [json.loads(line) for line in (d / "outcomes.jsonl").read_text().splitlines()]
    by_id = {c.id: c for s in SPLITS for c in load(s, purpose="eval")}
    pairs = [(by_id[o["case"]], {k: v for k, v in o.items() if k != "case"}) for o in outs]
    meta["rescored"] = {
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_sha(),
    }
    write_report(d, build_report(pairs, meta))
    print(f"rescored {d} ({len(pairs)} cases)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", nargs="*", default=["test", "adversarial_test"])
    ap.add_argument("--cases-file", type=Path, help="JSON list of case ids (overrides --splits)")
    ap.add_argument("--llm", choices=["mock", "ollama", "anthropic", "router"], default="mock")
    ap.add_argument(
        "--route-classify", default="", help="router: comma list, e.g. ollama:m,anthropic:m"
    )
    ap.add_argument("--route-summarize", default="", help="router: comma list of clients")
    ap.add_argument("--model", default="")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--cassette", type=Path, default=Path("cassettes/default.jsonl"))
    ap.add_argument("--record", type=Path, help="append real-model responses to this cassette")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "runs")
    ap.add_argument("--workers", type=int, default=1, help="cases run in parallel")
    ap.add_argument(
        "--rescore", action="store_true", help="rebuild runs/<name>/report.* from its outcomes"
    )
    args = ap.parse_args()
    if args.rescore:
        rescore(args.out / args.name)
        return

    if args.cases_file:
        wanted = json.loads(args.cases_file.read_text())
        pool = [c for s in SPLITS for c in load(s, purpose="eval")]
        cases = [c for c in pool if c.id in set(wanted)]
    else:
        cases = [c for s in args.splits for c in load(s, purpose="eval")]
    routes = {"classify": args.route_classify, "summarize": args.route_summarize}
    llm = make_llm(args.llm, args.model, args.url, args.record, args.cassette, routes)
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    meta = {
        "name": args.name,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_sha(),
        "dataset": {"version": manifest["version"], "sha256": manifest["dataset_sha256"]},
        "llm": llm.name,
        "model_digest": model_digest(args.url, args.model) if args.llm == "ollama" else None,
        "prompt_version": PROMPT_VERSION,
        "cases": len(cases),
        "splits": sorted({c.split for c in cases}, key=SPLITS.index),
    }
    print(f"{args.name}: {len(cases)} cases with {llm.name}", flush=True)
    pairs = evaluate(cases, llm, progress=True, workers=args.workers)
    report = build_report(pairs, meta)
    d = args.out / args.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "outcomes.jsonl").open("w", encoding="utf-8") as f:
        for c, o in pairs:
            f.write(json.dumps({"case": c.id, **o}, sort_keys=True) + "\n")
    write_report(d, report)
    s = report["summary"]
    keys = ("final_state_accuracy", "path_exact", "unsafe_action_rate", "injection_resistance")
    print(json.dumps({k: s[k] for k in (*keys, "llm_calls_per_alert")}, indent=1))


if __name__ == "__main__":
    main()
