"""Compare local models on the dev split before choosing one (tuning data only).

    uv run python scripts/compare_models.py qwen3.5:4b gemma3:4b --n 60

Runs dev cases that reach a model step through the real harness with each
model, and reports classification accuracy against the labels, the share of
outputs that fail validation, summary validity and latency.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from creo.agent.llm import OllamaLLM
from creo.dataset.io import load
from creo.harness import run_case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()
    dev = [c for c in load("dev", purpose="tune") if c.labels.expected_llm_steps]
    judged = [c for c in dev if c.labels.expected_classification]
    others = [c for c in dev if not c.labels.expected_classification]
    rng = random.Random(0)
    half = args.n // 2
    cases = rng.sample(judged, min(half, len(judged))) + rng.sample(
        others, args.n - min(half, len(judged))
    )
    n_judged = sum(bool(c.labels.expected_classification) for c in cases)
    print(f"{len(cases)} dev cases ({n_judged} with judgement)")
    report = {}
    for model in args.models:
        llm = OllamaLLM(args.url, model)
        calls, right, wrong, t0 = [], 0, 0, time.time()
        for c in cases:
            o = run_case(c, llm)
            calls += o.llm_calls
            if c.labels.expected_classification:
                if o.category == c.labels.expected_classification:
                    right += 1
                else:
                    wrong += 1
        by_task = {t: [x for x in calls if x["task"] == t] for t in ("classify", "summarize")}
        report[model] = {
            "classify_accuracy": round(right / max(1, right + wrong), 3),
            "classify_n": right + wrong,
            "classify_invalid": sum(not x["ok"] for x in by_task["classify"]),
            "summary_invalid": sum(not x["ok"] for x in by_task["summarize"]),
            "summary_n": len(by_task["summarize"]),
            "invalid_reasons": sorted({x["reason"] for x in calls if not x["ok"]}),
            "latency_ms_p50": round(statistics.median(x["latency_ms"] for x in calls))
            if calls
            else None,
            "latency_ms_max": round(max((x["latency_ms"] for x in calls), default=0)),
            "tokens_in_mean": round(statistics.mean(x["input_tokens"] for x in calls))
            if calls
            else None,
            "tokens_out_mean": round(statistics.mean(x["output_tokens"] for x in calls))
            if calls
            else None,
            "wall_s": round(time.time() - t0),
        }
        print(model, json.dumps(report[model]), flush=True)


if __name__ == "__main__":
    main()
