"""API cost per 1,000 alerts from a full agent run, pricing each call by the model that served it.

    uv run python -m creo.tune.cost p4-haiku-only p4-router-tuned-0.6b after-qwen3.5-4b

A router run mixes local and API calls; each recorded call carries the model
that answered it and its token counts, so local calls cost $0 and API calls
are priced from eval/prices.json (dated). Only test and adversarial_test
alerts are counted. Hardware and electricity for local models are not counted.
"""

from __future__ import annotations

import json
import statistics
import sys
from typing import Any

from creo.dataset.io import load
from creo.eval.run import EVAL_DIR

SPLITS = ("test", "adversarial_test")


def price_of(model: str, prices: dict[str, Any]) -> dict[str, float]:
    """Dated per-million prices; a model with no row (a local one) costs nothing."""
    row = prices["usd_per_million_tokens"].get(model)
    return (
        {"input": float(row["input"]), "output": float(row["output"])}
        if row
        else {"input": 0.0, "output": 0.0}
    )


def summarise(name: str, prices: dict[str, Any]) -> dict[str, Any]:
    held_out = {c.id for s in SPLITS for c in load(s, purpose="eval")}  # type: ignore[arg-type]
    alerts = usd = 0.0
    calls: list[dict[str, Any]] = []
    for line in (EVAL_DIR / "runs" / name / "outcomes.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["case"] not in held_out:
            continue
        alerts += 1
        for c in row["llm_calls"]:
            p = price_of(c["model"], prices)
            usd += (c["input_tokens"] * p["input"] + c["output_tokens"] * p["output"]) / 1e6
            calls.append(c)
    api = [
        c
        for c in calls
        if c["model"] in prices["usd_per_million_tokens"] and c["model"].startswith("claude")
    ]
    by_task: dict[str, list[float]] = {}
    for c in calls:
        by_task.setdefault(c["task"], []).append(c["latency_ms"])
    return {
        "run": name,
        "alerts": int(alerts),
        "model_calls_per_alert": round(len(calls) / alerts, 3),
        "api_calls_per_alert": round(len(api) / alerts, 3),
        "api_share_of_classify_calls": round(
            sum(c["task"] == "classify" for c in api)
            / max(1, sum(c["task"] == "classify" for c in calls)),
            3,
        ),
        "usd_per_1000_alerts": round(1000 * usd / alerts, 4),
        "latency_ms": {
            t: {
                "p50": round(statistics.median(v), 1),
                "p95": round(statistics.quantiles(v, n=20)[18], 1) if len(v) > 1 else None,
                "n": len(v),
            }
            for t, v in sorted(by_task.items())
        },
        "models": sorted({c["model"] for c in calls}),
    }


def main() -> None:
    prices = json.loads((EVAL_DIR / "prices.json").read_text())
    out = {n: summarise(n, prices) for n in sys.argv[1:]}
    print(json.dumps({"prices_checked": prices["checked"], "runs": out}, indent=1))


if __name__ == "__main__":
    main()
