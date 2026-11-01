"""Cost per alert and per 1,000 alerts, from measured tokens and dated prices.

    uv run python -m creo.eval.cost eval/runs/<name>/report.json

Token counts come from the eval run (what the local model actually used per
alert); prices from eval/prices.json, with the date they were checked. An API
model's tokenizer counts differently from the local one, so API rows are an
estimate at the same token counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from creo.eval.run import EVAL_DIR


def table(report: dict[str, Any], prices: dict[str, Any]) -> str:
    s = report["summary"]
    tin, tout = s["tokens_in_per_alert"], s["tokens_out_per_alert"]
    calls = s["llm_calls_per_alert"]["value"]
    rows = [
        f"Measured on `{report['meta']['name']}`: {s['alerts']} alerts, {calls:.3f} model calls, "
        f"{tin:.0f} input and {tout:.0f} output tokens per alert.",
        f"Prices checked {prices['checked']} ({prices['source']}).",
        "",
        "| Model | $ / M input | $ / M output | $ per alert | $ per 1,000 alerts |",
        "|---|--:|--:|--:|--:|",
    ]
    for model, p in prices["usd_per_million_tokens"].items():
        per_alert = (tin * p["input"] + tout * p["output"]) / 1e6
        cells = [
            model,
            f"{p['input']:.2f}",
            f"{p['output']:.2f}",
            f"{per_alert:.6f}",
            f"{1000 * per_alert:.4f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    rows += ["", "API rows assume the same token counts as the local model's tokenizer measured."]
    return "\n".join(rows) + "\n"


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    prices = json.loads((EVAL_DIR / "prices.json").read_text())
    out = table(report, prices)
    (Path(sys.argv[1]).parent / "cost.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
