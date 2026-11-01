"""Cost per alert and per 1,000 alerts, from measured tokens and dated prices.

    uv run python -m creo.eval.cost eval/runs/<name>/report.json

Token counts come from the eval run (what the local model actually used per
alert); prices from eval/prices.json, with the date they were checked. A
model's tokenizer counts the same request differently: when the run has a
claude_tokens.json (scripts/claude_token_counts.py), Claude rows use Claude's
own input counts. Output tokens are the local model's in every row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from creo.eval.run import EVAL_DIR


def table(
    report: dict[str, Any], prices: dict[str, Any], claude: dict[str, Any] | None = None
) -> str:
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
        counted = (claude or {}).get(model)
        mtin = counted["input_tokens_per_alert"] if counted else tin
        per_alert = (mtin * p["input"] + tout * p["output"]) / 1e6
        cells = [
            model,
            f"{p['input']:.2f}",
            f"{p['output']:.2f}",
            f"{per_alert:.6f}",
            f"{1000 * per_alert:.4f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    if claude:
        ratio = next(v["ratio_to_local"] for k, v in claude.items() if isinstance(v, dict))
        rows.append(
            f"Claude rows use Claude's input token counts for the same requests "
            f"({ratio:.2f}x the local count; system prompt and facts, no structured-output "
            "schema), and the local model's output counts."
        )
    else:
        rows.append(
            "API rows assume the same token counts as the local model's tokenizer measured."
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    prices = json.loads((EVAL_DIR / "prices.json").read_text())
    counts = Path(sys.argv[1]).parent / "claude_tokens.json"
    claude = json.loads(counts.read_text()) if counts.exists() else None
    out = table(report, prices, claude)
    (Path(sys.argv[1]).parent / "cost.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
