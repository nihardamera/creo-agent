"""Count Claude input tokens for every model request in an eval run.

    ANTHROPIC_API_KEY=... uv run python scripts/claude_token_counts.py eval/runs/after-qwen3.5-4b

The local model's tokenizer and Claude's count the same request differently,
so the cost table's Claude rows use these counts instead of the local ones.
Uses the token-counting endpoint (no charge). Counts the system prompt and the
facts message as the agent sends them; output tokens aren't counted, so the
cost table keeps the local model's output counts for those.
Writes <run>/claude_tokens.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from creo.agent.llm import SYSTEM

MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5")


def main() -> None:
    run = Path(sys.argv[1])
    outs = [json.loads(x) for x in (run / "outcomes.jsonl").read_text().splitlines()]
    calls = [(o["case"], c) for o in outs for c in o["llm_calls"]]
    client = anthropic.Anthropic(max_retries=8)

    def count(model: str, call: dict[str, object]) -> int:
        task = str(call["task"])
        return client.messages.count_tokens(
            model=model,
            system=SYSTEM[task],  # type: ignore[index]
            messages=[{"role": "user", "content": str(call["request_json"])}],
        ).input_tokens

    result: dict[str, object] = {"run": run.name, "alerts": len(outs), "calls": len(calls)}
    for model in MODELS:
        with ThreadPoolExecutor(4) as pool:
            counts = list(pool.map(lambda pc, m=model: count(m, pc[1]), calls))
        local = [int(c["input_tokens"]) for _, c in calls]  # type: ignore[call-overload]
        by_task: dict[str, list[int]] = {}
        for (_, c), n in zip(calls, counts, strict=True):
            by_task.setdefault(str(c["task"]), []).append(n)
        result[model] = {
            "input_tokens_per_alert": round(sum(counts) / len(outs), 1),
            "mean_by_task": {t: round(statistics.mean(v), 1) for t, v in sorted(by_task.items())},
            "ratio_to_local": round(sum(counts) / sum(local), 3),
        }
        print(model, result[model], flush=True)
    (run / "claude_tokens.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
