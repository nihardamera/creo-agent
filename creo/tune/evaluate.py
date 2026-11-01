"""Per-step evaluation of a model on the classify items, every rate with a 95% CI.

    uv run python -m creo.tune.evaluate --name qwen3.5-4b --llm ollama --model qwen3.5:4b
    uv run python -m creo.tune.evaluate --name haiku --llm anthropic --model claude-haiku-5-5
    uv run python -m creo.tune.evaluate --name grpo --llm openai --model creo-classify \\
        --url http://localhost:8080/v1
    uv run python -m creo.tune.evaluate --name grpo --completions tune/runs/grpo/raw

Writes tune/runs/<name>/<set>.jsonl (one scored answer per item) and
tune/runs/<name>/report.json. --completions scores answers generated
elsewhere (a GPU pod): <dir>/<set>.jsonl with {id, text, output_tokens, latency_ms}.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from creo.agent.llm import (
    AnthropicLLM,
    LLMClient,
    LLMError,
    LLMRequest,
    OllamaLLM,
    OpenAICompatLLM,
)
from creo.eval.metrics import bootstrap
from creo.tune.data import read
from creo.tune.score import as_row

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "tune" / "runs"
SETS = ("test", "test_probes")


def make_llm(kind: str, model: str, url: str, key: str) -> LLMClient:
    if kind == "ollama":
        return OllamaLLM(url, model)
    if kind == "anthropic":
        return AnthropicLLM(model)
    return OpenAICompatLLM(url, model, key)


def generate(llm: LLMClient, items: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    def one(it: dict[str, Any]) -> dict[str, Any]:
        try:
            r = llm.complete(LLMRequest("classify", it["facts"]))
        except LLMError as e:
            return {"id": it["id"], "text": "", "error": str(e), "output_tokens": 0}
        return {
            "id": it["id"],
            "text": r.text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_ms": r.latency_ms,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def summarise(rows: list[dict[str, Any]], probes: bool) -> dict[str, Any]:
    def rate(key: str, subset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return bootstrap([float(r[key]) for r in (rows if subset is None else subset)]).as_dict()

    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": rate("correct"),
        "valid_json": rate("valid"),
        "category_right_ignoring_validation": bootstrap(
            [float(r["category"] == r["gold"]) for r in rows]
        ).as_dict(),
        "refused": sum(r["error"].startswith("refusal") for r in rows),
        "errors": sum(bool(r["error"]) and not r["error"].startswith("refusal") for r in rows),
        "unsafe_choice": rate("unsafe"),
        "over_cautious": rate("over_cautious", [r for r in rows if r["gold_is_fix"]]),
        "invalid_reasons": {
            k: sum(r["why"] == k for r in rows) for k in sorted({r["why"] for r in rows})
        },
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(statistics.quantiles(lat, n=20)[18], 1) if len(lat) > 1 else None,
        "output_tokens_mean": round(statistics.mean(r["output_tokens"] for r in rows), 1),
    }
    if probes:
        out["followed_injection"] = rate("followed_injection")
        out["injection_resisted"] = bootstrap(
            [float(not r["followed_injection"] and not r["unsafe"]) for r in rows]
        ).as_dict()
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["gold"], []).append(float(r["correct"]))
    out["accuracy_by_gold"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())}
    return out


def score_set(items: list[dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {a["id"]: a for a in answers}
    rows = []
    for it in items:
        a = by_id.get(it["id"], {"text": "", "output_tokens": 0})
        rows.append(
            {
                **as_row(it, a["text"]),
                "gold_is_fix": it["gold"] in it["fix_categories"],
                "text": a["text"],
                "output_tokens": a.get("output_tokens", 0),
                "input_tokens": a.get("input_tokens"),
                "latency_ms": a.get("latency_ms"),
                "error": a.get("error", ""),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--llm", choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--model", default="")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--key", default="none")
    ap.add_argument("--completions", type=Path, help="score answers generated elsewhere")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    d = RUNS / args.name
    d.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"name": args.name, "llm": args.llm, "model": args.model, "sets": {}}
    for s in args.sets:
        items = read(s)
        if args.completions:
            src = args.completions / f"{s}.jsonl"
            answers = [json.loads(x) for x in src.read_text().splitlines() if x]
        else:
            llm = make_llm(args.llm, args.model, args.url, args.key)
            answers = generate(llm, items, args.workers)
        rows = score_set(items, answers)
        (d / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report["sets"][s] = summarise(rows, probes=s.endswith("probes"))
        print(s, json.dumps({k: report["sets"][s][k] for k in ("accuracy", "unsafe_choice")}))
    (d / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
