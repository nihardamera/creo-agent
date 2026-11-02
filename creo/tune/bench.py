"""Serving benchmark: a concurrency sweep of agent requests against an OpenAI-compatible server.

    uv run python -m creo.tune.bench --url http://localhost:8000/v1 --model creo-classify \\
        --name vllm-4090-prefix --usd-per-hour 0.34

The requests are the agent's own classify requests (test split). With
sort_keys JSON, the system prompt, the category list and the category meanings
come before the evidence, so every request on a runbook shares a long prefix.
Each request also gets a unique last field, because the sweep cycles through
185 requests and a server keeps its cache between levels: without it, prefix
caching would also be credited for whole repeated requests, which real alerts
never are.

Per concurrency level: TTFT and end-to-end latency p50/p95, mean inter-token
latency, output tokens/s, requests/s, and $/1M output tokens at the given
hourly price. Writes tune/bench/<name>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from creo.agent.llm import LLMRequest
from creo.tune.data import read

OUT = Path(__file__).resolve().parents[2] / "tune" / "bench"
LEVELS = (1, 2, 4, 8, 16, 32, 64)


async def one(
    client: httpx.AsyncClient, url: str, model: str, facts: dict[str, Any]
) -> dict[str, float]:
    facts = {**facts, "zz_request": uuid.uuid4().hex}  # sorts last: only the shared prefix caches
    body = {
        "model": model,
        "messages": LLMRequest("classify", facts).messages(),
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Qwen3's template thinks by default: without this the model writes a <think>
        # block until max_tokens and never answers (found in the first sweep).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    times: list[float] = []
    tokens = 0
    async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                tokens = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    times.append(time.perf_counter())
    end = time.perf_counter()
    if not times:
        raise RuntimeError("no tokens streamed")
    gaps = [b - a for a, b in itertools.pairwise(times)]
    return {
        "ttft": times[0] - t0,
        "latency": end - t0,
        "itl": statistics.mean(gaps) if gaps else 0.0,
        "tokens": float(tokens or len(times)),
    }


async def level(url: str, model: str, prompts: list[dict[str, Any]], c: int) -> dict[str, Any]:
    n = max(32, 4 * c)
    work = [prompts[i % len(prompts)] for i in range(n)]
    sem = asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=c + 4)) as cl:

        async def run(f: dict[str, Any]) -> dict[str, float]:
            async with sem:
                return await one(cl, url, model, f)

        await run(work[0])  # warm-up, not counted
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(run(f) for f in work))
        wall = time.perf_counter() - t0

    def pct(key: str, q: int) -> float:
        xs = sorted(r[key] for r in rs)
        return round(xs[min(len(xs) - 1, int(q / 100 * len(xs)))] * 1000, 1)

    tok = sum(r["tokens"] for r in rs)
    return {
        "concurrency": c,
        "requests": n,
        "ttft_ms_p50": pct("ttft", 50),
        "ttft_ms_p95": pct("ttft", 95),
        "latency_ms_p50": pct("latency", 50),
        "latency_ms_p95": pct("latency", 95),
        "itl_ms_mean": round(statistics.mean(r["itl"] for r in rs) * 1000, 2),
        "output_tokens_per_s": round(tok / wall, 1),
        "requests_per_s": round(n / wall, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--usd-per-hour", type=float, default=0.0, help="0 for local hardware")
    ap.add_argument("--levels", type=int, nargs="*", default=list(LEVELS))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    prompts = [it["facts"] for it in read("test")]
    rows = []
    for c in args.levels:
        r = asyncio.run(level(args.url, args.model, prompts, c))
        if args.usd_per_hour:
            r["usd_per_1m_output_tokens"] = round(
                args.usd_per_hour / (r["output_tokens_per_s"] * 3600) * 1e6, 3
            )
        print(json.dumps(r), flush=True)
        rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"name": args.name, "model": args.model, "usd_per_hour": args.usd_per_hour}
    (OUT / f"{args.name}.json").write_text(
        json.dumps({**meta, "note": args.note, "levels": rows}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
