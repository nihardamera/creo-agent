"""Teacher answers from Claude Haiku 5.5 on train and val, through the Batches API (half price).

    uv run python -m creo.tune.teacher submit     # writes tune/teacher/batch.json
    uv run python -m creo.tune.teacher collect    # waits, writes tune/teacher/<set>.jsonl
    uv run python -m creo.tune.teacher sft        # writes tune/data/sft_{train,val}.jsonl

The SFT targets are the teacher's answers that pass the agent's validation and
name the gold category (rejection sampling). A test item never goes to the
teacher: the batch is built from train and val files only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

from creo.agent.llm import SYSTEM, AnthropicLLM, LLMRequest
from creo.tune.data import OUT, read
from creo.tune.score import score

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "tune" / "teacher"
MODEL = "claude-haiku-5-5"
SETS = ("train", "train_probes", "val", "val_probes")


def submit() -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    llm = AnthropicLLM(MODEL)
    keys = [(s, it) for s in SETS for it in read(s)]
    # custom_id allows only [A-Za-z0-9_-], so requests are numbered and the index kept.
    reqs = [
        Request(
            custom_id=f"r{i}",
            params=cast(
                MessageCreateParamsNonStreaming, llm.params(LLMRequest("classify", it["facts"]))
            ),
        )
        for i, (_, it) in enumerate(keys)
    ]
    batch = llm.client.messages.batches.create(requests=reqs)
    DIR.mkdir(parents=True, exist_ok=True)
    index = {f"r{i}": [s, it["id"]] for i, (s, it) in enumerate(keys)}
    meta = {"id": batch.id, "n": len(reqs), "index": index}
    (DIR / "batch.json").write_text(json.dumps(meta) + "\n")
    print(batch.id, len(reqs))


def collect() -> None:
    llm = AnthropicLLM(MODEL)
    meta = json.loads((DIR / "batch.json").read_text())
    bid, index = meta["id"], meta["index"]
    while (b := llm.client.messages.batches.retrieve(bid)).processing_status != "ended":
        print(b.request_counts, flush=True)
        time.sleep(30)
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    usage = {"input_tokens": 0, "output_tokens": 0}
    for r in llm.client.messages.batches.results(bid):
        s, iid = index[r.custom_id]
        row: dict[str, Any] = {"id": iid, "text": "", "error": r.result.type}
        if r.result.type == "succeeded":
            msg = r.result.message
            usage["input_tokens"] += msg.usage.input_tokens
            usage["output_tokens"] += msg.usage.output_tokens
            if msg.stop_reason == "refusal":
                row["error"] = "refusal"
            else:
                row = {
                    "id": iid,
                    "text": next((c.text for c in msg.content if c.type == "text"), ""),
                    "output_tokens": msg.usage.output_tokens,
                    "error": "",
                }
        out[s].append(row)
    for s, rows in out.items():
        rows.sort(key=lambda x: x["id"])
        (DIR / f"{s}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    # Batch price: half the standard $0.10 / $0.50 per million (eval/prices.json, 2026-11-01).
    cost = (usage["input_tokens"] * 0.05 + usage["output_tokens"] * 0.25) / 1e6
    (DIR / "usage.json").write_text(json.dumps({**usage, "usd": round(cost, 4)}, indent=2) + "\n")
    print(usage, f"${cost:.3f}")


def sft() -> None:
    stats = {}
    for split in ("train", "val"):
        rows = []
        kept = total = 0
        for s in (split, f"{split}_probes"):
            answers = {
                a["id"]: a for a in map(json.loads, (DIR / f"{s}.jsonl").read_text().splitlines())
            }
            for it in read(s):
                total += 1
                a = answers.get(it["id"])
                if a is None or not a["text"] or not score(it, a["text"]).correct:
                    continue
                kept += 1
                # Re-serialised compactly so every target has one canonical form.
                target = json.dumps(json.loads(a["text"]), separators=(", ", ": "))
                rows.append(
                    {
                        "id": it["id"],
                        "messages": [
                            {"role": "system", "content": SYSTEM["classify"]},
                            {"role": "user", "content": json.dumps(it["facts"], sort_keys=True)},
                            {"role": "assistant", "content": target},
                        ],
                    }
                )
        (OUT / f"sft_{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        stats[split] = {"kept": kept, "of": total}
    (DIR / "sft_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["submit", "collect", "sft"])
    {"submit": submit, "collect": collect, "sft": sft}[ap.parse_args().command]()


if __name__ == "__main__":
    main()
