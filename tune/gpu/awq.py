"""AWQ 4-bit (W4A16) quantization with llmcompressor, calibrated on train prompts only.

    python tune/gpu/awq.py --model /workspace/models/grpo-sft-0.6b/merged \\
        --out /workspace/models/grpo-sft-0.6b-awq

Runs in its own venv: llmcompressor 0.15.0 pins compressed-tensors 0.20.1 and
vllm 0.32.0 pins 0.17.0. vLLM loading the result is checked by the quality eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=128)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    rows = [json.loads(x) for x in (DATA / "sft_train.jsonl").read_text().splitlines()]
    texts = [
        prompt(tok, json.loads(r["messages"][1]["content"])) + r["messages"][2]["content"]
        for r in rows[: args.samples]
    ]
    ds = Dataset.from_list([{"text": t} for t in texts]).map(
        lambda x: tok(x["text"], add_special_tokens=False), remove_columns=["text"]
    )
    recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])]
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=3072,
        num_calibration_samples=len(texts),
    )
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
