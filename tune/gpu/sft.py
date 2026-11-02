"""SFT a small Qwen on the teacher's filtered answers (LoRA), then merge.

python tune/gpu/sft.py --base Qwen/Qwen3-0.6B --out /workspace/models/sft-0.6b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import DATA, prompt


def rows(tok: object, name: str) -> Dataset:
    out = []
    for line in (DATA / f"{name}.jsonl").read_text().splitlines():
        ex = json.loads(line)
        facts = json.loads(ex["messages"][1]["content"])
        out.append({"prompt": prompt(tok, facts), "completion": ex["messages"][2]["content"]})
    return Dataset.from_list(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    cfg = SFTConfig(
        output_dir=str(args.out / "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers 5: a float in [0, 1) is a ratio (warmup_ratio is gone)
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_length=3072,
        completion_only_loss=True,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=20261102,
    )
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear"
    )
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=rows(tok, "sft_train"),
        eval_dataset=rows(tok, "sft_val"),
        processing_class=tok,
        peft_config=lora,
    )
    t0 = time.time()
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out / "merged")
    tok.save_pretrained(args.out / "merged")
    log = {"base": args.base, "seconds": round(time.time() - t0), "log": trainer.state.log_history}
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
