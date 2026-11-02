"""Generate answers with vLLM for the eval sets, or sample groups to find useful GRPO prompts.

    python tune/gpu/gen.py --model /workspace/models/sft-0.6b/merged --out /workspace/out/sft-0.6b
    python tune/gpu/gen.py --model ... --out ... --sets grpo grpo_probes --n 8 --temperature 1

Greedy answers go to <out>/<set>.jsonl as {id, text, output_tokens}, the input
creo.tune.evaluate --completions scores. With --n > 1 each row has `texts`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from common import SHAPE, prompt, read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="*", default=["val", "val_probes", "test", "test_probes"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--constrained", action="store_true", help="decode under the SHAPE grammar")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    so = StructuredOutputsParams(regex=SHAPE) if args.constrained else None
    sp = SamplingParams(
        n=args.n, temperature=args.temperature, max_tokens=256, seed=0, structured_outputs=so
    )
    args.out.mkdir(parents=True, exist_ok=True)
    timing = {}
    for s in args.sets:
        items = read(s)
        t0 = time.time()
        outs = llm.generate([prompt(tok, it["facts"]) for it in items], sp)
        timing[s] = round(time.time() - t0, 2)
        with (args.out / f"{s}.jsonl").open("w") as f:
            for it, o in zip(items, outs, strict=True):
                row = {"id": it["id"], "text": o.outputs[0].text}
                row["output_tokens"] = len(o.outputs[0].token_ids)
                if args.n > 1:
                    row["texts"] = [c.text for c in o.outputs]
                    row["n_tokens"] = [len(c.token_ids) for c in o.outputs]
                f.write(json.dumps(row) + "\n")
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
