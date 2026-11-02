"""Publish the tuned classify model to the Hugging Face Hub.

    uv run --with huggingface-hub==2.4.0 python scripts/publish_hf.py           # dry run
    uv run --with huggingface-hub==2.4.0 python scripts/publish_hf.py --yes     # publish (public)

Repo nihar-damera/creo-classify-0.6b:
    README.md                        tune/MODEL_CARD.md
    model.safetensors, config, tokenizer, chat template    the merged bf16 model
    gguf/creo-classify-0.6b-{q8_0,q4_k_m}.gguf             for llama.cpp
The weights live in tune/models.nosync (gitignored). The token is read from the
macOS Keychain (service "hf-token") or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "tune" / "models.nosync"
REPO = "nihar-damera/creo-classify-0.6b"


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return subprocess.run(
        ["security", "find-generic-password", "-s", "hf-token", "-w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def files() -> dict[str, Path]:
    merged = MODELS / "creo-classify-0.6b"
    out = {"README.md": ROOT / "tune" / "MODEL_CARD.md"}
    out |= {p.name: p for p in sorted(merged.iterdir()) if p.is_file()}
    for q in ("q8_0", "q4_k_m"):
        name = f"creo-classify-0.6b-{q}.gguf"
        out[f"gguf/{name}"] = MODELS / name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--yes", action="store_true", help="publish; without it, a dry run")
    args = ap.parse_args()
    fs = files()
    missing = [str(p) for p in fs.values() if not p.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    print(f"model {REPO}: {len(fs)} files")
    for name, path in fs.items():
        print(f"  {name:42} {path.stat().st_size / 1e6:8.1f} MB")
    if not args.yes:
        print("dry run: nothing was uploaded. Re-run with --yes to publish (public).")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token())
    api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in fs.items()]
    commit = api.create_commit(
        repo_id=REPO, operations=ops, commit_message="Publish creo-classify-0.6b"
    )
    print(f"model: https://huggingface.co/{REPO} at revision {commit.oid}")


if __name__ == "__main__":
    main()
