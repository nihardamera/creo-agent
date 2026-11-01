"""Load the committed dataset, with a guard that keeps test splits out of tuning.

Every loader call states its purpose. Code that tunes anything (prompts,
screen patterns, judge rubrics, thresholds) passes purpose="tune" and can
only read the dev splits; asking for a test split raises. Only the eval
harness passes purpose="eval".
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Literal

from creo.dataset.schema import SPLITS, TUNABLE_SPLITS, Case, Split

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "v1"
Purpose = Literal["tune", "eval"]


class HeldOutSplitError(PermissionError):
    """Raised when tuning code asks for a held-out split."""


def _read(split: Split, root: Path) -> bytes:
    with gzip.open(root / f"{split}.jsonl.gz", "rb") as f:
        return f.read()


def load(split: Split, *, purpose: Purpose, root: Path = DATASET_DIR) -> list[Case]:
    if purpose == "tune" and split not in TUNABLE_SPLITS:
        raise HeldOutSplitError(
            f"split {split!r} is held out; tuning may read {sorted(TUNABLE_SPLITS)}"
        )
    data = _read(split, root)
    return [Case.model_validate_json(line) for line in data.decode("utf-8").splitlines() if line]


def load_all(root: Path = DATASET_DIR) -> list[Case]:
    """Every split, for validation and reporting only."""
    return [c for s in SPLITS for c in load(s, purpose="eval", root=root)]


def verify_manifest(root: Path = DATASET_DIR) -> list[str]:
    """Problems found comparing the files on disk with manifest.json."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for split in SPLITS:
        digest = hashlib.sha256(_read(split, root)).hexdigest()
        if digest != manifest["splits"][split]["sha256"]:
            problems.append(f"{split}: content hash differs from manifest.json")
    from creo.dataset.generate import config_hashes

    if config_hashes() != manifest["config_sha256"]:
        problems.append("config files changed since the dataset was generated; regenerate it")
    return problems
