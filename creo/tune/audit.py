"""Reward-hacking audit of a scored run: does the policy dodge the unsafe penalty by never fixing?

    uv run python -m creo.tune.audit grpo-base-0.6b sft-0.6b --set val

For each run: predicted vs gold category shares, how often a fix-worthy alert
got a wait/escalate category instead (over-cautious), fix recall, the most
common confusions, and how much of the JSON the model actually varies.
Reads tune/runs/<name>/<set>.jsonl written by creo.tune.evaluate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from creo.eval.metrics import bootstrap

RUNS = Path(__file__).resolve().parents[2] / "tune" / "runs"


def rows(name: str, s: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (RUNS / name / f"{s}.jsonl").read_text().splitlines() if x]


def audit(rs: list[dict[str, Any]]) -> dict[str, Any]:
    fix_gold = [r for r in rs if r["gold_is_fix"]]
    answered = [r for r in rs if r["category"]]
    texts = [json.loads(r["text"]) for r in rs if r["valid"]]
    return {
        "n": len(rs),
        "accuracy": bootstrap([float(r["correct"]) for r in rs]).as_dict(),
        "unsafe": bootstrap([float(r["unsafe"]) for r in rs]).as_dict(),
        "over_cautious_on_fix_alerts": bootstrap(
            [float(r["over_cautious"]) for r in fix_gold]
        ).as_dict(),
        "fix_recall": bootstrap([float(r["correct"]) for r in fix_gold]).as_dict(),
        "share_answered_fix": round(
            sum(r["category"] in r["fix_categories"] for r in answered) / max(1, len(answered)), 3
        ),
        "share_gold_fix": round(len(fix_gold) / max(1, len(rs)), 3),
        "predicted": Counter(r["category"] for r in rs).most_common(),
        "gold": Counter(r["gold"] for r in rs).most_common(),
        "top_confusions": Counter(
            f"{r['gold']} -> {r['category']}" for r in rs if not r["correct"]
        ).most_common(6),
        "distinct_confidences": len({t["confidence"] for t in texts}),
        "evidence_count": Counter(len(t["evidence"]) for t in texts).most_common(),
    }


def main() -> None:
    from creo.tune.data import read

    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--set", default="val")
    args = ap.parse_args()
    fixes = {it["id"]: it["fix_categories"] for it in read(args.set)}
    out = {}
    for name in args.names:
        rs = rows(name, args.set)
        for r in rs:
            r["fix_categories"] = fixes[r["id"]]
        out[name] = audit(rs)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
