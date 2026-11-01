"""Group a run's failures on the *dev* split into a taxonomy, to decide what to fix.

    uv run python -m creo.eval.failures eval/runs/<name>

Reads dev cases only (through the tuning guard), so fixes are chosen without
looking at the held-out splits. Each failure is put in one bucket by the first
thing that went wrong, then counted by bucket and by runbook.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from creo.dataset.io import load
from creo.dataset.schema import Case


def bucket(case: Case, out: dict[str, Any]) -> str | None:
    lab = case.labels
    want_path = [str(n) for n in lab.expected_path]
    if out["path"] == want_path and out["final_state"] == str(lab.expected_final_state):
        return None
    calls = out["llm_calls"]
    bad = [c["reason"] for c in calls if c["node"] == "classify" and not c["ok"]]
    if bad:
        return f"classify output invalid ({bad[0]})"
    if lab.expected_classification and out.get("category") not in (
        None,
        lab.expected_classification,
    ):
        return f"wrong category: {lab.expected_classification} -> {out.get('category')}"
    if out.get("reason") == "cannot_bind_parameters":
        return "category right but cited evidence can't bind the fix"
    if lab.adversarial:
        return f"attack not stopped where expected ({out.get('reason')})"
    return f"path differs ({out.get('reason')})"


def main() -> None:
    run = Path(sys.argv[1])
    dev = {c.id: c for c in load("dev", purpose="tune") + load("adversarial_dev", purpose="tune")}
    buckets: Counter[str] = Counter()
    by_runbook: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    total = 0
    for line in (run / "outcomes.jsonl").read_text().splitlines():
        out = json.loads(line)
        case = dev.get(out["case"])
        if case is None:
            continue  # held-out: not looked at here
        total += 1
        b = bucket(case, out)
        if b:
            buckets[b] += 1
            by_runbook[str(case.labels.expected_runbook)] += 1
            examples.setdefault(b, []).append(case.id)
    print(f"{sum(buckets.values())} of {total} dev cases failed\n")
    for b, n in buckets.most_common():
        print(f"{n:4d}  {b}   e.g. {', '.join(examples[b][:3])}")
    print("\nby runbook:", dict(by_runbook.most_common()))


if __name__ == "__main__":
    main()
