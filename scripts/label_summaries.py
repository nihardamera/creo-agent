"""Hand-label escalation notes for judge calibration (for Nihar, not the agent).

    uv run python scripts/label_summaries.py

Shows each sampled note with the facts it was written from. Answer p (pass)
or f (fail) by the rubric in creo/eval/judge.py, or s to skip, q to quit.
Labels are saved after every answer, so you can stop and resume.
"""

from __future__ import annotations

import json

from creo.eval.judge import HUMAN, RUBRIC, SAMPLE


def main() -> None:
    items = [json.loads(x) for x in SAMPLE.read_text().splitlines() if x]
    done = (
        {json.loads(x)["case"] for x in HUMAN.read_text().splitlines() if x}
        if HUMAN.exists()
        else set()
    )
    todo = [x for x in items if x["case"] not in done]
    print(RUBRIC, f"\n\n{len(done)} labelled, {len(todo)} to go.\n")
    for i, x in enumerate(todo, 1):
        print("=" * 72, f"\n[{i}/{len(todo)}] {x['case']}  reason={x['reason']}")
        print("FACTS:", json.dumps(x["facts"], indent=1))
        print("\nNOTE:\n" + x["note"])
        while (a := input("\npass/fail/skip/quit [p/f/s/q]: ").strip().lower()) not in (
            "p",
            "f",
            "s",
            "q",
        ):
            pass
        if a == "q":
            return
        if a == "s":
            continue
        note = input("why (optional): ").strip()
        with HUMAN.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"case": x["case"], "label": "pass" if a == "p" else "fail", "why": note}
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
