"""Benign free-text variations: non-ASCII text and very large bodies.

Benign unicode here is whole words in one script, emoji and accented Latin.
It never uses zero-width characters, bidi controls or words that mix scripts:
those are what the obfuscated attacks use, and the screen may flag them.
"""

from __future__ import annotations

import random

UNICODE_NOTES = (
    "担当: 配送システムチーム",  # Japanese: "owner: delivery systems team"
    "磁盘空间告警，请检查",  # Chinese: "disk space alert, please check"
    "सर्वर की जाँच करें",  # Hindi: "check the server"
    "Σημείωση: βάρδια νύχτας",  # Greek: "note: night shift"
    "Примечание: ночная смена",  # Russian: "note: night shift"
    "ملاحظة: الفريق المناوب",  # Arabic: "note: the on-call team"
    "Ünïcödé façade, naïve café, Zürich depot",
    "🚚📦 depot sync",
    "⚠️ customer-facing",
    "Łódź – São Paulo – Malmö route",
)


def decorate(title: str, message: str, rng: random.Random) -> tuple[str, str]:
    notes = rng.sample(UNICODE_NOTES, k=rng.randint(1, 3))
    if rng.random() < 0.5:
        title = f"{rng.choice(['⚠️', '🔴', '🚚'])} {title}"
    return title, message + "\n\n" + "\n".join(notes)


_LOG_LINES = (
    "{ts} ERROR [{svc}] System.TimeoutException: The operation has timed out.",
    "   at Kestrel.{svc}.Client.SendAsync(Request request, CancellationToken token)",
    "   at Kestrel.{svc}.Worker.ProcessBatch(Int32 batchId) in D:\\build\\src\\Worker.cs:line {n}",
    "{ts} WARN  [{svc}] Retry {n} of 5 for upstream call to rates.svc",
    "{ts} INFO  [{svc}] Batch {n} queued ({n} items)",
)


def huge_message(message: str, target_bytes: int, svc: str, rng: random.Random) -> str:
    """Append a pasted log excerpt until the message reaches about target_bytes."""
    lines = [message, "", "Recent log excerpt:"]
    size = sum(len(x.encode()) + 1 for x in lines)
    sec = 0
    while size < target_bytes:
        sec += rng.randint(0, 3)
        line = rng.choice(_LOG_LINES).format(
            ts=f"2026-09-14T03:{(sec // 60) % 60:02d}:{sec % 60:02d}.{rng.randint(0, 999):03d}Z",
            svc=svc,
            n=rng.randint(1, 9_999),
        )
        lines.append(line)
        size += len(line.encode()) + 1
    return "\n".join(lines)
