"""A small SQLite stand-in for Jira Service Management: create, comment, transition.

Used in-process by the eval harness and behind the mock-tickets HTTP service.
Every operation is also written to an append-only log the eval reads.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from creo.agent.env import Clock

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, created_ms INTEGER, status TEXT, title TEXT, body TEXT,
    host TEXT, priority TEXT, assigned_group TEXT, source_run TEXT
);
CREATE TABLE IF NOT EXISTS ops (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, at_ms INTEGER, ticket_id TEXT, op TEXT,
    detail TEXT, source_run TEXT
);
"""
STATUSES = ("open", "resolved", "escalated")


class TicketError(ValueError):
    pass


class SqliteTickets:
    def __init__(self, path: str | Path, clock: Clock) -> None:
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.clock = clock
        self.lock = threading.RLock()

    def _log(self, ticket_id: str, op: str, detail: str) -> None:
        row = self.conn.execute(
            "SELECT source_run FROM tickets WHERE id=?", (ticket_id,)
        ).fetchone()
        self.conn.execute(
            "INSERT INTO ops (at_ms, ticket_id, op, detail, source_run) VALUES (?, ?, ?, ?, ?)",
            (self.clock.now_ms(), ticket_id, op, detail, row["source_run"] if row else None),
        )

    def create(self, *, title: str, body: str, host: str, priority: str, source_run: str) -> str:
        with self.lock:
            n = self.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1001
            tid = f"KF-{n}"
            self.conn.execute(
                "INSERT INTO tickets VALUES (?, ?, 'open', ?, ?, ?, ?, NULL, ?)",
                (tid, self.clock.now_ms(), title, body, host, priority, source_run),
            )
            self._log(tid, "create", title[:200])
            return tid

    def _require(self, ticket_id: str) -> None:
        if not self.conn.execute("SELECT 1 FROM tickets WHERE id=?", (ticket_id,)).fetchone():
            raise TicketError(f"no ticket {ticket_id}")

    def comment(self, ticket_id: str, text: str) -> None:
        with self.lock:
            self._require(ticket_id)
            self._log(ticket_id, "comment", text)

    def transition(self, ticket_id: str, status: str, *, group: str | None = None) -> None:
        if status not in STATUSES:
            raise TicketError(f"unknown status {status}")
        with self.lock:
            self._require(ticket_id)
            self.conn.execute(
                "UPDATE tickets SET status=?, assigned_group=COALESCE(?, assigned_group)"
                " WHERE id=?",
                (status, group, ticket_id),
            )
            self._log(ticket_id, "transition", f"{status} {group or ''}".strip())

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def ops(self, ticket_id: str | None = None) -> list[dict[str, Any]]:
        rows = (
            self.conn.execute("SELECT * FROM ops WHERE ticket_id=? ORDER BY seq", (ticket_id,))
            if ticket_id
            else self.conn.execute("SELECT * FROM ops ORDER BY seq")
        )
        return [dict(r) for r in rows.fetchall()]

    def all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM tickets ORDER BY id").fetchall()]
