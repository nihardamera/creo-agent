"""The agent's own durable records, in SQLite.

LangGraph's checkpointer stores each run's graph state. This store keeps
what the runtime needs around the graph: which events were seen, which runs
are open (for dedupe and correlation), recent transitions (for flapping),
what each paused run is waiting for, executor jobs, approval requests and
model calls. Both survive a restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from creo.approvals import ApprovalRequest, Grant

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, run_id TEXT, at_ms INTEGER);
CREATE TABLE IF NOT EXISTS run_index (
    run_id TEXT PRIMARY KEY, received_ms INTEGER, host TEXT, title TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, monitor_id TEXT, host TEXT, incident_key TEXT,
    opened_ms INTEGER, closed_ms INTEGER, final_state TEXT, ticket_id TEXT,
    is_open INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_open ON runs (is_open, monitor_id, host);
CREATE TABLE IF NOT EXISTS transitions (monitor_id TEXT, host TEXT, transition TEXT, at_ms INTEGER);
CREATE TABLE IF NOT EXISTS pending (
    run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, due_ms INTEGER, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued', result TEXT, created_ms INTEGER
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, request TEXT NOT NULL,
    decision TEXT, grant_json TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT, node TEXT, task TEXT, model TEXT, ok INTEGER, reason TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL, request_json TEXT
);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    # ---- ids and flags ---------------------------------------------------------

    def new_run_id(self) -> str:
        with self.lock:
            row = self.conn.execute("SELECT v FROM meta WHERE k='run_seq'").fetchone()
            n = int(row["v"]) + 1 if row else 1
            self.conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('run_seq', ?)", (str(n),))
            return f"run-{n:06d}"

    def index_run(self, run_id: str, received_ms: int, host: str, title: str) -> None:
        self._q(
            "INSERT OR IGNORE INTO run_index VALUES (?, ?, ?, ?)",
            (run_id, received_ms, host, title),
        )

    def recent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._q(
            "SELECT * FROM run_index ORDER BY received_ms DESC, run_id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def killswitch(self) -> bool:
        rows = self._q("SELECT v FROM meta WHERE k='killswitch'")
        return bool(rows) and rows[0]["v"] == "on"

    def set_killswitch(self, on: bool) -> None:
        self._q(
            "INSERT OR REPLACE INTO meta (k, v) VALUES ('killswitch', ?)", ("on" if on else "off",)
        )

    # ---- events, runs, transitions -------------------------------------------------

    def seen_event(self, event_id: str) -> bool:
        return bool(self._q("SELECT 1 FROM events WHERE event_id=?", (event_id,)))

    def add_event(self, event_id: str, run_id: str, at_ms: int) -> None:
        self._q("INSERT OR IGNORE INTO events VALUES (?, ?, ?)", (event_id, run_id, at_ms))

    def open_run(
        self, run_id: str, monitor_id: str, host: str, incident_key: str | None, at_ms: int
    ) -> None:
        self._q(
            "INSERT OR REPLACE INTO runs (run_id, monitor_id, host, incident_key, opened_ms,"
            " is_open) VALUES (?, ?, ?, ?, ?, 1)",
            (run_id, monitor_id, host, incident_key, at_ms),
        )

    def find_open_run(self, monitor_id: str, host: str, *, exclude: str = "") -> str | None:
        rows = self._q(
            "SELECT run_id FROM runs WHERE is_open=1 AND monitor_id=? AND host=? AND run_id<>?"
            " ORDER BY opened_ms LIMIT 1",
            (monitor_id, host, exclude),
        )
        return rows[0]["run_id"] if rows else None

    def find_incident(self, key: str, since_ms: int, *, exclude: str) -> str | None:
        rows = self._q(
            "SELECT run_id FROM runs WHERE is_open=1 AND incident_key=? AND opened_ms>=?"
            " AND run_id<>? ORDER BY opened_ms LIMIT 1",
            (key, since_ms, exclude),
        )
        return rows[0]["run_id"] if rows else None

    def set_ticket(self, run_id: str, ticket_id: str) -> None:
        self._q("UPDATE runs SET ticket_id=? WHERE run_id=?", (ticket_id, run_id))

    def ticket_of(self, run_id: str) -> str | None:
        rows = self._q("SELECT ticket_id FROM runs WHERE run_id=?", (run_id,))
        return rows[0]["ticket_id"] if rows else None

    def close_run(self, run_id: str, final_state: str, at_ms: int) -> None:
        self._q(
            "UPDATE runs SET is_open=0, closed_ms=?, final_state=? WHERE run_id=?",
            (at_ms, final_state, run_id),
        )

    def add_transition(self, monitor_id: str, host: str, transition: str, at_ms: int) -> None:
        self._q(
            "INSERT INTO transitions VALUES (?, ?, ?, ?)", (monitor_id, host, transition, at_ms)
        )

    def count_transitions(self, monitor_id: str, host: str, since_ms: int) -> int:
        rows = self._q(
            "SELECT COUNT(*) AS n FROM transitions WHERE monitor_id=? AND host=? AND at_ms>=?",
            (monitor_id, host, since_ms),
        )
        return int(rows[0]["n"])

    # ---- what paused runs wait for --------------------------------------------------

    def set_pending(
        self, run_id: str, kind: str, due_ms: int | None, payload: dict[str, Any]
    ) -> None:
        self._q(
            "INSERT OR REPLACE INTO pending VALUES (?, ?, ?, ?)",
            (run_id, kind, due_ms, json.dumps(payload)),
        )

    def clear_pending(self, run_id: str) -> None:
        self._q("DELETE FROM pending WHERE run_id=?", (run_id,))

    def pending(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = (
            self._q("SELECT * FROM pending WHERE run_id=?", (run_id,))
            if run_id
            else self._q("SELECT * FROM pending ORDER BY due_ms")
        )
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # ---- executor jobs ---------------------------------------------------------

    def add_job(
        self, job_id: str, run_id: str, kind: str, payload: dict[str, Any], at_ms: int
    ) -> None:
        self._q(
            "INSERT OR IGNORE INTO jobs (job_id, run_id, kind, payload, created_ms)"
            " VALUES (?, ?, ?, ?, ?)",
            (job_id, run_id, kind, json.dumps(payload), at_ms),
        )

    def claim_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_ms, job_id LIMIT ?",
                (limit,),
            ).fetchall()
            for r in rows:
                self.conn.execute("UPDATE jobs SET status='claimed' WHERE job_id=?", (r["job_id"],))
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def finish_job(self, job_id: str, result: dict[str, Any]) -> str | None:
        """Record a result; returns the job's run id (None for an unknown job)."""
        rows = self._q("SELECT run_id FROM jobs WHERE job_id=?", (job_id,))
        if not rows:
            return None
        self._q(
            "UPDATE jobs SET status='done', result=? WHERE job_id=?", (json.dumps(result), job_id)
        )
        return str(rows[0]["run_id"])

    def job_results(self, job_ids: list[str]) -> dict[str, dict[str, Any]] | None:
        """All results, or None while any job is unfinished."""
        out = {}
        for jid in job_ids:
            rows = self._q("SELECT status, result FROM jobs WHERE job_id=?", (jid,))
            if not rows or rows[0]["status"] != "done":
                return None
            out[jid] = json.loads(rows[0]["result"])
        return out

    # ---- approvals -------------------------------------------------------------

    def add_approval(self, req: ApprovalRequest) -> None:
        self._q(
            "INSERT OR IGNORE INTO approvals (approval_id, run_id, request) VALUES (?, ?, ?)",
            (req.approval_id, req.run_id, req.model_dump_json()),
        )

    def approval(self, approval_id: str) -> tuple[ApprovalRequest, str | None] | None:
        rows = self._q(
            "SELECT request, decision FROM approvals WHERE approval_id=?", (approval_id,)
        )
        if not rows:
            return None
        return ApprovalRequest.model_validate_json(rows[0]["request"]), rows[0]["decision"]

    def decide(self, approval_id: str, decision: str, grant: Grant | None) -> None:
        self._q(
            "UPDATE approvals SET decision=?, grant_json=? WHERE approval_id=?",
            (decision, grant.model_dump_json() if grant else None, approval_id),
        )

    def approvals(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = (
            self._q("SELECT * FROM approvals WHERE run_id=? ORDER BY approval_id", (run_id,))
            if run_id
            else self._q("SELECT * FROM approvals ORDER BY approval_id")
        )
        return [{**dict(r), "request": json.loads(r["request"])} for r in rows]

    # ---- model calls -----------------------------------------------------------

    def log_llm(
        self,
        run_id: str,
        node: str,
        task: str,
        model: str,
        ok: bool,
        reason: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        request: dict[str, Any],
    ) -> None:
        self._q(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                node,
                task,
                model,
                int(ok),
                reason,
                input_tokens,
                output_tokens,
                latency_ms,
                json.dumps(request, sort_keys=True),
            ),
        )

    def llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM llm_calls WHERE run_id=?", (run_id,))]
