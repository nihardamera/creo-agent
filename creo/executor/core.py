"""The executor: the simulated jump host, the only thing that touches a host.

It trusts nothing it is sent. For every job it checks, in order: the kill
switch, the allowlist, the parameter types, that the target and its resources
exist in its own copy of the inventory, and, for a change, that the grant is
signed, names a known approver, matches this exact action, parameters and
run, hasn't expired and hasn't been used before. Only then does it act.
Every attempt, refused or not, is logged.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from openinference.semconv.trace import OpenInferenceSpanKindValues as Kind
from openinference.semconv.trace import SpanAttributes as A
from pydantic import ValidationError

from creo import actions, approvals, tracing
from creo.agent.env import Clock
from creo.catalog import Diagnostic
from creo.config import Inventory, Policy

Outcome = Literal["ok", "error"]


class Backend(Protocol):
    """Where actions really happen. Here: a simulated fleet."""

    def diagnostic(self, host: str, name: str) -> Any: ...

    def execute(self, action: str, params: dict[str, Any]) -> tuple[Outcome, str]: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, at_ms INTEGER, job_id TEXT, run_id TEXT, kind TEXT,
    action TEXT, params TEXT, approval_id TEXT, approver TEXT, status TEXT, code TEXT
);
CREATE TABLE IF NOT EXISTS used_approvals (approval_id TEXT PRIMARY KEY, at_ms INTEGER);
"""


class Executor:
    def __init__(
        self,
        *,
        inventory: Inventory,
        policy: Policy,
        approval_key: bytes,
        backend: Backend,
        clock: Clock,
        db: str | Path = ":memory:",
        killswitch: Callable[[], bool] = lambda: False,
    ) -> None:
        self.inventory, self.policy, self.key = inventory, policy, approval_key
        self.backend, self.clock, self.killswitch = backend, clock, killswitch
        self.conn = sqlite3.connect(str(db), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def _log(
        self, job: dict[str, Any], status: str, code: str, grant: approvals.Grant | None = None
    ) -> dict[str, Any]:
        p = job.get("payload", {})
        self.conn.execute(
            "INSERT INTO attempts (at_ms, job_id, run_id, kind, action, params, approval_id,"
            " approver, status, code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.clock.now_ms(),
                job.get("job_id"),
                job.get("run_id"),
                job.get("kind"),
                p.get("action") or p.get("name"),
                json.dumps(p.get("params", {}), sort_keys=True),
                grant.approval_id if grant else None,
                grant.approver if grant else None,
                status,
                code,
            ),
        )
        return {"status": status, "code": code}

    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        kind, p = job.get("kind"), job.get("payload", {})
        name = str(p.get("action") or p.get("name"))
        attrs = {A.TOOL_NAME: name}
        with (
            self.lock,
            tracing.extracted(p.get("trace")),
            tracing.span(f"executor {kind}", Kind.TOOL, job.get("run_id"), **attrs) as sp,
        ):
            if kind == "diagnostic":
                result = self._diagnostic(job, p)
            elif kind == "action":
                result = self._action(job, p)
            else:
                result = self._log(job, "refused", "unknown_job_kind")
            sp.set_attribute("creo.executor.status", result["status"])
            sp.set_attribute("creo.executor.code", result["code"])
            return result

    def _diagnostic(self, job: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
        try:
            name = Diagnostic(str(p.get("name")))
        except ValueError:
            return self._log(job, "refused", "not_allowlisted")
        host = p.get("host")
        if not isinstance(host, str) or self.inventory.host(host) is None:
            return self._log(job, "refused", "unknown_host")
        output = self.backend.diagnostic(host, name.value)
        return {**self._log(job, "ok", "read_only"), "output": output}

    def _action(self, job: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
        if self.killswitch():
            return self._log(job, "refused", "killswitch")
        try:
            action, typed = actions.parse(str(p.get("action")), dict(p.get("params") or {}))
            actions.check_inventory(action, typed, self.inventory)
        except actions.ActionRefused as e:
            return self._log(job, "refused", e.code)
        try:
            grant = approvals.Grant.model_validate_json(p.get("grant") or "")
        except ValidationError:
            return self._log(job, "refused", "no_grant")
        params = typed.model_dump()
        problem = approvals.check(
            self.key,
            grant,
            run_id=str(p.get("run_id") or job.get("run_id")),
            action=action.value,
            params=params,
            now_ms=self.clock.now_ms(),
            approvers=self.policy.approvers,
        )
        if problem:
            return self._log(job, "refused", problem, grant)
        try:
            self.conn.execute(
                "INSERT INTO used_approvals VALUES (?, ?)", (grant.approval_id, self.clock.now_ms())
            )
        except sqlite3.IntegrityError:
            return self._log(job, "refused", "approval_already_used", grant)
        status, detail = self.backend.execute(action.value, params)
        return {**self._log(job, status, detail, grant), "detail": detail}

    def attempts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            rows = self.conn.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY seq", (run_id,)
            )
        else:
            rows = self.conn.execute("SELECT * FROM attempts ORDER BY seq")
        return [{**dict(r), "params": json.loads(r["params"])} for r in rows]
