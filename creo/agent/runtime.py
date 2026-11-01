"""The runtime around the graph: ingress, waking paused runs, approvals, executor jobs.

One `Agent` owns a graph compiled with a SQLite checkpointer and a Store.
Every paused run is recorded in the store with what it waits for:
- "wait": a timer (due_ms), or a Recovered webhook, whichever comes first;
- "approval": a human decision, or expiry at due_ms;
- "jobs": executor results.

A fresh Agent built on the same files carries on from there, which is what
makes a run survive a restart.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import ValidationError

from creo import approvals
from creo.agent.env import Env
from creo.agent.graph import compile_graph
from creo.catalog import Transition
from creo.webhook import AlertWebhook


@dataclass(frozen=True)
class Receipt:
    status: Literal["accepted", "rejected", "woke", "recorded"]
    run_id: str | None = None
    reason: str = ""


class ApprovalError(ValueError):
    pass


class Agent:
    def __init__(self, env: Env, checkpointer: Any) -> None:
        self.env = env
        self.graph = compile_graph(checkpointer)
        self.lock = threading.RLock()

    # ---- driving the graph ---------------------------------------------------------

    def _config(self, run_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": run_id}}

    def _drive(self, run_id: str, inp: Any) -> None:
        result = self.graph.invoke(inp, self._config(run_id), context=self.env, durability="sync")
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            value = interrupts[0].value
            self.env.store.set_pending(run_id, value["kind"], value.get("due_ms"), value)
        else:
            self.env.store.clear_pending(run_id)

    def _resume(self, run_id: str, value: Any) -> None:
        self.env.store.clear_pending(run_id)
        self._drive(run_id, Command(resume=value))

    def state(self, run_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(run_id)).values)

    def finished(self, run_id: str) -> bool:
        snap = self.graph.get_state(self._config(run_id))
        return not snap.next and not self.env.store.pending(run_id)

    # ---- ingress -------------------------------------------------------------------

    def receive(self, body: bytes) -> Receipt:
        """Handle one webhook body. Refusals carry a reason and never start a run."""
        if len(body) > self.env.policy.max_payload_bytes:
            return Receipt("rejected", reason="payload_too_large")
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Receipt("rejected", reason="invalid_json")
        if not isinstance(data, dict):
            return Receipt("rejected", reason="not_an_object")
        try:
            alert = AlertWebhook.model_validate(data)
        except ValidationError:
            return Receipt("rejected", reason="schema")
        with self.lock:
            now = self.env.clock.now_ms()
            store = self.env.store
            store.add_transition(alert.alert_id, alert.hostname, alert.transition, now)
            if alert.transition is Transition.RECOVERED:
                run = store.find_open_run(alert.alert_id, alert.hostname)
                if run:
                    return self._recovered(run, alert)
            run_id = store.new_run_id()
            empty: dict[str, Any] = {"path": [], "flags": [], "actions": []}
            self._drive(run_id, {"run_id": run_id, "alert": data, "received_ms": now, **empty})
            return Receipt("accepted", run_id=run_id)

    def _recovered(self, run_id: str, alert: AlertWebhook) -> Receipt:
        """A Recovered webhook short-circuits whatever wait the run is in."""
        self.env.store.add_event(alert.id, run_id, self.env.clock.now_ms())
        pending = self.env.store.pending(run_id)
        kind = pending[0]["kind"] if pending else None
        if kind == "wait":
            self._resume(run_id, {"reason": "recovered"})
            return Receipt("woke", run_id=run_id)
        if kind == "approval":
            approval_id = pending[0]["payload"]["approval_id"]
            self.env.store.decide(approval_id, "withdrawn", None)
            self._resume(run_id, {"decision": "withdrawn"})
            return Receipt("woke", run_id=run_id)
        return Receipt("recorded", run_id=run_id, reason="run_not_waiting")

    # ---- timers -------------------------------------------------------------------

    def tick(self) -> int:
        """Resume every run whose timer or approval expiry has passed. Returns how many."""
        n = 0
        with self.lock:
            now = self.env.clock.now_ms()
            for p in self.env.store.pending():
                if p["due_ms"] is None or p["due_ms"] > now:
                    continue
                if p["kind"] == "wait":
                    self._resume(p["run_id"], {"reason": "timer"})
                elif p["kind"] == "approval":
                    self.env.store.decide(p["payload"]["approval_id"], "timeout", None)
                    self._resume(p["run_id"], {"decision": "timeout"})
                else:
                    continue
                n += 1
        return n

    def next_due_ms(self) -> int | None:
        dues = [p["due_ms"] for p in self.env.store.pending() if p["due_ms"] is not None]
        return min(dues) if dues else None

    # ---- approvals --------------------------------------------------------------------

    def pending_approvals(self) -> list[dict[str, Any]]:
        out = []
        for p in self.env.store.pending():
            if p["kind"] == "approval":
                found = self.env.store.approval(p["payload"]["approval_id"])
                if found:
                    out.append(found[0].model_dump())
        return out

    def decide(self, approval_id: str, decision: approvals.Decision, approver: str) -> None:
        """A named human's decision. Signs a grant bound to that exact request."""
        with self.lock:
            found = self.env.store.approval(approval_id)
            if found is None:
                raise ApprovalError("unknown approval")
            req, previous = found
            if previous is not None:
                raise ApprovalError(f"already decided: {previous}")
            pending = self.env.store.pending(req.run_id)
            if not pending or pending[0]["kind"] != "approval":
                raise ApprovalError("run is not waiting for this approval")
            if approver not in self.env.policy.approvers:
                raise ApprovalError(f"{approver!r} is not a named approver")
            now = self.env.clock.now_ms()
            if now >= req.expires_ms:
                raise ApprovalError("expired")
            grant = approvals.sign(self.env.approval_key, req, decision, approver, now)
            self.env.store.decide(approval_id, decision, grant)
            self._resume(req.run_id, {"decision": decision, "grant": grant.model_dump()})

    # ---- executor jobs (the executor polls; nothing calls in to it) ------------------

    def claim_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            return self.env.store.claim_jobs(limit)

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        with self.lock:
            run_id = self.env.store.finish_job(job_id, result)
            if run_id is None:
                return
            pending = self.env.store.pending(run_id)
            if not pending or pending[0]["kind"] != "jobs":
                return
            results = self.env.store.job_results(pending[0]["payload"]["job_ids"])
            if results is not None:
                self._resume(run_id, results)

    # ---- the kill switch ----------------------------------------------------------

    def set_killswitch(self, on: bool) -> None:
        self.env.store.set_killswitch(on)
