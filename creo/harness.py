"""Run one dataset case through the real agent and executor, in-process.

Everything is real except the outside world: the graph, checkpointer, store,
screen, executor and approval checks are the production code paths. The
monitoring system, hosts and approver follow the case's world script, and
the clock is simulated, so a two-hour wait takes milliseconds.

The approver answers only the primary run's requests. Setup runs (earlier
alerts in duplicate, correlated and flapping cases) get no answers, so their
requests time out.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from creo.agent.env import Env, SimClock
from creo.agent.llm import LLMClient
from creo.agent.runtime import Agent
from creo.agent.store import Store
from creo.config import load_inventory, load_policy, load_runbooks
from creo.dataset.generate import body_bytes
from creo.dataset.schema import Case, Event
from creo.executor.core import Executor
from creo.sim.world import CaseWorld
from creo.tickets.store import SqliteTickets

HARNESS_KEY = b"harness-approval-key"
MAX_STEPS = 500
# Stop simulating this long after the primary alert if the run never ends.
HORIZON_MS = 6 * 3_600_000


@dataclass
class Outcome:
    case_id: str
    run_id: str | None
    rejected_reason: str = ""
    finished: bool = False
    path: list[str] = field(default_factory=list)
    final_state: str | None = None
    runbook: str | None = None
    category: str | None = None
    reason: str | None = None
    flags: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    ticket: str = "none"
    escalation_group: str | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, str] | None = None


class CaseRun:
    def __init__(self, case: Case, llm: LLMClient) -> None:
        self.case = case
        first = min([0, *(e.at_s for e in case.world.prior_events)])
        self.clock = SimClock(case.epoch_ms + first * 1000)
        self.world = CaseWorld(case, self.clock)
        self.store = Store(":memory:")
        self.tickets = SqliteTickets(":memory:", self.clock)
        policy = load_policy()
        self.env = Env(
            inventory=load_inventory(),
            runbooks=load_runbooks(),
            policy=policy,
            store=self.store,
            tickets=self.tickets,
            monitoring=self.world,
            llm=llm,
            clock=self.clock,
            approval_key=HARNESS_KEY,
        )
        self.agent = Agent(
            self.env, SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
        )
        self.executor = Executor(
            inventory=self.env.inventory,
            policy=policy,
            approval_key=HARNESS_KEY,
            backend=self.world,
            clock=self.clock,
            killswitch=self.store.killswitch,
        )
        self.primary: str | None = None
        self.answers = list(case.world.approvals)
        self.handled: set[str] = set()
        self.rejected = ""

    def _post(self, ev: Event, primary: bool) -> None:
        receipt = self.agent.receive(body_bytes(ev))
        if primary:
            self.primary = receipt.run_id
            self.rejected = receipt.reason if receipt.status == "rejected" else ""

    def _answer_approvals(self) -> bool:
        """Answer the primary run's next new approval request. True if one was answered.

        A "timeout" answer means nobody answers: the request is left alone and
        expires through the agent's own timer, so events scheduled before the
        expiry (a Recovered webhook, say) still arrive in order.
        """
        for req in self.agent.pending_approvals():
            if req["run_id"] != self.primary or req["approval_id"] in self.handled:
                continue
            self.handled.add(req["approval_id"])
            w = self.case.world
            if w.approval_policy == "approve_all":
                answer = "approve"
            else:
                answer = self.answers.pop(0) if self.answers else "timeout"
            if answer == "timeout":
                return False
            approver = w.approver or "a.mehta"
            self.agent.decide(req["approval_id"], answer, approver)  # type: ignore[arg-type]
            return True
        return False

    def _settle(self, until_ms: int | None) -> None:
        """Run jobs and approvals, and advance through timers up to until_ms (None: until done)."""
        for _ in range(MAX_STEPS):
            jobs = self.agent.claim_jobs()
            if jobs:
                for job in jobs:
                    self.agent.complete_job(job["job_id"], self.executor.run(job))
                continue
            if self._answer_approvals():
                continue
            if until_ms is None and (self.primary is None or self.agent.finished(self.primary)):
                return
            due = self.agent.next_due_ms()
            limit = until_ms if until_ms is not None else self.case.epoch_ms + HORIZON_MS
            if due is None or due > limit:
                if until_ms is not None:
                    self.clock.advance_to(until_ms)
                return
            self.clock.advance_to(due)
            self.agent.tick()

    def run(self) -> Outcome:
        w = self.case.world
        timeline = [(e.at_s, 0, e) for e in w.prior_events]
        timeline += [(0, 1, self.case.alert)] + [(e.at_s, 2, e) for e in w.followups]
        for at_s, order, ev in sorted(timeline, key=lambda t: (t[0], t[1])):
            self._settle(self.case.epoch_ms + at_s * 1000)
            self._post(ev, primary=order == 1)
        self._settle(None)
        return self.outcome()

    def outcome(self) -> Outcome:
        out = Outcome(case_id=self.case.id, run_id=self.primary, rejected_reason=self.rejected)
        if self.primary is None:
            out.finished, out.final_state = True, "rejected"
            return out
        s = self.agent.state(self.primary)
        out.finished = self.agent.finished(self.primary)
        out.path = [str(p) for p in s.get("path", [])]
        out.final_state = str(s["final_state"]) if s.get("final_state") else None
        out.runbook = s.get("runbook")
        out.category = s.get("category")
        out.reason = s.get("reason")
        out.flags = list(s.get("flags", []))
        out.escalation_group = s.get("escalation_group")
        out.summary = s.get("summary")
        out.proposals = [
            {"action": a["request"]["action"], "params": a["request"]["params"]}
            for a in self.store.approvals(self.primary)
        ]
        for a in self.executor.attempts(self.primary):
            if a["kind"] != "action":
                continue
            rec = {
                "action": a["action"],
                "params": a["params"],
                "status": a["status"],
                "code": a["code"],
            }
            (out.refused if a["status"] == "refused" else out.executed).append(rec)
        ops = [o for o in self.tickets.ops() if o["source_run"] == self.primary]
        tid = s.get("ticket_id")
        if any(o["op"] == "create" for o in ops):
            out.ticket = "create"
        elif tid:
            out.ticket = "link"
        out.llm_calls = self.store.llm_calls(self.primary)
        return out


def run_case(case: Case, llm: LLMClient) -> Outcome:
    return CaseRun(case, llm).run()
