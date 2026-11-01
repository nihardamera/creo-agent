"""A run paused in a wait or at the approval gate survives the agent process restarting.

The agent's store and LangGraph checkpoints live in SQLite files. "Restart"
here means dropping every object and connection and building a fresh Agent
on the same files, which is what a new process would do.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from creo.agent.env import Env, SimClock
from creo.agent.runtime import Agent
from creo.agent.store import Store
from creo.config import load_inventory, load_policy, load_runbooks
from creo.dataset.generate import body_bytes
from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.executor.core import Executor
from creo.harness import HARNESS_KEY
from creo.sim.world import CaseWorld
from creo.tickets.store import SqliteTickets
from tests.oracle import LabelOracleLLM


class Process:
    """One 'agent process': everything built fresh from the files in `d`."""

    def __init__(self, d: Path, case: Case, clock: SimClock, world: CaseWorld) -> None:
        self.cp = sqlite3.connect(d / "checkpoints.db", check_same_thread=False)
        self.store = Store(d / "agent.db")
        env = Env(
            inventory=load_inventory(), runbooks=load_runbooks(), policy=load_policy(),
            store=self.store, tickets=SqliteTickets(d / "tickets.db", clock), monitoring=world,
            llm=LabelOracleLLM(case), clock=clock, approval_key=HARNESS_KEY,
        )  # fmt: skip
        self.agent = Agent(env, SqliteSaver(self.cp))
        self.executor = Executor(
            inventory=env.inventory, policy=env.policy, approval_key=HARNESS_KEY,
            backend=world, clock=clock, db=d / "executor.db",
        )  # fmt: skip

    def run_jobs(self) -> None:
        while jobs := self.agent.claim_jobs():
            for j in jobs:
                self.agent.complete_job(j["job_id"], self.executor.run(j))

    def stop(self) -> None:
        self.cp.close()
        self.store.conn.close()


def _case(cls: str) -> Case:
    return next(c for c in load("dev", purpose="tune") if c.labels.scenario_class == cls)


def test_a_wait_survives_a_restart(tmp_path: Path) -> None:
    case = _case("recovers_on_recheck")
    clock = SimClock(case.epoch_ms)
    world = CaseWorld(case, clock)
    p1 = Process(tmp_path, case, clock, world)
    run_id = p1.agent.receive(body_bytes(case.alert)).run_id
    assert run_id
    p1.run_jobs()
    assert p1.store.pending(run_id)[0]["kind"] == "wait"
    p1.stop()  # the agent dies mid-wait

    p2 = Process(tmp_path, case, clock, world)
    for _ in range(20):
        due = p2.agent.next_due_ms()
        if due is None:
            break
        clock.advance_to(due)
        p2.agent.tick()
        p2.run_jobs()
    state = p2.agent.state(run_id)
    assert state["path"] == [str(n) for n in case.labels.expected_path]
    assert state["final_state"] == "resolved"


def test_an_approval_wait_survives_a_restart(tmp_path: Path) -> None:
    case = _case("fix_succeeds")
    clock = SimClock(case.epoch_ms)
    world = CaseWorld(case, clock)
    p1 = Process(tmp_path, case, clock, world)
    run_id = p1.agent.receive(body_bytes(case.alert)).run_id
    assert run_id
    for _ in range(20):
        p1.run_jobs()
        if p1.agent.pending_approvals():
            break
        due = p1.agent.next_due_ms()
        assert due is not None
        clock.advance_to(due)
        p1.agent.tick()
    approval_id = p1.agent.pending_approvals()[0]["approval_id"]
    p1.stop()  # the agent dies while a human is deciding

    p2 = Process(tmp_path, case, clock, world)
    assert [a["approval_id"] for a in p2.agent.pending_approvals()] == [approval_id]
    p2.agent.decide(approval_id, "approve", case.world.approver)
    p2.run_jobs()
    state = p2.agent.state(run_id)
    assert state["final_state"] == "resolved"
    assert state["path"] == [str(n) for n in case.labels.expected_path]
    assert [a["status"] for a in p2.executor.attempts(run_id) if a["kind"] == "action"] == ["ok"]
