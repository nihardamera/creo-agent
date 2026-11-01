"""Paths the dataset doesn't cover: bad model output, the kill switch, recovery during approval."""

from __future__ import annotations

import json

import pytest

from creo.agent.runtime import ApprovalError
from creo.dataset.io import load
from creo.dataset.schema import Case, Event
from creo.harness import CaseRun
from tests.oracle import LabelOracleLLM, ScriptedLLM


def _case(cls: str, *, runbook: str | None = None, judgement: bool | None = None) -> Case:
    for c in load("dev", purpose="tune"):
        if c.labels.scenario_class != cls:
            continue
        if runbook and c.labels.expected_runbook != runbook:
            continue
        if judgement is not None and bool(c.labels.expected_classification) != judgement:
            continue
        return c
    raise LookupError(cls)


def test_invalid_classification_escalates_with_a_template_note() -> None:
    c = _case("fix_succeeds", judgement=True)
    run = CaseRun(c, ScriptedLLM("not json", "also not json"))
    o = run.run()
    assert o.path[-3:] == ["classify", "escalate", "finalize"]
    assert o.reason == "classification_invalid" and o.executed == []
    assert o.summary and o.summary["source"] == "template"
    assert [call["ok"] for call in o.llm_calls] == [0, 0]


@pytest.mark.parametrize(
    ("reply", "flag"),
    [
        ({"category": "nuke", "evidence": ["p1"], "confidence": 1}, "llm_category_not_allowed"),
        ({"category": "PLACEHOLDER", "evidence": ["x9"], "confidence": 1}, "llm_unknown_evidence"),
        ({"category": "PLACEHOLDER", "evidence": [], "confidence": 1}, "llm_schema"),
        ({"category": "PLACEHOLDER", "evidence": ["proc1"], "confidence": 7}, "llm_schema"),
    ],
)
def test_model_output_outside_its_limits_is_never_used(reply: dict[str, object], flag: str) -> None:
    c = _case("fix_succeeds", judgement=True)
    if reply["category"] == "PLACEHOLDER":
        reply = {**reply, "category": c.labels.expected_classification}
    o = CaseRun(c, ScriptedLLM(json.dumps(reply))).run()
    assert o.reason == "classification_invalid" and flag in o.flags
    assert o.proposals == [] and o.executed == []


def test_kill_switch_stops_proposals() -> None:
    c = _case("fix_succeeds")
    run = CaseRun(c, LabelOracleLLM(c))
    run.agent.set_killswitch(True)
    o = run.run()
    assert o.reason == "killswitch" and o.proposals == [] and o.executed == []
    assert o.final_state == "escalated"


def test_recovery_while_waiting_for_approval_withdraws_the_request() -> None:
    c = _case("approval_timeout", runbook="win-service-down")
    # The monitor recovers 10 minutes after the proposal, before anyone answers.
    payload = dict(c.alert.payload or {})
    rb_wait_s = (5 + 5 + 5) * 60  # win-service-down: 5 min, then two rechecks 5 min apart
    payload.update(transition="Recovered", alert_type="success", id="9999999999")
    late = Event(at_s=rb_wait_s + 600, payload=payload)
    world = c.world.model_copy(update={"approvals": (), "followups": (late,)})
    case = c.model_copy(update={"world": world})
    o = CaseRun(case, LabelOracleLLM(case)).run()
    assert o.path[-3:] == ["approval", "resolve", "finalize"]
    assert o.reason == "recovered_webhook" and o.executed == []


def test_only_named_approvers_can_decide_and_only_once() -> None:
    c = _case("fix_succeeds")
    run = CaseRun(c, LabelOracleLLM(c))
    run.answers = []  # nobody answers, so the run stops at the approval gate
    run.case = c.model_copy(update={"world": c.world.model_copy(update={"approvals": ()})})
    for ev_at, ev in [(0, c.alert)]:
        run._settle(c.epoch_ms + ev_at)
        run._post(ev, primary=True)
    for _ in range(50):  # drive to the approval
        jobs = run.agent.claim_jobs()
        for j in jobs:
            run.agent.complete_job(j["job_id"], run.executor.run(j))
        if run.agent.pending_approvals():
            break
        due = run.agent.next_due_ms()
        assert due is not None
        run.clock.advance_to(due)
        run.agent.tick()
    req = run.agent.pending_approvals()[0]
    with pytest.raises(ApprovalError, match="not a named approver"):
        run.agent.decide(req["approval_id"], "approve", "ops-lead")
    run.agent.decide(req["approval_id"], "deny", "a.mehta")
    with pytest.raises(ApprovalError, match="already decided"):
        run.agent.decide(req["approval_id"], "approve", "a.mehta")
    assert run.agent.state(run.primary or "")["reason"] == "approval_denied"
