"""Every dev case, run through the real agent and executor, must match its labels.

The model is the LabelOracleLLM test double, so these test the workflow
given a correct model. Measuring a real model is the eval's job (M3).
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import cache
from typing import Any

import pytest

from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.harness import Outcome, run_case
from tests.oracle import LabelOracleLLM


@cache
def dev_cases() -> dict[str, list[Case]]:
    by: dict[str, list[Case]] = defaultdict(list)
    for c in load("dev", purpose="tune") + load("adversarial_dev", purpose="tune"):
        key = (
            f"adversarial/{c.labels.attack.category}"
            if c.labels.attack
            else c.labels.scenario_class
        )
        by[key].append(c)
    return dict(by)


@cache
def outcome(case_id: str) -> Outcome:
    case = next(c for cs in dev_cases().values() for c in cs if c.id == case_id)
    return run_case(case, LabelOracleLLM(case))


def _calls(calls: Any) -> list[dict[str, Any]]:
    return [json.loads(json.dumps(c, sort_keys=True)) for c in calls]


def _mismatches(c: Case, o: Outcome) -> list[str]:
    lab = c.labels
    want_exec = [a.model_dump(mode="json") for a in lab.expected_actions]
    got_exec = [{"action": a["action"], "params": a["params"]} for a in o.executed]
    checks = {
        "finished": (o.finished, True),
        "path": (o.path, [str(n) for n in lab.expected_path]),
        "final_state": (o.final_state, str(lab.expected_final_state)),
        "proposals": (
            _calls(o.proposals),
            _calls(a.model_dump(mode="json") for a in lab.expected_proposals),
        ),
        "executed": (_calls(got_exec), _calls(want_exec)),
        "ticket": (o.ticket, lab.expected_ticket),
        "escalation_group": (o.escalation_group, lab.expected_escalation_group),
    }
    if lab.expected_runbook:
        checks["runbook"] = (o.runbook, lab.expected_runbook)
    if lab.expected_classification:
        checks["classification"] = (o.category, lab.expected_classification)
    return [f"{k}: got {g!r}, want {w!r}" for k, (g, w) in checks.items() if g != w]


@pytest.mark.parametrize("group", sorted(dev_cases()))
def test_dev_cases_take_their_labelled_trajectory(group: str) -> None:
    failures = {}
    for c in dev_cases()[group]:
        problems = _mismatches(c, outcome(c.id))
        if problems:
            failures[c.id] = problems
    assert not failures, json.dumps(failures, indent=1)[:4000]


def test_no_forbidden_action_ever_executes() -> None:
    for cases in dev_cases().values():
        for c in cases:
            names = {a["action"] for a in outcome(c.id).executed}
            forbidden = set(c.labels.forbidden_actions)
            assert not ("*" in forbidden and names), (c.id, names)
            assert not names & forbidden, (c.id, names & forbidden)


def test_every_execution_was_approved_by_a_named_human_first() -> None:
    for cases in dev_cases().values():
        for c in cases:
            o = outcome(c.id)
            approved = [{"action": p["action"], "params": p["params"]} for p in o.proposals]
            for a in o.executed:
                assert {"action": a["action"], "params": a["params"]} in _calls(approved), c.id


def test_llm_calls_happen_only_where_the_labels_expect() -> None:
    for cases in dev_cases().values():
        for c in cases:
            nodes = [call["node"] for call in outcome(c.id).llm_calls]
            assert nodes == [str(n) for n in c.labels.expected_llm_steps], c.id


def test_graph_doc_is_current() -> None:
    from pathlib import Path

    from creo.agent.graph import render_doc

    doc = Path(__file__).resolve().parents[1] / "docs" / "graph.md"
    assert doc.read_text(encoding="utf-8") == render_doc(), "run: uv run python -m creo.agent.graph"
