"""One labelled case: the alert to post, the scripted world around it, and its labels.

A case has one *primary* alert, the one the labels describe. Some cases also
post setup alerts before it (`world.prior_events`, e.g. the open alert that a
duplicate repeats) and follow-ups after it (`world.followups`, e.g. a
Recovered webhook). Only the primary alert is scored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from creo.catalog import Action, Diagnostic, FinalState, Node

Split = Literal["dev", "test", "adversarial_dev", "adversarial_test"]
SPLITS: tuple[Split, ...] = ("dev", "test", "adversarial_dev", "adversarial_test")
# Splits that tuning code (prompts, screen patterns, rubrics) may read.
TUNABLE_SPLITS: frozenset[Split] = frozenset({"dev", "adversarial_dev"})

Approval = Literal["approve", "deny", "timeout"]
ExecResult = Literal["ok", "error"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionCall(_Strict):
    action: Action
    params: dict[str, str | int]


class Event(_Strict):
    """One HTTP POST to the agent's webhook."""

    # Seconds relative to the primary alert's post (negative: before it).
    at_s: int
    # The JSON payload; None when the body is not a JSON object (see raw_body).
    payload: dict[str, Any] | None = None
    # The literal body, set only when it can't be expressed as `payload`
    # (invalid JSON, a JSON array, an oversized body).
    raw_body: str | None = None


class World(_Strict):
    """How the simulated environment answers the agent during this case."""

    # The monitor reports OK from this many seconds after the primary alert
    # (0: already OK at triage). None: it never recovers by itself.
    ok_from_s: int | None
    # Read-only diagnostic outputs, as the executor would return them.
    diagnostics: dict[Diagnostic, Any] = Field(default_factory=dict)
    # "script": approval requests are answered by `approvals`, in order (a
    # request beyond the list times out). "approve_all": a careless approver
    # approves any request, every execution succeeds and fixes the monitor.
    # The attacks use approve_all, so an attack that gets as far as a
    # proposal shows up as an unsafe execution instead of a denial.
    approval_policy: Literal["script", "approve_all"] = "script"
    approvals: tuple[Approval, ...] = ()
    approver: str = ""
    # One entry per executed fix: the executor's result, and whether the
    # monitor is OK when the agent verifies afterwards.
    executor: tuple[ExecResult, ...] = ()
    fix_effect: tuple[bool, ...] = ()
    prior_events: tuple[Event, ...] = ()
    followups: tuple[Event, ...] = ()


class Attack(_Strict):
    category: str
    family: str
    # What the attack tries to make the agent do (for reporting near-misses).
    targets: tuple[str, ...]


class Labels(_Strict):
    scenario_class: str
    adversarial: bool
    attack: Attack | None = None
    # The runbook select_runbook must pick; None when it must pick none or is
    # not reached.
    expected_runbook: str | None
    # The category classify must return; None for runbooks without judgement.
    expected_classification: str | None
    # Every node visited, in order, loops unrolled. Empty: refused at ingress.
    expected_path: tuple[Node, ...]
    expected_llm_steps: tuple[Node, ...]
    expected_proposals: tuple[ActionCall, ...]
    # Executions that should happen (each one approved first).
    expected_actions: tuple[ActionCall, ...]
    # Action names that must never execute in this case; "*" means any action.
    forbidden_actions: tuple[str, ...]
    expected_final_state: FinalState
    expected_ticket: Literal["create", "link", "none"]
    expected_escalation_group: str | None


class Case(_Strict):
    id: str
    split: Split
    # When the primary alert is posted, in dataset time (epoch ms). A replayer
    # shifts every timestamp in the case by the same amount.
    epoch_ms: int
    alert: Event
    world: World
    labels: Labels
