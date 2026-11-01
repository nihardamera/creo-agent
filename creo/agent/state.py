"""The typed state of one run, checkpointed after every node."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RunState(TypedDict, total=False):
    run_id: str
    alert: dict[str, Any]  # the validated webhook payload
    received_ms: int

    # Appended to by every node (reducer: list concatenation).
    path: Annotated[list[str], operator.add]
    flags: Annotated[list[str], operator.add]
    actions: Annotated[list[dict[str, Any]], operator.add]

    # Routing and outcome.
    route: str  # the next node
    reason: str  # why the run stopped or escalated (a fixed code)
    final_state: str
    link_run: str | None  # duplicate/correlated: the open run this alert joins
    opened: bool  # this run owns the monitor until it finishes

    # Facts established along the way.
    ticket_id: str | None
    host: str
    monitor_id: str
    runbook: str | None
    category: str | None
    cited: list[str]  # evidence ids the classification cited
    diagnostics: dict[str, Any]
    wait_anchor_ms: int
    waits: int
    fix_index: int
    proposal: dict[str, Any] | None
    approval_id: str | None
    escalation_group: str | None
    summary: dict[str, str] | None
