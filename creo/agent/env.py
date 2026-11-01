"""What a running graph can reach: config, stores, the outside services, a clock.

Passed to every node as LangGraph runtime context, never checkpointed, so no
key or client ever lands in a checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from creo.agent.llm import LLMClient
from creo.agent.store import Store
from creo.config import Inventory, Policy, Runbook

MonitorStatus = Literal["OK", "Alert", "No Data"]


class Clock(Protocol):
    def now_ms(self) -> int: ...


class WallClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class SimClock:
    """A clock the harness moves by hand, so a two-hour wait takes no real time."""

    def __init__(self, start_ms: int) -> None:
        self.t = start_ms

    def now_ms(self) -> int:
        return self.t

    def advance_to(self, t_ms: int) -> None:
        self.t = max(self.t, t_ms)


class Monitoring(Protocol):
    """The monitoring system's API (a Datadog stand-in): current monitor status."""

    def status(self, monitor_id: str, host: str) -> MonitorStatus: ...


class Tickets(Protocol):
    """The ticketing system (a Jira Service Management stand-in)."""

    def create(
        self, *, title: str, body: str, host: str, priority: str, source_run: str
    ) -> str: ...

    def comment(self, ticket_id: str, text: str) -> None: ...

    def transition(self, ticket_id: str, status: str, *, group: str | None = None) -> None: ...


@dataclass
class Env:
    inventory: Inventory
    runbooks: dict[str, Runbook]
    policy: Policy
    store: Store
    tickets: Tickets
    monitoring: Monitoring
    llm: LLMClient
    clock: Clock
    approval_key: bytes
