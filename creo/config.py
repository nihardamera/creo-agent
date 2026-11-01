"""Typed models and loaders for the three config files in config/.

- runbooks.yaml: one entry per runbook key; the monitor's tags select one.
- policy.yaml: global limits (staleness, flapping, approval expiry, tag schema).
- inventory.yaml: the fictional fleet, the source of truth for host facts.
  Alert tags are never trusted over it.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creo.catalog import Action, Diagnostic

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Plan(StrEnum):
    WAIT_THEN_FIX = "wait_then_fix"
    WAIT_THEN_ESCALATE = "wait_then_escalate"
    FIX_NOW = "fix_now"
    ESCALATE_NOW = "escalate_now"

    @property
    def waits(self) -> bool:
        return self in (Plan.WAIT_THEN_FIX, Plan.WAIT_THEN_ESCALATE)

    @property
    def fixes(self) -> bool:
        return self in (Plan.WAIT_THEN_FIX, Plan.FIX_NOW)


class FixStep(_Strict):
    action: Action
    # Only offered on hosts with one of these inventory roles (None: any role).
    roles: tuple[str, ...] | None = None
    # Fixed numeric parameters, e.g. older_than_days. Host facts are bound at run time.
    params: dict[str, int] = Field(default_factory=dict)


class Category(_Strict):
    plan: Plan
    fixes: tuple[FixStep, ...] = ()

    @model_validator(mode="after")
    def _fixes_match_plan(self) -> Self:
        if self.plan.fixes and not self.fixes:
            raise ValueError(f"plan {self.plan} needs at least one fix")
        if not self.plan.fixes and self.fixes:
            raise ValueError(f"plan {self.plan} must not list fixes")
        return self


class WaitPolicy(_Strict):
    minutes: int = Field(ge=0, le=120)
    rechecks: int = Field(ge=0, le=6)
    interval_minutes: int = Field(ge=0, le=60)


class Escalation(_Strict):
    group: str
    priority: Literal["P1", "P2", "P3", "P4"]


class Runbook(_Strict):
    key: str
    description: str
    monitor_types: tuple[str, ...]
    roles: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]
    # True: the classify step asks the model to pick a category from the
    # diagnostics. False: there is one category, "default", and no model call.
    judgement: bool
    wait: WaitPolicy
    categories: dict[str, Category]
    escalation: Escalation
    # kill_process may only target these process names.
    allowed_processes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.judgement and set(self.categories) != {"default"}:
            raise ValueError(f"{self.key}: a runbook without judgement has one 'default' category")
        if self.judgement and "default" in self.categories:
            raise ValueError(f"{self.key}: judgement categories must be named")
        if any(c.plan.waits for c in self.categories.values()) and (
            self.wait.minutes == 0 or self.wait.rechecks == 0
        ):
            raise ValueError(f"{self.key}: a waiting plan needs wait.minutes and wait.rechecks")
        kills = any(
            f.action is Action.KILL_PROCESS for c in self.categories.values() for f in c.fixes
        )
        if kills and not self.allowed_processes:
            raise ValueError(f"{self.key}: kill_process needs allowed_processes")
        return self

    def fixes_for(self, category: str, role: str) -> tuple[FixStep, ...]:
        """The ordered fixes this category offers on a host with this role."""
        return tuple(
            f for f in self.categories[category].fixes if f.roles is None or role in f.roles
        )


class Flapping(_Strict):
    transitions: int = Field(ge=2)
    window_minutes: int = Field(ge=1)


class Policy(_Strict):
    stale_after_minutes: int
    flapping: Flapping
    correlation_window_minutes: int
    approval_ttl_minutes: int
    max_actions_per_hour: int
    approvers: tuple[str, ...]
    allowed_link_domains: tuple[str, ...]
    max_payload_bytes: int
    no_fix_tiers: tuple[int, ...]
    default_escalation: Escalation
    security_escalation: Escalation
    allowed_tag_keys: tuple[str, ...]


class Process(_Strict):
    image: str
    # The service or app pool that owns the process.
    owner: str


class Host(_Strict):
    name: str
    site: str
    role: str
    os: Literal["windows", "linux"]
    tier: int = Field(ge=0, le=3)
    env: Literal["prod", "staging"]
    services: tuple[str, ...] = ()
    app_pools: tuple[str, ...] = ()
    drives: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    processes: tuple[Process, ...] = ()
    http_checks: tuple[str, ...] = ()
    certs: tuple[str, ...] = ()


class Inventory(_Strict):
    company: str
    domain: str
    hosts: tuple[Host, ...]

    def host(self, name: str) -> Host | None:
        return next((h for h in self.hosts if h.name == name), None)


class RunbookFile(_Strict):
    version: int
    runbooks: dict[str, Runbook]

    @model_validator(mode="before")
    @classmethod
    def _inject_keys(cls, data: dict[str, object]) -> dict[str, object]:
        books = data.get("runbooks")
        if isinstance(books, dict):
            data["runbooks"] = {k: {"key": k, **v} for k, v in books.items()}
        return data


def _read(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_runbooks(config_dir: Path = CONFIG_DIR) -> dict[str, Runbook]:
    return dict(RunbookFile.model_validate(_read(config_dir / "runbooks.yaml")).runbooks)


@cache
def load_policy(config_dir: Path = CONFIG_DIR) -> Policy:
    return Policy.model_validate(_read(config_dir / "policy.yaml"))


@cache
def load_inventory(config_dir: Path = CONFIG_DIR) -> Inventory:
    return Inventory.model_validate(_read(config_dir / "inventory.yaml"))
