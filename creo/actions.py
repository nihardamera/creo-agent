"""Typed parameters for every allowlisted action, and the inventory checks.

The agent uses these to build a proposal; the executor uses the same models
to refuse anything malformed, whatever the agent sent. A parameter is either
a fixed-range number or an identifier that must exist on the target host.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from creo.catalog import Action
from creo.config import Host, Inventory

# Identifiers: no spaces, wildcards, separators or traversal.
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")]
HostName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9\-]{0,62}$")]


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: HostName


class HostOnly(Params):
    pass


class ServiceParams(Params):
    service_name: Name


class AppPoolParams(Params):
    app_pool: Name


class ClearTempParams(Params):
    drive: Annotated[str, Field(pattern=r"^[A-Z]:$")]
    older_than_days: Annotated[int, Field(ge=1, le=30)]


class RotateIisLogsParams(Params):
    older_than_days: Annotated[int, Field(ge=7, le=90)]


class KillProcessParams(Params):
    process_name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{1,60}\.exe$")]
    pid: Annotated[int, Field(ge=1, le=4_194_304)]


class TaskParams(Params):
    task_name: Name


class UnitParams(Params):
    unit: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9@._\-]{0,62}\.service$")]


class VacuumParams(Params):
    max_size_mb: Annotated[int, Field(ge=100, le=2_000)]


class ConsumerParams(Params):
    consumer: Name


PARAMS: dict[Action, type[Params]] = {
    Action.RESTART_WINDOWS_SERVICE: ServiceParams,
    Action.START_APP_POOL: AppPoolParams,
    Action.RECYCLE_APP_POOL: AppPoolParams,
    Action.RESTART_IIS: HostOnly,
    Action.CLEAR_TEMP_FILES: ClearTempParams,
    Action.ROTATE_IIS_LOGS: RotateIisLogsParams,
    Action.CLEAR_WU_CACHE: HostOnly,
    Action.KILL_PROCESS: KillProcessParams,
    Action.RERUN_SCHEDULED_TASK: TaskParams,
    Action.FLUSH_DNS_CACHE: HostOnly,
    Action.RESTART_SYSTEMD_UNIT: UnitParams,
    Action.VACUUM_JOURNAL: VacuumParams,
    Action.CLEAN_PACKAGE_CACHE: HostOnly,
    Action.RESTART_QUEUE_CONSUMER: ConsumerParams,
}

# Actions that only make sense on one OS.
WINDOWS_ONLY = frozenset(
    {
        Action.RESTART_WINDOWS_SERVICE,
        Action.START_APP_POOL,
        Action.RECYCLE_APP_POOL,
        Action.RESTART_IIS,
        Action.CLEAR_TEMP_FILES,
        Action.ROTATE_IIS_LOGS,
        Action.CLEAR_WU_CACHE,
        Action.KILL_PROCESS,
        Action.RERUN_SCHEDULED_TASK,
        Action.FLUSH_DNS_CACHE,
    }
)


class ActionRefused(ValueError):
    """An action or its parameters failed a check. `code` is stable for tests and logs."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def parse(action: str, params: dict[str, Any]) -> tuple[Action, Params]:
    """Allowlist and type-check an action. Raises ActionRefused."""
    try:
        act = Action(action)
    except ValueError:
        raise ActionRefused("not_allowlisted", f"{action!r} is not an allowlisted action") from None
    try:
        return act, PARAMS[act].model_validate(params)
    except ValidationError as e:
        raise ActionRefused("bad_params", str(e.errors(include_url=False))) from None


def check_inventory(action: Action, params: Params, inventory: Inventory) -> Host:
    """The target host exists and has every resource the parameters name."""
    host = inventory.host(params.host)
    if host is None:
        raise ActionRefused("unknown_host", params.host)
    if action in WINDOWS_ONLY and host.os != "windows":
        raise ActionRefused("wrong_os", f"{action} needs windows, {host.name} runs {host.os}")
    if action not in WINDOWS_ONLY and host.os != "linux":
        raise ActionRefused("wrong_os", f"{action} needs linux, {host.name} runs {host.os}")
    match params:
        case ServiceParams(service_name=s) if s not in host.services:
            raise ActionRefused("unknown_resource", f"service {s} is not on {host.name}")
        case AppPoolParams(app_pool=p) if p not in host.app_pools:
            raise ActionRefused("unknown_resource", f"app pool {p} is not on {host.name}")
        case ClearTempParams(drive=d) if d not in host.drives:
            raise ActionRefused("unknown_resource", f"drive {d} is not on {host.name}")
        case KillProcessParams(process_name=n) if n not in {p.image for p in host.processes}:
            raise ActionRefused("unknown_resource", f"process {n} does not run on {host.name}")
        case TaskParams(task_name=t) if t not in host.tasks:
            raise ActionRefused("unknown_resource", f"task {t} is not on {host.name}")
        case UnitParams(unit=u) if u not in host.units:
            raise ActionRefused("unknown_resource", f"unit {u} is not on {host.name}")
        case ConsumerParams(consumer=c) if c not in host.consumers:
            raise ActionRefused("unknown_resource", f"consumer {c} is not on {host.name}")
    if action is Action.ROTATE_IIS_LOGS and host.role != "web":
        raise ActionRefused("unknown_resource", f"{host.name} has no IIS logs")
    return host
