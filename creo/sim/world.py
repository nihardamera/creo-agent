"""The simulated environment for one dataset case: monitoring and the hosts.

It answers exactly as the case's `world` script says: when the monitor reads
OK, what each diagnostic returns, whether each executed fix works. Monitors
other than the case's primary one stay alerting. Every host returns the
case's diagnostics (setup runs in multi-alert cases need plausible answers;
only the primary run is scored).
"""

from __future__ import annotations

from typing import Any

from creo.agent.env import Clock, MonitorStatus
from creo.catalog import Diagnostic
from creo.dataset.schema import Case
from creo.executor.core import Outcome


class CaseWorld:
    def __init__(self, case: Case, clock: Clock) -> None:
        self.case, self.clock = case, clock
        payload = case.alert.payload or {}
        self.monitor_id = str(payload.get("alert_id", ""))
        self.host = str(payload.get("hostname", ""))
        self.fixed = False
        self.executions = 0

    # Monitoring API.
    def status(self, monitor_id: str, host: str) -> MonitorStatus:
        if monitor_id != self.monitor_id or host != self.host:
            return "Alert"
        w = self.case.world
        if self.fixed:
            return "OK"
        if (
            w.ok_from_s is not None
            and self.clock.now_ms() >= self.case.epoch_ms + w.ok_from_s * 1000
        ):
            return "OK"
        return "Alert"

    # Hosts, as the executor reaches them.
    def diagnostic(self, host: str, name: str) -> Any:
        out = self.case.world.diagnostics.get(Diagnostic(name))
        return out if out is not None else {"error": "not available on this host"}

    def execute(self, action: str, params: dict[str, Any]) -> tuple[Outcome, str]:
        w, i = self.case.world, self.executions
        self.executions += 1
        if w.approval_policy == "approve_all":
            result: Outcome = "ok"
            works = True
        else:
            result = w.executor[i] if i < len(w.executor) else "ok"
            works = w.fix_effect[i] if i < len(w.fix_effect) else False
        if result == "ok" and works and params.get("host") == self.host:
            self.fixed = True
        return result, "simulated"
