"""HTTP clients for the services, implementing the same interfaces the harness fakes."""

from __future__ import annotations

from typing import Any, cast

import httpx

from creo.agent.env import MonitorStatus
from creo.executor.core import Outcome


class HttpTickets:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def create(self, *, title: str, body: str, host: str, priority: str, source_run: str) -> str:
        r = self.http.post(
            "/tickets",
            json={
                "title": title,
                "body": body,
                "host": host,
                "priority": priority,
                "source_run": source_run,
            },
        )
        r.raise_for_status()
        return str(r.json()["id"])

    def comment(self, ticket_id: str, text: str) -> None:
        self.http.post(f"/tickets/{ticket_id}/comments", json={"text": text}).raise_for_status()

    def transition(self, ticket_id: str, status: str, *, group: str | None = None) -> None:
        r = self.http.post(
            f"/tickets/{ticket_id}/transitions", json={"status": status, "group": group}
        )
        r.raise_for_status()


class HttpMonitoring:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def status(self, monitor_id: str, host: str) -> MonitorStatus:
        r = self.http.get(f"/monitors/{monitor_id}/status", params={"host": host})
        r.raise_for_status()
        return cast(MonitorStatus, r.json()["status"])


class HttpHosts:
    """The executor's way onto the (simulated) hosts."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def diagnostic(self, host: str, name: str) -> Any:
        r = self.http.post(f"/hosts/{host}/diagnostics/{name}")
        if r.status_code == 404:
            return {"error": "host unreachable"}
        r.raise_for_status()
        return r.json()["output"]

    def execute(self, action: str, params: dict[str, Any]) -> tuple[Outcome, str]:
        r = self.http.post(
            f"/hosts/{params['host']}/execute", json={"action": action, "params": params}
        )
        if r.status_code == 404:
            return "error", "host unreachable"
        r.raise_for_status()
        data = r.json()
        return cast(Outcome, data["result"]), str(data["detail"])
