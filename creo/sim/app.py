"""The `sim` service: the monitoring API and the simulated hosts, for live replays.

The replayer registers each case before posting its alerts. The agent asks
this service for monitor status; the executor runs diagnostics and actions
against it. Each host answers from the most recently registered case on it.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from creo.dataset.schema import Case
from creo.settings import Settings, clock
from creo.sim.world import CaseWorld


class Register(BaseModel):
    case: Case
    monitor_id: str  # the id the replayer actually posts (namespaced per replay)


class Execute(BaseModel):
    action: str
    params: dict[str, Any]


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    clk = clock(s)
    app = FastAPI(title="creo sim (monitoring + simulated hosts)")
    lock = threading.Lock()
    by_monitor: dict[tuple[str, str], CaseWorld] = {}
    by_host: dict[str, CaseWorld] = {}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/cases")
    def register(body: Register) -> dict[str, str]:
        world = CaseWorld(body.case, clk)
        world.monitor_id = body.monitor_id
        with lock:
            by_monitor[(body.monitor_id, world.host)] = world
            by_host[world.host] = world
        return {"status": "registered", "host": world.host}

    @app.get("/monitors/{monitor_id}/status")
    def status(monitor_id: str, host: str) -> dict[str, str]:
        world = by_monitor.get((monitor_id, host))
        return {"status": world.status(monitor_id, host) if world else "Alert"}

    @app.post("/hosts/{host}/diagnostics/{name}")
    def diagnostic(host: str, name: str) -> dict[str, Any]:
        world = by_host.get(host)
        if world is None:
            raise HTTPException(404, "no simulated host by that name")
        return {"output": world.diagnostic(host, name)}

    @app.post("/hosts/{host}/execute")
    def execute(host: str, body: Execute) -> dict[str, str]:
        world = by_host.get(host)
        if world is None:
            raise HTTPException(404, "no simulated host by that name")
        result, detail = world.execute(body.action, body.params)
        return {"result": result, "detail": detail}

    return app
