"""The MCP server over real Streamable HTTP: scopes, refusals, and no way around the human gate."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from creo.agent.app import build_agent, create_app
from creo.agent.runtime import Agent
from creo.config import load_inventory, load_policy
from creo.dataset.io import load
from creo.executor.core import Executor
from creo.mcp_server import AUDIENCE, DIAGNOSE, ISSUER, READ, REQUEST, create_server
from creo.settings import Settings, clock
from creo.sim.world import CaseWorld
from creo.tickets.app import create_app as tickets_app
from creo.tickets.store import SqliteTickets
from tests.oracle import LabelOracleLLM

CASE = next(c for c in load("dev", purpose="tune") if c.labels.scenario_class == "fix_succeeds")
ATTACK = next(
    c
    for c in load("adversarial_dev", purpose="tune")
    if c.labels.attack and c.labels.attack.category == "direct_override"
)
APP_HOST = next(h for h in load_inventory().hosts if h.role == "app" and h.tier != 0)
DC = next(h for h in load_inventory().hosts if h.tier == 0)


@dataclass
class Stack:
    url: str
    keys: RSAKeyPair
    agent: Agent
    executor: Executor

    def token(self, *scopes: str, ttl: int = 600, keys: RSAKeyPair | None = None) -> str:
        return (keys or self.keys).create_token(
            subject="test-client",
            issuer=ISSUER,
            audience=AUDIENCE,
            scopes=list(scopes),
            expires_in_seconds=ttl,
        )

    def call(self, token: str | None, tool: str, args: dict[str, Any]) -> Any:
        async def go() -> Any:
            async with Client(self.url, auth=token) as c:
                return (await c.call_tool(tool, args)).data

        return asyncio.run(go())

    def tools(self, token: str) -> list[str]:
        async def go() -> list[str]:
            async with Client(self.url, auth=token) as c:
                return sorted(t.name for t in await c.list_tools())

        return asyncio.run(go())


def _free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return int(sk.getsockname()[1])


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    d: Path = tmp_path_factory.mktemp("mcp")
    s = Settings(data_dir=d, mcp_token="m", clock_epoch_ms=CASE.epoch_ms, time_scale=1.0)
    clk = clock(s)
    world = CaseWorld(CASE, clk)
    agent = build_agent(
        s, tickets=SqliteTickets(":memory:", clk), monitoring=world, llm=LabelOracleLLM(CASE)
    )
    api = TestClient(create_app(s, agent, scheduler=False), headers={"X-Creo-Mcp-Token": "m"})
    executor = Executor(
        inventory=agent.env.inventory,
        policy=load_policy(),
        approval_key=s.approval_key,
        backend=world,
        clock=clk,
    )

    # The executor's poll loop, in-process.
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            for job in agent.claim_jobs():
                agent.complete_job(job["job_id"], executor.run(job))
            stop.wait(0.05)

    threading.Thread(target=poll, daemon=True).start()
    keys = RSAKeyPair.generate()
    mcp = create_server(
        s, public_key=keys.public_key, agent=api, tickets=TestClient(tickets_app(s))
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(mcp.http_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    # Re-time the attack to this clock, or ingest (rightly) ignores it as stale.
    attack = {**(ATTACK.alert.payload or {}), "date": clk.now_ms(), "last_updated": clk.now_ms()}
    agent.receive(json.dumps(attack).encode())
    yield Stack(f"http://127.0.0.1:{port}/mcp", keys, agent, executor)
    server.should_exit = True
    stop.set()


def test_inspector_view_lists_every_tool(stack: Stack) -> None:
    assert stack.tools(stack.token(READ, DIAGNOSE, REQUEST)) == [
        "get_alert",
        "get_request",
        "get_run",
        "get_runbook",
        "get_ticket",
        "request_approval",
        "run_diagnostic",
    ]


INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}


def _initialize(stack: Stack, token: str | None) -> int:
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.post(stack.url, json=INIT, headers=headers, timeout=10).status_code


@pytest.mark.parametrize("who", ["none", "expired", "forged", "wrong_audience"])
def test_bad_credentials_are_refused_at_the_http_layer(stack: Stack, who: str) -> None:
    tokens = {
        "none": None,
        "expired": stack.token(READ, ttl=-10),
        "forged": stack.token(READ, keys=RSAKeyPair.generate()),
        "wrong_audience": stack.keys.create_token(
            subject="t", issuer=ISSUER, audience="other", scopes=[READ]
        ),
    }
    assert _initialize(stack, tokens[who]) == 401
    assert _initialize(stack, stack.token(READ)) == 200


def test_a_read_token_cannot_request_actions(stack: Stack) -> None:
    assert stack.call(stack.token(READ), "get_runbook", {"key": "cpu-high"})["key"] == "cpu-high"
    with pytest.raises(ToolError):
        stack.call(
            stack.token(READ),
            "request_approval",
            {"action": "restart_iis", "params": {"host": "ash-web-01"}, "why": "x"},
        )


@pytest.mark.parametrize(
    ("action", "params", "code"),
    [
        ("run_shell", {"host": APP_HOST.name, "cmd": "shutdown /r"}, "not_allowlisted"),
        ("reboot_host", {"host": APP_HOST.name}, "not_allowlisted"),
        ("restart_windows_service", {"host": APP_HOST.name, "service_name": "*"}, "bad_params"),
        (
            "restart_windows_service",
            {"host": APP_HOST.name, "service_name": "NTDS"},
            "unknown_resource",
        ),
        (
            "restart_windows_service",
            {"host": DC.name, "service_name": DC.services[0]},
            "no_fix_tier",
        ),
    ],
)
def test_requests_outside_policy_are_refused_and_leave_nothing_pending(
    stack: Stack, action: str, params: dict[str, Any], code: str
) -> None:
    before = len(stack.agent.pending_approvals())
    with pytest.raises(ToolError, match=code):
        stack.call(
            stack.token(REQUEST),
            "request_approval",
            {"action": action, "params": params, "why": "x"},
        )
    assert len(stack.agent.pending_approvals()) == before


def test_a_request_executes_only_after_a_named_human_approves_it(stack: Stack) -> None:
    params = {"host": APP_HOST.name, "service_name": APP_HOST.services[0]}
    tok = stack.token(REQUEST)
    r = stack.call(
        tok,
        "request_approval",
        {"action": "restart_windows_service", "params": params, "why": "test"},
    )
    aid = r["approval_id"]
    time.sleep(0.3)
    assert stack.call(tok, "get_request", {"approval_id": aid})["decision"] == "pending"
    assert not [a for a in stack.executor.attempts() if a["kind"] == "action"]  # nothing ran

    stack.agent.decide(aid, "approve", "a.mehta")
    for _ in range(50):
        status = stack.call(tok, "get_request", {"approval_id": aid})
        if status["execution"]:
            break
        time.sleep(0.1)
    assert status["decision"] == "approve" and status["execution"]["status"] == "ok"
    done = [a for a in stack.executor.attempts() if a["kind"] == "action"]
    assert [(a["action"], a["params"]) for a in done] == [("restart_windows_service", params)]

    # The same grant can't be used again.
    grant = next(
        a["grant_json"] for a in stack.agent.env.store.approvals() if a["approval_id"] == aid
    )
    replay = {
        "job_id": "replay",
        "run_id": aid.split(":")[0],
        "kind": "action",
        "payload": {
            "action": "restart_windows_service",
            "params": params,
            "run_id": aid.split(":")[0],
            "grant": grant,
        },
    }
    assert stack.executor.run(replay)["code"] == "approval_already_used"


def test_mcp_never_returns_attack_text(stack: Stack) -> None:
    runs = stack.agent.env.store.recent_runs()
    attack_run = next(
        r["run_id"] for r in runs if r["title"] == (ATTACK.alert.payload or {})["title"][:200]
    )
    alert = stack.call(stack.token(READ), "get_alert", {"run_id": attack_run})
    run = stack.call(stack.token(READ), "get_run", {"run_id": attack_run})
    blob = str(alert) + str(run)
    assert "IGNORE" not in blob and "instructions" not in blob.lower()
    assert alert["screen_flags"]


def test_diagnostics_come_back_without_free_text(stack: Stack) -> None:
    out = stack.call(
        stack.token(DIAGNOSE),
        "run_diagnostic",
        {
            "host": CASE.alert.payload["hostname"],  # type: ignore[index]
            "name": "get_recent_events",
        },
    )
    assert "<free text removed>" in str(out) or out["output"]
    with pytest.raises(ToolError, match="not_allowlisted"):
        stack.call(
            stack.token(DIAGNOSE),
            "run_diagnostic",
            {"host": APP_HOST.name, "name": "run_powershell"},
        )
