"""The agent service's HTTP surface: authentication, refusals and an escaped dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from creo.agent.app import build_agent, create_app
from creo.dataset.generate import body_bytes
from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.settings import Settings, clock
from creo.sim.world import CaseWorld
from creo.tickets.store import SqliteTickets
from tests.oracle import LabelOracleLLM

CASE: Case = next(
    c for c in load("dev", purpose="tune") if c.labels.scenario_class == "fix_succeeds"
)
APPROVE = {"decision": "approve", "approver": "a.mehta"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    s = Settings(
        data_dir=tmp_path,
        webhook_secret="w",
        approver_token="a",
        executor_token="x",
        clock_epoch_ms=CASE.epoch_ms,
        time_scale=1.0,
    )
    clk = clock(s)
    agent = build_agent(
        s,
        tickets=SqliteTickets(":memory:", clk),
        monitoring=CaseWorld(CASE, clk),
        llm=LabelOracleLLM(CASE),
    )
    return TestClient(create_app(s, agent, scheduler=False))


def _post(c: TestClient, body: bytes, secret: str | None = "w") -> httpx.Response:
    headers = {"X-Creo-Secret": secret} if secret is not None else {}
    return c.post("/webhook", content=body, headers=headers)


def test_webhook_needs_the_shared_secret(client: TestClient) -> None:
    body = body_bytes(CASE.alert)
    assert _post(client, body, secret=None).status_code == 401
    assert _post(client, body, secret="wrong").status_code == 401
    ok = _post(client, body)
    assert ok.status_code == 202 and ok.json()["run_id"]


@pytest.mark.parametrize(
    ("body", "code", "reason"),
    [
        (b"{not json", 400, "invalid_json"),
        (b"[]", 400, "not_an_object"),
        (json.dumps({"title": "x"}).encode(), 400, "schema"),
        (b"{" + b" " * 300_000 + b"}", 413, "payload_too_large"),
    ],
)
def test_bad_bodies_are_refused_with_a_reason(
    client: TestClient, body: bytes, code: int, reason: str
) -> None:
    r = _post(client, body)
    assert r.status_code == code and r.json()["reason"] == reason


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        ("/api/approvals/x", {}),
        ("/api/approvals/x", {"X-Creo-Approver-Token": "x"}),  # the executor's token
        ("/api/jobs/claim", {}),
        ("/api/jobs/claim", {"X-Creo-Executor-Token": "a"}),  # the approver's token
        ("/api/killswitch?on=true", {}),
    ],
)
def test_each_api_needs_its_own_token(
    client: TestClient, path: str, headers: dict[str, str]
) -> None:
    assert client.post(path, json=APPROVE, headers=headers).status_code == 401


def test_the_executor_can_poll_with_its_token(client: TestClient) -> None:
    r = client.post("/api/jobs/claim", headers={"X-Creo-Executor-Token": "x"})
    assert r.status_code == 200 and r.json() == {"jobs": [], "killswitch": False}


def test_unknown_approval_is_a_conflict_not_a_crash(client: TestClient) -> None:
    r = client.post("/api/approvals/nope", json=APPROVE, headers={"X-Creo-Approver-Token": "a"})
    assert r.status_code == 409


def test_dashboard_escapes_alert_text(client: TestClient) -> None:
    payload = dict(CASE.alert.payload or {})
    payload["title"] = "<script>alert('x')</script> disk full"
    payload["id"] = "1234567890"
    assert _post(client, json.dumps(payload).encode()).status_code == 202
    page = client.get("/").text
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert" in page
