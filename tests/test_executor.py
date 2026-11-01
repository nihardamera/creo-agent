"""The executor trusts nothing it is sent: each refusal below happens before any host is touched."""

from __future__ import annotations

from typing import Any

import pytest

from creo.agent.env import SimClock
from creo.approvals import ApprovalRequest, sign
from creo.config import load_inventory, load_policy
from creo.executor.core import Executor, Outcome

KEY = b"executor-test-key"
INV = load_inventory()
APP = next(h for h in INV.hosts if h.role == "app")


class FakeHosts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def diagnostic(self, host: str, name: str) -> Any:
        return {"ok": True}

    def execute(self, action: str, params: dict[str, Any]) -> tuple[Outcome, str]:
        self.calls.append((action, params))
        return "ok", "simulated"


def _setup(killswitch: bool = False) -> tuple[Executor, FakeHosts, SimClock]:
    clock, hosts = SimClock(1_000_000), FakeHosts()
    ex = Executor(inventory=INV, policy=load_policy(), approval_key=KEY, backend=hosts,
                  clock=clock, killswitch=lambda: killswitch)  # fmt: skip
    return ex, hosts, clock


def _job(action: str, params: dict[str, Any], *, key: bytes = KEY, approver: str = "a.mehta",
         sign_params: dict[str, Any] | None = None, run_id: str = "r1", ttl_ms: int = 60_000,
         job_id: str = "j1") -> dict[str, Any]:  # fmt: skip
    req = ApprovalRequest(approval_id=f"{run_id}:fix0", run_id=run_id, action=action,
                          params=sign_params or params, requested_ms=1_000_000,
                          expires_ms=1_000_000 + ttl_ms)  # fmt: skip
    grant = sign(key, req, "approve", approver, 1_000_100)
    payload = {
        "action": action,
        "params": params,
        "run_id": run_id,
        "grant": grant.model_dump_json(),
    }
    return {"job_id": job_id, "run_id": run_id, "kind": "action", "payload": payload}


GOOD = {"host": APP.name, "service_name": APP.services[0]}


def test_a_correctly_approved_action_runs_once() -> None:
    ex, hosts, _ = _setup()
    assert ex.run(_job("restart_windows_service", GOOD))["status"] == "ok"
    assert hosts.calls == [("restart_windows_service", GOOD)]


@pytest.mark.parametrize(
    ("job", "code"),
    [
        (_job("run_shell", {"host": APP.name, "cmd": "shutdown /r"}), "not_allowlisted"),
        (_job("restart_windows_service", {**GOOD, "service_name": "*"}), "bad_params"),
        (_job("restart_windows_service", {**GOOD, "service_name": "NTDS"}), "unknown_resource"),
        (_job("restart_windows_service", GOOD, key=b"forged"), "bad_signature"),
        (_job("restart_windows_service", GOOD, approver="ops-lead"), "unknown_approver"),
        # Approved one service, sent another: the grant doesn't cover it.
        (_job("restart_windows_service", GOOD,
              sign_params={**GOOD, "service_name": APP.services[-1] + "x"}), "digest_mismatch"),
        (_job("restart_windows_service", GOOD, ttl_ms=-1), "expired"),
    ],
)  # fmt: skip
def test_refusals(job: dict[str, Any], code: str) -> None:
    ex, hosts, _ = _setup()
    result = ex.run(job)
    assert (result["status"], result["code"]) == ("refused", code)
    assert hosts.calls == []
    assert ex.attempts()[-1]["code"] == code  # every attempt is logged


def test_an_approval_cannot_be_replayed() -> None:
    ex, hosts, _ = _setup()
    job = _job("restart_windows_service", GOOD)
    assert ex.run(job)["status"] == "ok"
    again = ex.run({**job, "job_id": "j2"})
    assert again == {"status": "refused", "code": "approval_already_used"}
    assert len(hosts.calls) == 1


def test_a_job_without_a_grant_is_refused() -> None:
    ex, hosts, _ = _setup()
    job = _job("restart_windows_service", GOOD)
    del job["payload"]["grant"]
    assert ex.run(job)["code"] == "no_grant"
    assert hosts.calls == []


def test_the_kill_switch_stops_even_approved_actions() -> None:
    ex, hosts, _ = _setup(killswitch=True)
    assert ex.run(_job("restart_windows_service", GOOD))["code"] == "killswitch"
    assert hosts.calls == []


def test_diagnostics_are_read_only_and_allowlisted() -> None:
    ex, _, _ = _setup()
    ok = ex.run({"job_id": "d1", "run_id": "r1", "kind": "diagnostic",
                 "payload": {"host": APP.name, "name": "get_service_status"}})  # fmt: skip
    assert ok["status"] == "ok"
    bad = ex.run({"job_id": "d2", "run_id": "r1", "kind": "diagnostic",
                  "payload": {"host": APP.name, "name": "run_powershell"}})  # fmt: skip
    assert bad["code"] == "not_allowlisted"
