import pytest

from creo.actions import ActionRefused, check_inventory, parse
from creo.approvals import ApprovalRequest, check, sign
from creo.catalog import Action
from creo.config import load_inventory, load_policy

KEY = b"test-key"
APPROVERS = load_policy().approvers


def _req(**over: object) -> ApprovalRequest:
    base: dict[str, object] = {
        "approval_id": "a1",
        "run_id": "r1",
        "action": "restart_windows_service",
        "params": {"host": "ash-app-01", "service_name": "KfRateCalc"},
        "requested_ms": 1_000,
        "expires_ms": 1_000 + 30 * 60_000,
    }
    return ApprovalRequest.model_validate({**base, **over})


def _check(grant_req: ApprovalRequest, **job: object) -> str | None:
    grant = sign(KEY, grant_req, "approve", "a.mehta", now_ms=2_000)
    target = {
        "run_id": "r1",
        "action": grant_req.action,
        "params": grant_req.params,
        "now_ms": 3_000,
    }
    target.update(job)
    return check(KEY, grant, approvers=APPROVERS, **target)  # type: ignore[arg-type]


def test_a_grant_authorises_exactly_its_action() -> None:
    assert _check(_req()) is None


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        ({"action": "restart_iis", "params": {"host": "ash-app-01"}}, "digest_mismatch"),
        ({"params": {"host": "ash-app-01", "service_name": "KfDockSched"}}, "digest_mismatch"),
        ({"params": {"host": "ash-app-02", "service_name": "KfRateCalc"}}, "digest_mismatch"),
        ({"run_id": "r2"}, "digest_mismatch"),
        ({"now_ms": 1_000 + 30 * 60_000}, "expired"),
    ],
)
def test_a_grant_does_not_authorise_anything_else(job: dict[str, object], reason: str) -> None:
    assert _check(_req(), **job) == reason


def test_forged_or_tampered_grants_fail() -> None:
    req = _req()
    grant = sign(KEY, req, "approve", "a.mehta", now_ms=2_000)
    kwargs = {"run_id": "r1", "action": req.action, "params": req.params, "now_ms": 3_000}
    forged = sign(b"wrong-key", req, "approve", "a.mehta", now_ms=2_000)
    assert check(KEY, forged, approvers=APPROVERS, **kwargs) == "bad_signature"  # type: ignore[arg-type]
    later = grant.model_copy(update={"expires_ms": grant.expires_ms + 3_600_000})
    assert check(KEY, later, approvers=APPROVERS, **kwargs) == "bad_signature"  # type: ignore[arg-type]
    other = grant.model_copy(update={"approver": "j.okafor"})
    assert check(KEY, other, approvers=APPROVERS, **kwargs) == "bad_signature"  # type: ignore[arg-type]


def test_denials_and_unknown_approvers_do_not_authorise() -> None:
    req = _req()
    kwargs = {"run_id": "r1", "action": req.action, "params": req.params, "now_ms": 3_000}
    denied = sign(KEY, req, "deny", "a.mehta", now_ms=2_000)
    assert check(KEY, denied, approvers=APPROVERS, **kwargs) == "not_approved"  # type: ignore[arg-type]
    stranger = sign(KEY, req, "approve", "ops-lead", now_ms=2_000)
    assert check(KEY, stranger, approvers=APPROVERS, **kwargs) == "unknown_approver"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action", "params", "code"),
    [
        ("run_shell", {"host": "ash-app-01", "cmd": "shutdown /r"}, "not_allowlisted"),
        ("reboot_host", {"host": "ash-app-01"}, "not_allowlisted"),
        ("restart_windows_service", {"host": "ash-app-01", "service_name": "*"}, "bad_params"),
        ("restart_windows_service", {"host": "*", "service_name": "KfRateCalc"}, "bad_params"),
        ("start_app_pool", {"host": "ash-web-01", "app_pool": "../../Windows"}, "bad_params"),
        (
            "clear_temp_files",
            {"host": "ash-app-01", "drive": "C:\\*", "older_than_days": 7},
            "bad_params",
        ),
        (
            "clear_temp_files",
            {"host": "ash-app-01", "drive": "C:", "older_than_days": 0},
            "bad_params",
        ),
        ("restart_iis", {"host": "ash-web-01", "extra": "x"}, "bad_params"),
        ("kill_process", {"host": "ash-app-01", "process_name": "lsass", "pid": 1}, "bad_params"),
    ],
)
def test_parse_refuses_unlisted_actions_and_bad_params(
    action: str, params: dict[str, object], code: str
) -> None:
    with pytest.raises(ActionRefused) as e:
        parse(action, params)
    assert e.value.code == code


def test_inventory_check_refuses_resources_the_host_does_not_have() -> None:
    inv = load_inventory()
    app = next(h for h in inv.hosts if h.role == "app")
    missing = next(s for s in ("NTDS", "W3SVC") if s not in app.services)
    act, params = parse("restart_windows_service", {"host": app.name, "service_name": missing})
    with pytest.raises(ActionRefused) as e:
        check_inventory(act, params, inv)
    assert e.value.code == "unknown_resource"
    act, params = parse(
        "restart_windows_service", {"host": "nope-01", "service_name": "KfRateCalc"}
    )
    with pytest.raises(ActionRefused) as e:
        check_inventory(act, params, inv)
    assert e.value.code == "unknown_host"
    linux = next(h for h in inv.hosts if h.os == "linux")
    act, params = parse("restart_iis", {"host": linux.name})
    with pytest.raises(ActionRefused) as e:
        check_inventory(act, params, inv)
    assert e.value.code == "wrong_os"


def test_inventory_check_accepts_a_real_resource() -> None:
    inv = load_inventory()
    app = next(h for h in inv.hosts if h.role == "app")
    act, params = parse(
        "restart_windows_service", {"host": app.name, "service_name": app.services[0]}
    )
    assert check_inventory(act, params, inv).name == app.name
    assert act is Action.RESTART_WINDOWS_SERVICE
