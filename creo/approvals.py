"""Approval binding: one named human, one action, its exact parameters, an expiry.

A request's digest is a SHA-256 over the canonical JSON of the run, the
action, its parameters and the expiry. The approver's decision is signed with
HMAC-SHA256 over that digest, so a grant can't be reused for a different
action, different parameters, a different run, or after it expires. The
executor recomputes the digest from the job it was actually sent and checks
the signature before doing anything.

Limitation: HMAC means the agent and the executor share the key. Anyone who
holds it (for example, someone who has taken over the agent process) can mint
approvals. An approval service holding an asymmetric signing key that the
agent never sees would close that; see SECURITY.md (M5).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Decision = Literal["approve", "deny"]


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    run_id: str
    action: str
    params: dict[str, str | int]
    requested_ms: int
    expires_ms: int

    @property
    def digest(self) -> str:
        return request_digest(self.run_id, self.action, self.params, self.expires_ms)


def request_digest(run_id: str, action: str, params: dict[str, Any], expires_ms: int) -> str:
    body = {"run_id": run_id, "action": action, "params": params, "expires_ms": expires_ms}
    return hashlib.sha256(canonical(body)).hexdigest()


class Grant(BaseModel):
    """A signed decision on one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    digest: str
    decision: Decision
    approver: str
    decided_ms: int
    expires_ms: int
    signature: str


def _mac(
    key: bytes, approval_id: str, digest: str, decision: str, approver: str, expires_ms: int
) -> str:
    msg = canonical([approval_id, digest, decision, approver, expires_ms])
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def sign(key: bytes, req: ApprovalRequest, decision: Decision, approver: str, now_ms: int) -> Grant:
    return Grant(
        approval_id=req.approval_id,
        digest=req.digest,
        decision=decision,
        approver=approver,
        decided_ms=now_ms,
        expires_ms=req.expires_ms,
        signature=_mac(key, req.approval_id, req.digest, decision, approver, req.expires_ms),
    )


def check(
    key: bytes,
    grant: Grant,
    *,
    run_id: str,
    action: str,
    params: dict[str, Any],
    now_ms: int,
    approvers: tuple[str, ...],
) -> str | None:
    """None if the grant authorises exactly this action now, else the reason it doesn't."""
    expected = _mac(
        key, grant.approval_id, grant.digest, grant.decision, grant.approver, grant.expires_ms
    )
    if not hmac.compare_digest(expected, grant.signature):
        return "bad_signature"
    if grant.digest != request_digest(run_id, action, params, grant.expires_ms):
        return "digest_mismatch"  # approved something else: other action, params or run
    if grant.decision != "approve":
        return "not_approved"
    if grant.approver not in approvers:
        return "unknown_approver"
    if now_ms >= grant.expires_ms:
        return "expired"
    return None
