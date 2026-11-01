"""The MCP server: read-only tools, plus a way to *ask* a human for an action.

    python -m creo.mcp_server keygen                    # demo signing keys
    python -m creo.mcp_server token --scopes creo:read  # mint a token
    python -m creo.mcp_server serve                     # Streamable HTTP on :8003/mcp

Tools and the scope each needs:
- get_alert, get_run, get_runbook, get_ticket   creo:read
- run_diagnostic (read-only, via the executor)   creo:diagnose
- request_approval, get_request                  creo:request

No tool changes anything. request_approval only creates a pending request on
the dashboard; a named human must approve it, the agent signs a grant bound
to that exact action, and the executor re-checks everything before acting.

What comes back is typed data. Raw alert text, ticket bodies and diagnostic
free text are left out: an MCP client is usually another model, and that text
is exactly where injected instructions hide.

Tokens are RS256 JWTs. The server holds only the public key; whoever holds the
private key mints tokens (here a CLI, standing in for an identity provider).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import require_scopes
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.dependencies import get_access_token
from pydantic import SecretStr

from creo.agent.screen import screen_text
from creo.config import load_policy, load_runbooks
from creo.settings import Settings

ISSUER = "https://creo.kestrel-freight.example/mcp-tokens"
AUDIENCE = "creo-mcp"
READ, DIAGNOSE, REQUEST = "creo:read", "creo:diagnose", "creo:request"
FREE_TEXT_KEYS = {"message", "error", "title", "body", "detail"}


def key_paths(s: Settings) -> tuple[Path, Path]:
    return s.data_dir / "mcp_private.pem", s.data_dir / "mcp_public.pem"


def keygen(s: Settings) -> None:
    priv, pub = key_paths(s)
    if priv.exists():
        return
    s.data_dir.mkdir(parents=True, exist_ok=True)
    pair = RSAKeyPair.generate()
    priv.write_text(pair.private_key.get_secret_value())
    priv.chmod(0o600)
    pub.write_text(pair.public_key)


def mint(s: Settings, subject: str, scopes: list[str], ttl_s: int = 3600) -> str:
    priv, pub = key_paths(s)
    pair = RSAKeyPair(private_key=SecretStr(priv.read_text()), public_key=pub.read_text())
    return pair.create_token(
        subject=subject, issuer=ISSUER, audience=AUDIENCE, scopes=scopes, expires_in_seconds=ttl_s
    )


def _strip_free_text(obj: Any) -> Any:
    """Typed data only: drop free-text fields, recursively."""
    if isinstance(obj, dict):
        return {
            k: ("<free text removed>" if k in FREE_TEXT_KEYS else _strip_free_text(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_free_text(v) for v in obj]
    return obj


def create_server(
    s: Settings | None = None,
    *,
    public_key: str | None = None,
    agent: httpx.Client | None = None,
    tickets: httpx.Client | None = None,
) -> FastMCP:
    s = s or Settings()
    if public_key is None:
        public_key = key_paths(s)[1].read_text()
    auth = JWTVerifier(public_key=public_key, issuer=ISSUER, audience=AUDIENCE, algorithm="RS256")
    api = agent or httpx.Client(
        base_url=s.agent_url, timeout=30.0, headers={"X-Creo-Mcp-Token": s.mcp_token}
    )
    tickets = tickets or httpx.Client(base_url=s.tickets_url, timeout=10.0)
    policy = load_policy()
    mcp = FastMCP(
        "creo",
        instructions=(
            "Read-only access to CREO, an alert-remediation agent on synthetic data. "
            "Nothing here changes a host. request_approval only asks a named human."
        ),
        auth=auth,
    )

    def _get(client: httpx.Client, path: str) -> Any:
        r = client.get(path)
        if r.status_code == 404:
            raise ToolError("not found")
        r.raise_for_status()
        return r.json()

    @mcp.tool(auth=require_scopes(READ), annotations={"readOnlyHint": True})
    def get_run(run_id: str) -> dict[str, Any]:
        """A run's state: host, runbook, category, path, why it stopped, actions and results."""
        run = _get(api, f"/api/runs/{run_id}")
        keep = (
            "run_id",
            "host",
            "final_state",
            "reason",
            "runbook",
            "category",
            "path",
            "waiting_for",
            "flags",
            "actions",
            "ticket_id",
            "escalation_group",
        )
        return {k: run.get(k) for k in keep}

    @mcp.tool(auth=require_scopes(READ), annotations={"readOnlyHint": True})
    def get_alert(run_id: str) -> dict[str, Any]:
        """The alert behind a run, as typed facts: host, monitor, screening result. No raw text."""
        run = _get(api, f"/api/runs/{run_id}")
        title = next(
            (r["title"] for r in _get(api, "/api/runs?limit=200") if r["run_id"] == run_id), ""
        )
        flags = [f.code for f in screen_text(title, "title", policy)]
        return {
            "run_id": run_id,
            "host": run.get("host"),
            "runbook": run.get("runbook"),
            "screen_flags": sorted(set(run.get("flags", [])) | set(flags)),
            "text": "withheld: alert text is untrusted input",
        }

    @mcp.tool(auth=require_scopes(READ), annotations={"readOnlyHint": True})
    def get_runbook(key: str) -> dict[str, Any]:
        """A runbook's config: what it applies to, diagnostics, waits, categories, fixes."""
        rb = load_runbooks().get(key)
        if rb is None:
            raise ToolError(f"no runbook {key!r}")
        return rb.model_dump(mode="json")

    @mcp.tool(auth=require_scopes(READ), annotations={"readOnlyHint": True})
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        """A ticket's status, assigned group and history (operation types only, no text)."""
        t = _get(tickets, f"/tickets/{ticket_id}")
        return {
            "id": t["id"],
            "status": t["status"],
            "host": t["host"],
            "priority": t["priority"],
            "assigned_group": t["assigned_group"],
            "history": [{"op": o["op"], "at_ms": o["at_ms"]} for o in t.get("history", [])],
        }

    @mcp.tool(auth=require_scopes(DIAGNOSE), annotations={"readOnlyHint": True})
    def run_diagnostic(host: str, name: str) -> dict[str, Any]:
        """Run one allowlisted read-only diagnostic via the executor. Free text is removed."""
        r = api.post("/api/diagnostics", json={"host": host, "name": name})
        r.raise_for_status()
        result = r.json()
        if result.get("status") != "ok":
            raise ToolError(f"refused: {result.get('code')}")
        return {"status": "ok", "output": _strip_free_text(result.get("output"))}

    @mcp.tool(
        auth=require_scopes(REQUEST), annotations={"readOnlyHint": False, "destructiveHint": False}
    )
    def request_approval(action: str, params: dict[str, Any], why: str) -> dict[str, Any]:
        """Ask a named human to approve one allowlisted action. This never executes anything itself.

        The request appears on the CREO dashboard. If a human approves it, the
        executor runs exactly that action with exactly these parameters, once.
        """
        token = get_access_token()
        client = token.client_id if token else "unknown"
        r = api.post(
            "/api/requests", json={"action": action, "params": params, "client": client, "why": why}
        )
        if r.status_code == 422:
            raise ToolError(
                f"refused: {r.json()['detail']['code']}: {r.json()['detail']['detail']}"
            )
        r.raise_for_status()
        return dict(r.json())

    @mcp.tool(auth=require_scopes(REQUEST), annotations={"readOnlyHint": True})
    def get_request(approval_id: str) -> dict[str, Any]:
        """An approval request's status (pending, approve, deny, timeout) and any result."""
        return dict(_get(api, f"/api/requests/{approval_id}"))

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen")
    t = sub.add_parser("token")
    t.add_argument("--subject", default="demo-client")
    t.add_argument("--scopes", nargs="+", default=[READ])
    t.add_argument("--ttl", type=int, default=3600)
    sub.add_parser("serve")
    args = ap.parse_args()
    s = Settings()
    if args.cmd == "keygen":
        keygen(s)
        print(f"keys in {s.data_dir}")
    elif args.cmd == "token":
        print(mint(s, args.subject, args.scopes, args.ttl))
    else:
        keygen(s)  # demo convenience: a real deployment gets its public key from an IdP
        create_server(s).run(
            transport="streamable-http",
            host="0.0.0.0",
            port=int(os.environ.get("CREO_MCP_PORT", "8003")),
        )


if __name__ == "__main__":
    main()
