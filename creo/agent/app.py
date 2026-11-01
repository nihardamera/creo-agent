"""The `agent` service: webhook ingress, the wake scheduler, the APIs and a tiny dashboard.

- POST /webhook needs the shared secret in X-Creo-Secret; the body must pass
  the schema and size limit or it is refused with a reason.
- /api/approvals needs the approver token; the decision must name an approver
  from policy. The agent signs a grant bound to that exact request.
- /api/jobs is how the executor polls for work (it has no inbound port) and
  returns results; it needs the executor token.
- / is a dashboard: recent runs, what each waits for, approve/deny buttons.
  All alert text is HTML-escaped: titles and messages are attacker-controlled.
"""

from __future__ import annotations

import hmac
import html
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from creo import tracing
from creo.agent.env import Env
from creo.agent.llm import CassetteLLM, LLMClient, OllamaLLM, OpenAICompatLLM
from creo.agent.runtime import Agent, ApprovalError
from creo.agent.store import Store
from creo.clients import HttpMonitoring, HttpTickets
from creo.config import load_inventory, load_policy, load_runbooks
from creo.settings import Settings, clock


def make_llm(s: Settings) -> LLMClient:
    if s.llm_mode == "ollama":
        return OllamaLLM(s.ollama_url, s.ollama_model)
    if s.llm_mode == "openai":
        if not (s.openai_key and s.openai_model):
            raise SystemExit("CREO_LLM=openai needs OPENAI_API_KEY and OPENAI_MODEL")
        return OpenAICompatLLM(s.openai_url, s.openai_model, s.openai_key)
    return CassetteLLM(s.cassette)


def build_agent(
    s: Settings, *, tickets: Any = None, monitoring: Any = None, llm: Any = None
) -> Agent:
    s.data_dir.mkdir(parents=True, exist_ok=True)
    env = Env(
        inventory=load_inventory(),
        runbooks=load_runbooks(),
        policy=load_policy(),
        store=Store(s.data_dir / "agent.db"),
        tickets=tickets or HttpTickets(s.tickets_url),
        monitoring=monitoring or HttpMonitoring(s.sim_url),
        llm=llm or make_llm(s),
        clock=clock(s),
        approval_key=s.approval_key,
    )
    conn = sqlite3.connect(s.data_dir / "checkpoints.db", check_same_thread=False)
    return Agent(env, SqliteSaver(conn))


def _same(a: str | None, b: str) -> bool:
    return a is not None and hmac.compare_digest(a.encode(), b.encode())


class Decision(BaseModel):
    decision: Literal["approve", "deny"]
    approver: str


def create_app(
    settings: Settings | None = None, agent: Agent | None = None, *, scheduler: bool = True
) -> FastAPI:
    s = settings or Settings()
    tracing.setup("creo-agent")
    ag = agent or build_agent(s)
    stop = threading.Event()

    def wake_loop() -> None:
        while not stop.is_set():
            try:
                ag.tick()
            except Exception as e:  # keep the scheduler alive; the run's state is checkpointed
                print(f"tick failed: {e!r}", flush=True)
            stop.wait(0.5)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        t = threading.Thread(target=wake_loop, daemon=True) if scheduler else None
        if t:
            t.start()
        yield
        stop.set()

    app = FastAPI(title="creo agent", lifespan=lifespan)
    app.state.agent = ag

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "llm": ag.env.llm.name, "killswitch": ag.env.store.killswitch()}

    @app.post("/webhook")
    async def webhook(
        request: Request, x_creo_secret: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        if not _same(x_creo_secret, s.webhook_secret):
            raise HTTPException(401, "bad or missing X-Creo-Secret")
        body = await request.body()
        r = ag.receive(body)
        if r.status == "rejected":
            code = 413 if r.reason == "payload_too_large" else 400
            return JSONResponse({"status": "rejected", "reason": r.reason}, status_code=code)
        return JSONResponse({"status": r.status, "run_id": r.run_id}, status_code=202)

    # ---- runs and approvals ------------------------------------------------------

    def run_summary(run_id: str) -> dict[str, Any]:
        st = ag.state(run_id)
        pending = ag.env.store.pending(run_id)
        return {
            "run_id": run_id,
            "host": st.get("host"),
            "final_state": st.get("final_state"),
            "reason": st.get("reason"),
            "runbook": st.get("runbook"),
            "category": st.get("category"),
            "path": st.get("path", []),
            "waiting_for": pending[0]["kind"] if pending else None,
            "due_ms": pending[0]["due_ms"] if pending else None,
            "flags": st.get("flags", []),
            "actions": st.get("actions", []),
            "ticket_id": st.get("ticket_id"),
            "escalation_group": st.get("escalation_group"),
            "summary": st.get("summary"),
        }

    @app.get("/api/runs")
    def runs(limit: int = 50) -> list[dict[str, Any]]:
        return [{**r, **run_summary(r["run_id"])} for r in ag.env.store.recent_runs(limit)]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return {**run_summary(run_id), "llm_calls": ag.env.store.llm_calls(run_id)}

    @app.get("/api/approvals")
    def approvals_list() -> list[dict[str, Any]]:
        return ag.pending_approvals()

    def _decide(approval_id: str, d: Decision, token: str | None) -> None:
        if not _same(token, s.approver_token):
            raise HTTPException(401, "bad or missing approver token")
        try:
            ag.decide(approval_id, d.decision, d.approver)
        except ApprovalError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/approvals/{approval_id}")
    def decide(
        approval_id: str, d: Decision, x_creo_approver_token: Annotated[str | None, Header()] = None
    ) -> dict[str, str]:
        _decide(approval_id, d, x_creo_approver_token)
        return {"status": "decided"}

    @app.post("/ui/approvals/{approval_id}")
    def decide_form(
        approval_id: str,
        decision: Annotated[str, Form()],
        approver: Annotated[str, Form()],
        token: Annotated[str, Form()],
    ) -> RedirectResponse:
        if decision not in ("approve", "deny"):
            raise HTTPException(400, "decision must be approve or deny")
        _decide(approval_id, Decision(decision=decision, approver=approver), token)  # type: ignore[arg-type]
        return RedirectResponse("/", status_code=303)

    # ---- executor jobs -------------------------------------------------------------

    def _executor(token: str | None) -> None:
        if not _same(token, s.executor_token):
            raise HTTPException(401, "bad or missing executor token")

    @app.post("/api/jobs/claim")
    def claim(x_creo_executor_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        _executor(x_creo_executor_token)
        return {"jobs": ag.claim_jobs(), "killswitch": ag.env.store.killswitch()}

    @app.post("/api/jobs/{job_id}/result")
    def result(
        job_id: str,
        body: dict[str, Any],
        x_creo_executor_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, str]:
        _executor(x_creo_executor_token)
        ag.complete_job(job_id, body)
        return {"status": "recorded"}

    @app.post("/api/killswitch")
    def killswitch(
        on: bool, x_creo_approver_token: Annotated[str | None, Header()] = None
    ) -> dict[str, bool]:
        if not _same(x_creo_approver_token, s.approver_token):
            raise HTTPException(401, "bad or missing approver token")
        ag.set_killswitch(on)
        return {"killswitch": on}

    # ---- dashboard -----------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        e = html.escape
        now = ag.env.clock.now_ms()
        rows = []
        for r in runs(40):
            due = f" (in {max(0, (r['due_ms'] - now) // 1000)} s)" if r.get("due_ms") else ""
            state = r["final_state"] or f"waiting: {r['waiting_for'] or 'running'}{due}"
            path = " → ".join(r["path"][-6:])
            cells = [r["run_id"], r["host"] or "", r["title"], state, r["reason"] or "", path]
            rows.append("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in cells) + "</tr>")
        options = "".join(f"<option>{e(a)}</option>" for a in ag.env.policy.approvers)
        cards = []
        for a in ag.pending_approvals():
            mins = max(0, (a["expires_ms"] - now) // 60_000)
            cards.append(
                f"<div class=card><b>{e(a['action'])}</b> <code>{e(str(a['params']))}</code>"
                f"<br>run {e(a['run_id'])}, expires in {mins} min"
                f"<form method=post action='/ui/approvals/{e(a['approval_id'])}'>"
                f"<select name=approver>{options}</select> "
                "<input name=token type=password placeholder='approver token' required> "
                "<button name=decision value=approve>Approve</button> "
                "<button name=decision value=deny>Deny</button></form></div>"
            )
        ks = "ON: no changes will run" if ag.env.store.killswitch() else "off"
        return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=3>
<meta name=viewport content="width=device-width,initial-scale=1"><title>CREO</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:4px 6px;
text-align:left;vertical-align:top}}.card{{background:#fff;border:1px solid #ccc;padding:8px;
margin:6px 0}}code{{font-size:12px}}</style></head><body>
<h1>CREO</h1><p>Synthetic data only. Model: {e(ag.env.llm.name)}. Kill switch: {e(ks)}.</p>
<h2>Pending approvals</h2>{"".join(cards) or "<p>None.</p>"}
<h2>Recent runs</h2><table><tr><th>Run</th><th>Host</th><th>Alert</th><th>State</th>
<th>Reason</th><th>Last steps</th></tr>{"".join(rows)}</table></body></html>"""

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
