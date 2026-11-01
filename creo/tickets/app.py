"""The `mock-tickets` service: a Jira Service Management stand-in over HTTP."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from creo.settings import Settings, clock
from creo.tickets.store import SqliteTickets, TicketError


class NewTicket(BaseModel):
    title: str
    body: str
    host: str
    priority: str
    source_run: str


class Comment(BaseModel):
    text: str


class Move(BaseModel):
    status: str
    group: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteTickets(s.data_dir / "tickets.db", clock(s))
    app = FastAPI(title="creo mock tickets")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tickets")
    def create(t: NewTicket) -> dict[str, str]:
        return {"id": store.create(**t.model_dump())}

    @app.post("/tickets/{tid}/comments")
    def comment(tid: str, c: Comment) -> dict[str, str]:
        try:
            store.comment(tid, c.text)
        except TicketError as e:
            raise HTTPException(404, str(e)) from None
        return {"status": "ok"}

    @app.post("/tickets/{tid}/transitions")
    def transition(tid: str, m: Move) -> dict[str, str]:
        try:
            store.transition(tid, m.status, group=m.group)
        except TicketError as e:
            raise HTTPException(400, str(e)) from None
        return {"status": "ok"}

    @app.get("/tickets")
    def all_tickets() -> list[dict[str, Any]]:
        return store.all()

    @app.get("/tickets/{tid}")
    def one(tid: str) -> dict[str, Any]:
        t = store.get(tid)
        if t is None:
            raise HTTPException(404, "no such ticket")
        return {**t, "history": store.ops(tid)}

    return app
