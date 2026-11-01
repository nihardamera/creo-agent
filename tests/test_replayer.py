"""Replays must not collide with earlier replays on the same agent."""

from __future__ import annotations

from creo.dataset.io import load
from creo.replayer import localise
from creo.webhook import AlertWebhook


def test_two_replay_sessions_send_distinct_valid_event_ids() -> None:
    case = next(c for c in load("dev", purpose="tune") if c.alert.payload)
    a, b = localise(case, 7, 0, session=1), localise(case, 7, 0, session=2)
    pa, pb = a.alert.payload or {}, b.alert.payload or {}
    assert pa["id"] != pb["id"] and pa["alert_id"] != pb["alert_id"]
    for p in (pa, pb):
        AlertWebhook.model_validate(p)  # still within the webhook's id pattern
