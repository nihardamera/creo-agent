"""The model never sees raw alert text.

Every model request made while running every dev case (attacks included) is
checked against that case's raw text: no title, message, raw tag string or
diagnostic free text, nor any 16-character piece of them, may appear.
"""

from __future__ import annotations

import json
from typing import Any

from creo.agent.facts import CPU_TREND_POINTS, _change, _trend
from creo.agent.llm import LLMRequest, LLMResponse
from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.harness import run_case
from tests.oracle import LabelOracleLLM

ALLOWED_FACT_KEYS = {
    "classify": {
        "task",
        "runbook",
        "runbook_description",
        "host_role",
        "categories",
        "category_meanings",
        "evidence",
    },
    "summarize": {
        "task",
        "host",
        "host_role",
        "host_tier",
        "runbook",
        "reason",
        "flags",
        "category",
        "steps_taken",
        "actions",
    },
}


class Spy(LabelOracleLLM):
    def __init__(self, case: Case) -> None:
        super().__init__(case)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return super().complete(req)


def _free_text(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += [v] if k in ("message", "error") and isinstance(v, str) else _free_text(v)
        return out
    if isinstance(obj, list):
        return [s for v in obj for s in _free_text(v)]
    return []


def _raw_texts(c: Case) -> list[str]:
    p = c.alert.payload or {}
    texts = [str(p.get("title", "")), str(p.get("message", "")), str(p.get("tags", ""))]
    texts += _free_text(c.world.diagnostics)
    return [t for t in texts if t]


def _pieces(text: str, n: int = 16) -> set[str]:
    words = text.split()
    out = set()
    for i in range(len(words)):
        piece = ""
        for w in words[i:]:
            piece = f"{piece} {w}".strip()
            if len(piece) >= n:
                out.add(piece)
                break
    return out


def test_model_requests_contain_no_raw_alert_text() -> None:
    checked = 0
    for c in load("dev", purpose="tune") + load("adversarial_dev", purpose="tune"):
        spy = Spy(c)
        run_case(c, spy)
        raw = _raw_texts(c)
        pieces = set().union(*(_pieces(t) for t in raw)) if raw else set()
        for req in spy.requests:
            sent = json.dumps(req.messages())
            assert set(req.facts) <= ALLOWED_FACT_KEYS[req.task], (c.id, set(req.facts))
            for text in raw:
                if len(text) >= 16:
                    assert text not in sent, (c.id, text[:80])
            leaked = [p for p in pieces if p in sent]
            assert not leaked, (c.id, leaked[:3])
            checked += 1
    assert checked > 200  # the check really ran over many requests


def test_attack_text_never_reaches_the_model() -> None:
    for c in load("adversarial_dev", purpose="tune"):
        spy = Spy(c)
        run_case(c, spy)
        sent = " ".join(json.dumps(r.messages()) for r in spy.requests).lower()
        for marker in (
            "ignore all previous",
            "approved by",
            "web.config",
            '"tool"',
            "decode and follow",
        ):
            assert marker not in sent, (c.id, marker)


def test_trends_follow_the_fitted_line() -> None:
    assert _trend(_change([90, 76, 61, 45, 30]), CPU_TREND_POINTS) == "falling"
    assert _trend(_change([97, 99, 96, 98, 97]), CPU_TREND_POINTS) == "steady"
    assert _trend(_change([2000, 2500, 3000, 3500]), 0.25 * 2750) == "rising"
