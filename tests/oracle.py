"""Test doubles for the model. Never used outside tests.

LabelOracleLLM answers `classify` with the case's labelled category and
cites the evidence a correct answer would cite. It exists to test the
workflow given a correct model, so a failure here is a workflow bug, never
a model error. It says nothing about how well any real model classifies.
"""

from __future__ import annotations

import json
from typing import Any

from creo.agent.llm import LLMRequest, LLMResponse
from creo.dataset.schema import Case


def _growth(e: dict[str, Any]) -> float:
    mem = e["mem_mb_last_30m"]
    return float(mem[-1] - mem[0])


class LabelOracleLLM:
    name = "label-oracle"

    def __init__(self, case: Case) -> None:
        self.case = case

    def _cite(self, facts: dict[str, Any], category: str) -> list[str]:
        ev = facts["evidence"]
        calls = self.case.labels.expected_proposals
        params = calls[0].params if calls else {}
        procs = [e for e in ev if e["kind"] == "process"]
        # Without a proposal in the labels (e.g. the monitor recovers first), cite
        # what the typed evidence shows, as a correct model would.
        if category == "runaway_process" and procs:
            if "pid" in params:
                return [e["id"] for e in procs if e["pid"] == params["pid"]][:1]
            return [max(procs, key=lambda e: min(e["cpu_pct_last_5m"]))["id"]]
        if category in ("service_leak", "app_pool_leak") and procs:
            owner = params.get("service_name") or params.get("app_pool")
            mine = [e for e in procs if e["owner"] == owner] or procs
            return [max(mine, key=_growth)["id"]]
        if category == "service_crash":
            svcs = [e for e in ev if e["kind"] == "service"]
            want = params.get("service_name")
            hit = [e for e in svcs if e["service"] == want] if want else []
            hit = hit or [e for e in svcs if (e["restarts_24h"] or 0) > 0]
            return [e["id"] for e in hit][:1]
        return [ev[0]["id"]] if ev else []

    def complete(self, req: LLMRequest) -> LLMResponse:
        f = req.facts
        if req.task == "classify":
            cat = self.case.labels.expected_classification
            if cat not in f["categories"]:
                cat = f["categories"][0]
            body: dict[str, Any] = {
                "category": cat,
                "evidence": self._cite(f, cat),
                "confidence": 0.9,
            }
        else:
            body = {
                "summary": f"Automation stopped on {f['host']} with reason {f['reason']}. "
                "A human should review the ticket.",
                "next_step": "Check the host and the ticket history.",
            }
        text = json.dumps(body)
        return LLMResponse(text, self.name, len(json.dumps(f)) // 4, len(text) // 4, 0.0)


class ScriptedLLM:
    """Returns fixed texts in order (for testing validation and fallbacks)."""

    name = "scripted"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return LLMResponse(self.texts.pop(0) if self.texts else "{}", self.name, 10, 10, 0.0)
