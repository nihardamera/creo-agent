"""Model clients and the validation every model output passes before it is used.

Two tasks, both fed only by creo.agent.facts:
- classify: pick one of the runbook's categories, citing evidence ids;
- summarize: a short escalation note for the human taking over.

Clients:
- CassetteLLM replays recorded responses (the default for tests, CI and the
  demo; no network, no keys). A request with no recording fails, which the
  agent treats like any invalid output: it falls back to its deterministic path.
- OllamaLLM talks to a local Ollama server; OpenAICompatLLM to any
  OpenAI-compatible API. RecordingLLM wraps either to write a cassette.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from creo.approvals import canonical

PROMPT_VERSION = "v2"
Task = Literal["classify", "summarize"]

SYSTEM = {
    "classify": (
        "You classify the likely cause of a server alert for an operations runbook. "
        "You get JSON facts: the runbook, the host's role, the allowed categories, what "
        "each category's evidence looks like (category_meanings) and typed evidence records "
        "from read-only diagnostics. Compare the evidence with each meaning, including how "
        "numeric series change over time, and choose exactly one category from the allowed "
        "list. Cite the ids of the evidence records that support it, including the process "
        "or service record the category is about. "
        "If the evidence does not clearly support a category that implies a fix, prefer "
        "a category that waits or escalates. Reply with JSON only."
    ),
    "summarize": (
        "You write a short escalation note for the on-call engineer taking over an alert. "
        "You get JSON facts: the host, the runbook, why the automation stopped (reason "
        "code), any screening flags, the steps taken and action results. In two to four "
        "plain sentences say what happened, what was tried and what a human should check "
        "next. Mention the host by name. Do not invent facts, links or commands. Reply "
        "with JSON only."
    ),
}


class ClassifyOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    evidence: list[str] = Field(min_length=1, max_length=6)
    confidence: float = Field(ge=0, le=1)


class SummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=40, max_length=800)
    next_step: str = Field(min_length=5, max_length=240)


SCHEMAS: dict[Task, dict[str, Any]] = {
    "classify": ClassifyOut.model_json_schema(),
    "summarize": SummaryOut.model_json_schema(),
}


@dataclass(frozen=True)
class LLMRequest:
    task: Task
    facts: dict[str, Any]

    @property
    def key(self) -> str:
        """Identifies the request for recording and replay."""
        body = {"v": PROMPT_VERSION, "task": self.task, "facts": self.facts}
        return hashlib.sha256(canonical(body)).hexdigest()

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM[self.task]},
            {"role": "user", "content": json.dumps(self.facts, sort_keys=True)},
        ]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def complete(self, req: LLMRequest) -> LLMResponse: ...


class CassetteLLM:
    """Replays responses recorded from a real model, keyed by request."""

    def __init__(self, path: Path) -> None:
        self.name = f"cassette:{path.name}"
        self.records: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def complete(self, req: LLMRequest) -> LLMResponse:
        rec = self.records.get(req.key)
        if rec is None:
            raise LLMError("no recorded response for this request")
        return LLMResponse(
            text=rec["text"],
            model=rec["model"],
            input_tokens=rec["input_tokens"],
            output_tokens=rec["output_tokens"],
            latency_ms=rec["latency_ms"],
        )


class RecordingLLM:
    """Calls a real model and appends each response to a cassette."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner, self.path, self.name = inner, path, f"recording:{inner.name}"
        self.lock = threading.Lock()

    def complete(self, req: LLMRequest) -> LLMResponse:
        resp = self.inner.complete(req)
        rec = {"key": req.key, "task": req.task, **resp.__dict__}
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return resp


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url, self.model, self.timeout_s = base_url.rstrip("/"), model, timeout_s
        self.name = f"ollama:{model}"

    def complete(self, req: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": req.messages(),
            "format": SCHEMAS[req.task],
            "stream": False,
            # A short structured answer: no reasoning trace (ignored by models that can't think).
            "think": False,
            "options": {"temperature": 0, "seed": 0},
        }
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"ollama call failed: {e}") from e
        return LLMResponse(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float = 60.0) -> None:
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout_s, self.name = timeout_s, f"openai-compat:{model}"

    def complete(self, req: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": req.messages(),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": req.task, "schema": SCHEMAS[req.task]},
            },
        }
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"API call failed: {e}") from e
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def anthropic_schema(task: Task) -> dict[str, Any]:
    """SCHEMAS[task] without the bounds structured outputs doesn't take; validation checks them."""
    by_task: dict[str, dict[str, Any]] = {
        "classify": {
            "category": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "summarize": {"summary": {"type": "string"}, "next_step": {"type": "string"}},
    }
    props = by_task[task]
    return {
        "type": "object",
        "properties": props,
        "required": sorted(props),
        "additionalProperties": False,
    }


class AnthropicLLM:
    """Claude through the Messages API, JSON constrained by structured outputs.

    Claude Haiku 5.5 takes no sampling parameters; thinking is off for these
    short structured answers (allowed at the default effort).
    """

    def __init__(self, model: str, api_key: str | None = None, timeout_s: float = 60.0) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=3)
        self.model, self.name = model, f"anthropic:{model}"

    def params(self, req: LLMRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": 512,
            "system": SYSTEM[req.task],
            "messages": [{"role": "user", "content": json.dumps(req.facts, sort_keys=True)}],
            "thinking": {"type": "disabled"},
            "output_config": {
                "format": {"type": "json_schema", "schema": anthropic_schema(req.task)}
            },
        }

    def complete(self, req: LLMRequest) -> LLMResponse:
        import anthropic

        t0 = time.perf_counter()
        try:
            msg = self.client.messages.create(**self.params(req))
        except anthropic.APIError as e:
            raise LLMError(f"anthropic call failed: {e}") from e
        if msg.stop_reason == "refusal":
            cat = msg.stop_details.category if msg.stop_details else None
            raise LLMError(f"refusal: {cat}")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ---- validation: nothing the model says is used until it passes these ----------

URL = re.compile(r"https?://|www\.", re.IGNORECASE)
HOSTLIKE = re.compile(r"\b[a-z]{2,4}-[a-z]{2,4}-\d{2}\b")


def parse_classify(text: str, facts: dict[str, Any]) -> tuple[ClassifyOut | None, str]:
    try:
        out = ClassifyOut.model_validate_json(text)
    except ValidationError:
        return None, "schema"
    if out.category not in facts["categories"]:
        return None, "category_not_allowed"
    ids = {e["id"] for e in facts["evidence"]}
    if not set(out.evidence) <= ids:
        return None, "unknown_evidence"
    return out, "ok"


def parse_summary(text: str, facts: dict[str, Any]) -> tuple[SummaryOut | None, str]:
    try:
        out = SummaryOut.model_validate_json(text)
    except ValidationError:
        return None, "schema"
    body = f"{out.summary} {out.next_step}"
    if URL.search(body) or "```" in body:
        return None, "link_or_code"
    if facts["host"] not in out.summary:
        return None, "host_not_named"
    others = {h for h in HOSTLIKE.findall(body.lower()) if h != facts["host"]}
    if others:
        return None, "other_host_named"
    return out, "ok"


def template_summary(facts: dict[str, Any]) -> SummaryOut:
    """The deterministic note used when the model is unavailable or its output fails."""
    tried = ", ".join(f"{a['action']} ({a['result']})" for a in facts["actions"]) or "no actions"
    flags = f" Screening flags: {', '.join(facts['flags'])}." if facts["flags"] else ""
    return SummaryOut(
        summary=(
            f"Automation stopped on {facts['host']} (runbook {facts['runbook'] or 'none'}): "
            f"{facts['reason'].replace('_', ' ')}. Tried: {tried}.{flags}"
        ),
        next_step="Review the alert and the ticket history, then decide the next step.",
    )
