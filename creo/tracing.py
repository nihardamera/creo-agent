"""OpenTelemetry tracing with OpenInference attributes (rendered by Arize Phoenix).

Spans:
- "run segment" (AGENT): one graph invocation, from a webhook or a resume
  until the run pauses or ends. All of a run's segments share session.id =
  run_id, so a run that waited, restarted or crossed processes reads as one.
- "node <name>" (CHAIN; GUARDRAIL for the screen): every node visit.
- "llm <task>" (LLM): model, tokens, latency, cost. input.value is the
  redacted facts, never alert text, so traces obey the same boundary.
- "executor <kind>" (TOOL): in the executor process, a child of the node that
  queued the job via a W3C traceparent carried in the job.
- "wait wait" / "wait approval": the pause itself, on the wall clock, with its
  length on the agent's clock in creo.wait.seconds.

With CREO_OTLP_ENDPOINT unset, no exporter is installed and spans are no-ops.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues as Kind
from openinference.semconv.trace import SpanAttributes as A
from opentelemetry import context, propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

tracer = trace.get_tracer("creo")

# $ per million tokens, by model name prefix. Local models cost nothing per call.
PRICES: dict[str, tuple[float, float]] = {}


def setup(service: str, exporter: SpanExporter | None = None, *, batch: bool = True) -> None:
    """Install a tracer provider. Uses `exporter` if given, else OTLP/HTTP if configured."""
    endpoint = os.environ.get("CREO_OTLP_ENDPOINT", "")
    if exporter is None and not endpoint:
        return
    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=endpoint)
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(
        BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    )
    trace.set_tracer_provider(provider)


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    for prefix, (pin, pout) in PRICES.items():
        if model.startswith(prefix):
            return (tokens_in * pin + tokens_out * pout) / 1e6
    return 0.0


@contextmanager
def span(name: str, kind: Kind, run_id: str | None = None, **attrs: Any) -> Iterator[trace.Span]:
    # A pause (LangGraph raises to interrupt) is not an error, so exceptions don't mark spans.
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as s:
        s.set_attribute(A.OPENINFERENCE_SPAN_KIND, kind.value)
        if run_id:
            s.set_attribute(A.SESSION_ID, run_id)
            s.set_attribute("creo.run_id", run_id)
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, str | int | float | bool) else json.dumps(v))
        yield s


def llm_span(
    s: trace.Span,
    *,
    model: str,
    facts: dict[str, Any],
    text: str | None,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    ok: bool,
    reason: str,
) -> None:
    s.set_attribute(A.LLM_MODEL_NAME, model)
    s.set_attribute(A.INPUT_VALUE, json.dumps(facts, sort_keys=True))
    if text is not None:
        s.set_attribute(A.OUTPUT_VALUE, text)
    s.set_attribute(A.LLM_TOKEN_COUNT_PROMPT, tokens_in)
    s.set_attribute(A.LLM_TOKEN_COUNT_COMPLETION, tokens_out)
    s.set_attribute(A.LLM_TOKEN_COUNT_TOTAL, tokens_in + tokens_out)
    s.set_attribute(A.LLM_COST_TOTAL, cost(model, tokens_in, tokens_out))
    s.set_attribute("creo.llm.latency_ms", latency_ms)
    s.set_attribute("creo.llm.valid", ok)
    s.set_attribute("creo.llm.reason", reason)


def record_wait(
    kind: str, run_id: str, start_ms: int, end_ms: int, *, scale: float = 1.0, **attrs: Any
) -> None:
    """A finished pause, as a span ending now.

    start_ms and end_ms are on the agent's clock, which the demo runs `scale`
    times faster than the wall from a fixed epoch. The span sits on the wall
    clock like every other span (ending now, lasting the wait's real time);
    creo.wait.seconds keeps the length on the agent's clock, which is what the
    runbook waited.
    """
    end_ns = time.time_ns()
    start_ns = end_ns - int((end_ms - start_ms) * 1_000_000 / scale)
    s = tracer.start_span(f"wait {kind}", start_time=start_ns)
    s.set_attribute(A.OPENINFERENCE_SPAN_KIND, Kind.CHAIN.value)
    s.set_attribute(A.SESSION_ID, run_id)
    s.set_attribute("creo.run_id", run_id)
    s.set_attribute("creo.wait.seconds", (end_ms - start_ms) / 1000)
    for k, v in attrs.items():
        if v is not None:
            s.set_attribute(k, v)
    s.end(end_time=end_ns)


def inject() -> dict[str, str]:
    """The current span's context as W3C headers, to carry inside a job."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def extracted(carrier: dict[str, str] | None) -> Iterator[None]:
    token = context.attach(propagate.extract(carrier or {}))
    try:
        yield
    finally:
        context.detach(token)
