"""Spans for every node, model call, executor job and wait; and no raw alert text in any of them."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.harness import Outcome, run_case
from tests.oracle import LabelOracleLLM

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider() -> Iterator[None]:
    current = trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):  # set once per process; OTel forbids resetting
        p = TracerProvider()
        p.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(p)
    yield


def _run(cls: str, judgement: bool = False) -> tuple[Case, Outcome, list[ReadableSpan]]:
    case = next(
        c
        for c in load("dev", purpose="tune")
        if c.labels.scenario_class == cls and bool(c.labels.expected_classification) == judgement
    )
    EXPORTER.clear()
    out = run_case(case, LabelOracleLLM(case))
    spans = [
        s
        for s in EXPORTER.get_finished_spans()
        if (s.attributes or {}).get("creo.run_id") == out.run_id
    ]
    return case, out, spans


def _kind(s: ReadableSpan) -> str:
    return str((s.attributes or {}).get("openinference.span.kind"))


def test_every_node_visit_has_a_span() -> None:
    _, out, spans = _run("fix_succeeds", judgement=True)
    finished = [
        s.name.removeprefix("node ")
        for s in spans
        if s.name.startswith("node ") and not (s.attributes or {}).get("creo.paused")
    ]
    assert sorted(finished) == sorted(out.path)


def test_model_calls_are_llm_spans_with_tokens() -> None:
    _, out, spans = _run("fix_succeeds", judgement=True)
    llm = [s for s in spans if _kind(s) == "LLM"]
    assert len(llm) == len(out.llm_calls) >= 1
    for s in llm:
        a = s.attributes or {}
        assert a["llm.token_count.prompt"] > 0 and "llm.model_name" in a and "llm.cost.total" in a


def test_executor_spans_join_the_agent_trace() -> None:
    _, _, spans = _run("fix_succeeds")
    tools = [s for s in spans if _kind(s) == "TOOL"]
    nodes = {s.context.span_id: s for s in spans if s.name.startswith("node ")}
    assert tools
    for t in tools:
        assert t.parent is not None and t.parent.span_id in nodes  # the node that queued the job
        assert t.context.trace_id == nodes[t.parent.span_id].context.trace_id


def test_waits_are_spans_with_their_real_duration() -> None:
    _, _, spans = _run("never_recovers_no_fix")
    waits = [s for s in spans if s.name == "wait wait"]
    assert waits and all((s.attributes or {})["creo.wait.seconds"] > 0 for s in waits)


def test_no_span_carries_raw_alert_text() -> None:
    for c in load("adversarial_dev", purpose="tune")[:30] + load("dev", purpose="tune")[:30]:
        EXPORTER.clear()
        run_case(c, LabelOracleLLM(c))
        p = c.alert.payload or {}
        raw = [str(p.get("title", "")), str(p.get("message", ""))]
        for s in EXPORTER.get_finished_spans():
            blob = json.dumps(dict(s.attributes or {}))
            for text in raw:
                for line in (x.strip() for x in text.splitlines()):
                    if len(line) >= 16:
                        assert line not in blob, (c.id, s.name, line[:60])


def test_a_wait_span_sits_on_the_wall_clock_even_when_agent_time_runs_ahead() -> None:
    import time

    from creo import tracing

    EXPORTER.clear()
    far_future = 2_000_000_000_000  # agent clock years ahead, as in the 60x demo
    tracing.record_wait("wait", "run-x", far_future, far_future + 600_000, scale=60.0)
    (s,) = EXPORTER.get_finished_spans()
    assert s.start_time is not None and s.end_time is not None
    assert abs(s.end_time / 1e9 - time.time()) < 5
    assert (s.end_time - s.start_time) / 1e9 == 10.0  # 10 agent minutes at 60x
    assert (s.attributes or {})["creo.wait.seconds"] == 600.0
