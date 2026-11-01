"""Wire the nodes into a LangGraph state machine.

Every node names its successor in state["route"]; one conditional edge per
node follows it. The allowed successors of each node are listed here, so a
node can't jump somewhere the design doesn't allow: an unexpected route
fails the run loudly instead of silently taking another path.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from creo.agent.env import Env
from creo.agent.nodes import NODES
from creo.agent.state import RunState
from creo.catalog import Node

N = Node
EDGES: dict[Node, tuple[Node, ...]] = {
    N.INGEST: (N.SCREEN, N.FINALIZE),
    N.SCREEN: (N.DEDUPE, N.ESCALATE),
    N.DEDUPE: (N.TICKET,),
    N.TICKET: (N.GATHER_CONTEXT, N.ESCALATE, N.FINALIZE),
    N.GATHER_CONTEXT: (N.SELECT_RUNBOOK, N.RESOLVE, N.ESCALATE),
    N.SELECT_RUNBOOK: (N.DIAGNOSTICS, N.ESCALATE),
    N.DIAGNOSTICS: (N.CLASSIFY, N.ESCALATE),
    N.CLASSIFY: (N.DECIDE_WAIT, N.ESCALATE),
    N.DECIDE_WAIT: (N.WAIT, N.PROPOSE_FIX),
    N.WAIT: (N.RECHECK, N.RESOLVE),
    N.RECHECK: (N.WAIT, N.PROPOSE_FIX, N.RESOLVE, N.ESCALATE),
    N.PROPOSE_FIX: (N.APPROVAL, N.ESCALATE),
    N.APPROVAL: (N.EXECUTE, N.RESOLVE, N.ESCALATE),
    N.EXECUTE: (N.VERIFY, N.ESCALATE),
    N.VERIFY: (N.PROPOSE_FIX, N.RESOLVE, N.ESCALATE),
    N.RESOLVE: (N.FINALIZE,),
    N.ESCALATE: (N.FINALIZE,),
}


def _router(node: Node) -> Any:
    allowed = {str(n) for n in EDGES[node]}

    def route(state: RunState) -> str:
        nxt = str(state.get("route"))
        if nxt not in allowed:
            raise RuntimeError(f"{node} routed to {nxt!r}; allowed: {sorted(allowed)}")
        return nxt

    return route


def _plain(value: Any) -> Any:
    """Enum members become plain strings, so checkpoints hold only JSON-like data."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _node(fn: Callable[[RunState, Runtime[Env]], dict[str, Any]]) -> Any:
    def run(state: RunState, runtime: Runtime[Env]) -> dict[str, Any]:
        return {k: _plain(v) for k, v in fn(state, runtime).items()}

    run.__name__ = fn.__name__
    return run


def build_graph() -> StateGraph[RunState, Env, RunState, RunState]:
    g: StateGraph[RunState, Env, RunState, RunState] = StateGraph(RunState, context_schema=Env)
    for node, fn in NODES.items():
        g.add_node(str(node), _node(fn))
    g.add_edge(START, str(N.INGEST))
    for node, targets in EDGES.items():
        g.add_conditional_edges(str(node), _router(node), {str(t): str(t) for t in targets})
    g.add_edge(str(N.FINALIZE), END)
    return g


def compile_graph(checkpointer: Any) -> CompiledStateGraph[RunState, Env, RunState, RunState]:
    return build_graph().compile(checkpointer=checkpointer)


def mermaid() -> str:
    """The graph as a Mermaid diagram, for the README."""
    return compile_graph(None).get_graph().draw_mermaid()
