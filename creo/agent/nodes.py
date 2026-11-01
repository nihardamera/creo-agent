"""The workflow's nodes. Each returns its own name in `path` and the next node in `route`.

Only `classify` (for runbooks with judgement) and `escalate` (the summary)
call the model, and only with typed facts. Every model output is validated
before it is used, and every failure has a deterministic fallback.

Nodes that pause (diagnostics, wait, approval, execute) call interrupt().
On resume LangGraph re-runs the node from the top, so every side effect
before the interrupt is idempotent (fixed ids, INSERT OR IGNORE).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from creo import actions as act
from creo import approvals
from creo.agent.env import Env
from creo.agent.facts import classify_facts, evidence, summary_facts
from creo.agent.llm import LLMError, LLMRequest, parse_classify, parse_summary, template_summary
from creo.agent.screen import screen_alert, screen_diagnostics
from creo.agent.state import RunState
from creo.catalog import Action, FinalState, Node, Transition
from creo.config import FixStep, Host, Plan, Runbook
from creo.webhook import AlertWebhook, parse_tags

Out = dict[str, Any]
IGNORED_TRANSITIONS = {
    Transition.WARN,
    Transition.RE_WARN,
    Transition.NO_DATA,
    Transition.RE_NO_DATA,
}
SECURITY_REASONS = {
    "screen_flagged",
    "diagnostics_flagged",
    "tag_contradicts_inventory",
    "tag_names_missing_resource",
}
NOC_REASONS = {"unknown_monitor", "unknown_runbook", "runbook_conflict", "flapping", "unknown_host"}


def _alert(s: RunState) -> AlertWebhook:
    return AlertWebhook.model_validate(s["alert"])


def _host(s: RunState, env: Env) -> Host:
    host = env.inventory.host(s["host"])
    assert host is not None, "gather_context guarantees a known host"
    return host


def _rb(s: RunState, env: Env) -> Runbook:
    key = s.get("runbook")
    assert key, "select_runbook guarantees a runbook"
    return env.runbooks[key]


def _tags(s: RunState) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in parse_tags(_alert(s).tags):
        out.setdefault(k, []).append(v)
    return out


def _stop(node: Node, reason: str, route: Node = Node.ESCALATE, **more: Any) -> Out:
    return {"path": [node], "reason": reason, "route": route, **more}


def _ask(
    env: Env, run_id: str, node: Node, req: LLMRequest, parse: Callable[[str, dict[str, Any]], Any]
) -> Any:
    """One model call: validated output (None on any failure) and the reason, always logged."""
    out, why, resp = None, "unavailable", None
    try:
        resp = env.llm.complete(req)
        out, why = parse(resp.text, req.facts)
    except LLMError:
        pass
    env.store.log_llm(
        run_id,
        node,
        req.task,
        resp.model if resp else env.llm.name,
        out is not None,
        why,
        resp.input_tokens if resp else 0,
        resp.output_tokens if resp else 0,
        resp.latency_ms if resp else 0.0,
        req.facts,
    )
    return out, why


# ---- intake ------------------------------------------------------------------------


def ingest(s: RunState, runtime: Runtime[Env]) -> Out:
    env, a = runtime.context, _alert(s)
    now = s["received_ms"]
    if env.store.seen_event(a.id):
        return _stop(
            Node.INGEST, "redelivered_event", Node.FINALIZE, final_state=FinalState.DUPLICATE
        )
    env.store.add_event(a.id, s["run_id"], now)
    if now - a.date > env.policy.stale_after_minutes * 60_000:
        return _stop(Node.INGEST, "stale", Node.FINALIZE, final_state=FinalState.IGNORED)
    if a.transition in IGNORED_TRANSITIONS:
        return _stop(Node.INGEST, "not_actionable", Node.FINALIZE, final_state=FinalState.IGNORED)
    if a.transition is Transition.RECOVERED:  # the ingress routes Recovered for open runs elsewhere
        return _stop(Node.INGEST, "orphan_recovery", Node.FINALIZE, final_state=FinalState.IGNORED)
    return {
        "path": [Node.INGEST],
        "route": Node.SCREEN,
        "host": a.hostname,
        "monitor_id": a.alert_id,
    }


def screen(s: RunState, runtime: Runtime[Env]) -> Out:
    findings = screen_alert(_alert(s), runtime.context.policy)
    if findings:
        return _stop(Node.SCREEN, "screen_flagged", flags=[f"{f.code}@{f.field}" for f in findings])
    return {"path": [Node.SCREEN], "route": Node.DEDUPE}


def _incident_key(s: RunState, env: Env) -> str | None:
    host, tags = env.inventory.host(s["host"]), _tags(s)
    if host is None:
        return None
    for key in ("service", "app_pool"):
        if tags.get(key):
            return f"{host.site}|{key}:{tags[key][0]}"
    return None


def dedupe(s: RunState, runtime: Runtime[Env]) -> Out:
    env, now, run = runtime.context, s["received_ms"], s["run_id"]
    other = env.store.find_open_run(s["monitor_id"], s["host"], exclude=run)
    if other:
        return _stop(
            Node.DEDUPE, "duplicate", Node.TICKET, final_state=FinalState.DUPLICATE, link_run=other
        )
    window = env.policy.flapping.window_minutes * 60_000
    flapping = (
        env.store.count_transitions(s["monitor_id"], s["host"], now - window)
        >= env.policy.flapping.transitions
    )
    key = _incident_key(s, env)
    if not flapping and key:
        since = now - env.policy.correlation_window_minutes * 60_000
        incident = env.store.find_incident(key, since, exclude=run)
        if incident:
            return _stop(
                Node.DEDUPE,
                "correlated",
                Node.TICKET,
                final_state=FinalState.CORRELATED,
                link_run=incident,
            )
    env.store.open_run(run, s["monitor_id"], s["host"], key, now)
    out: Out = {"path": [Node.DEDUPE], "route": Node.TICKET, "opened": True, "link_run": None}
    if flapping:
        out["reason"] = "flapping"
    return out


def ticket(s: RunState, runtime: Runtime[Env]) -> Out:
    env, a = runtime.context, _alert(s)
    linked = s.get("link_run")
    if linked:
        tid = env.store.ticket_of(linked)
        if tid:
            env.tickets.comment(tid, f"Related alert {a.id} ({s.get('reason')}): {a.title}")
        return {"path": [Node.TICKET], "route": Node.FINALIZE, "ticket_id": tid}
    tid = env.tickets.create(
        title=a.title,
        body=a.message,
        host=a.hostname,
        priority=a.alert_priority or "P3",
        source_run=s["run_id"],
    )
    env.store.set_ticket(s["run_id"], tid)
    nxt = Node.ESCALATE if s.get("reason") == "flapping" else Node.GATHER_CONTEXT
    return {"path": [Node.TICKET], "route": nxt, "ticket_id": tid}


def gather_context(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    if env.inventory.host(s["host"]) is None:
        return _stop(Node.GATHER_CONTEXT, "unknown_host")
    if env.monitoring.status(s["monitor_id"], s["host"]) == "OK":
        return _stop(Node.GATHER_CONTEXT, "recovered_before_triage", Node.RESOLVE)
    return {"path": [Node.GATHER_CONTEXT], "route": Node.SELECT_RUNBOOK}


# ---- deciding what applies ------------------------------------------------------------

RESOURCE_TAGS = {
    "service": "services",
    "app_pool": "app_pools",
    "task": "tasks",
    "unit": "units",
    "consumer": "consumers",
    "device": "drives",
}


def select_runbook(s: RunState, runtime: Runtime[Env]) -> Out:
    env, tags = runtime.context, _tags(s)
    host = _host(s, env)
    me = Node.SELECT_RUNBOOK
    # Tags never override inventory. A contradiction is suspicious, not a typo to fix.
    facts = {"role": host.role, "tier": str(host.tier), "site": host.site, "env": host.env}
    for key, truth in facts.items():
        if any(v != truth for v in tags.get(key, [])):
            return _stop(me, "tag_contradicts_inventory", flags=[f"{key}_mismatch"])
    for key, field in RESOURCE_TAGS.items():
        have = getattr(host, field)
        if any(v not in have for v in tags.get(key, [])):
            return _stop(me, "tag_names_missing_resource", flags=[f"{key}_not_on_host"])
    books, types = tags.get("runbook", []), tags.get("monitor_type", [])
    if len(books) > 1 or len(types) > 1:
        return _stop(me, "runbook_conflict")
    if books:
        rb = env.runbooks.get(books[0])
        if rb is None:
            return _stop(me, "unknown_runbook")
        if (types and types[0] not in rb.monitor_types) or host.role not in rb.roles:
            return _stop(me, "runbook_conflict")
    else:
        matches = [
            r
            for r in env.runbooks.values()
            if types and types[0] in r.monitor_types and host.role in r.roles
        ]
        if len(matches) != 1:
            return _stop(me, "unknown_monitor")
        rb = matches[0]
    return {"path": [me], "route": Node.DIAGNOSTICS, "runbook": rb.key}


def diagnostics(s: RunState, runtime: Runtime[Env]) -> Out:
    env, run = runtime.context, s["run_id"]
    rb = _rb(s, env)
    ids = []
    for name in rb.diagnostics:
        jid = f"{run}:diag:{name.value}"
        env.store.add_job(
            jid, run, "diagnostic", {"host": s["host"], "name": name.value}, s["received_ms"]
        )
        ids.append(jid)
    results: dict[str, dict[str, Any]] = interrupt({"kind": "jobs", "job_ids": ids})
    outputs = {
        jid.rsplit(":", 1)[1]: r.get("output")
        for jid, r in results.items()
        if r.get("status") == "ok"
    }
    findings = screen_diagnostics(outputs, env.policy)
    if findings:
        flags = [f"{f.code}@{f.field}" for f in findings]
        return _stop(Node.DIAGNOSTICS, "diagnostics_flagged", flags=flags, diagnostics=outputs)
    return {"path": [Node.DIAGNOSTICS], "route": Node.CLASSIFY, "diagnostics": outputs}


def classify(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    rb, host = _rb(s, env), _host(s, env)
    if not rb.judgement:
        category, cited = "default", []
    else:
        facts = classify_facts(rb, host, s["diagnostics"], env.policy)
        out, why = _ask(
            env, s["run_id"], Node.CLASSIFY, LLMRequest("classify", facts), parse_classify
        )
        if out is None:
            return _stop(Node.CLASSIFY, "classification_invalid", flags=[f"llm_{why}"])
        category, cited = out.category, out.evidence
    plan = rb.categories[category].plan
    route = Node.ESCALATE if plan is Plan.ESCALATE_NOW else Node.DECIDE_WAIT
    out_: Out = {"path": [Node.CLASSIFY], "route": route, "category": category, "cited": cited}
    if plan is Plan.ESCALATE_NOW:
        out_["reason"] = "runbook_escalates"
    return out_


def decide_wait(s: RunState, runtime: Runtime[Env]) -> Out:
    plan = _rb(s, runtime.context).categories[s["category"] or "default"].plan
    nxt = Node.WAIT if plan.waits else Node.PROPOSE_FIX
    return {
        "path": [Node.DECIDE_WAIT],
        "route": nxt,
        "waits": 0,
        "fix_index": 0,
        "wait_anchor_ms": runtime.context.clock.now_ms(),
    }


# ---- waiting -------------------------------------------------------------------------


def wait(s: RunState, runtime: Runtime[Env]) -> Out:
    rb = _rb(s, runtime.context)
    k = s.get("waits", 0)
    due = s["wait_anchor_ms"] + (rb.wait.minutes + k * rb.wait.interval_minutes) * 60_000
    woke: dict[str, str] = interrupt({"kind": "wait", "due_ms": due})
    if woke.get("reason") == "recovered":
        return {
            "path": [Node.WAIT],
            "route": Node.RESOLVE,
            "waits": k + 1,
            "reason": "recovered_webhook",
        }
    return {"path": [Node.WAIT], "route": Node.RECHECK, "waits": k + 1}


def _after_waiting(s: RunState, env: Env) -> Out:
    """No recovery and no waits left: fix if the plan and policy allow it, else escalate."""
    plan = _rb(s, env).categories[s["category"] or "default"].plan
    if not plan.fixes:
        return {"route": Node.ESCALATE, "reason": "did_not_recover"}
    if _host(s, env).tier in env.policy.no_fix_tiers:
        return {"route": Node.ESCALATE, "reason": "no_fix_tier"}
    return {"route": Node.PROPOSE_FIX}


def recheck(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    if env.monitoring.status(s["monitor_id"], s["host"]) == "OK":
        return _stop(Node.RECHECK, "recovered_on_recheck", Node.RESOLVE)
    if s["waits"] < _rb(s, env).wait.rechecks:
        return {"path": [Node.RECHECK], "route": Node.WAIT}
    return {"path": [Node.RECHECK], **_after_waiting(s, env)}


# ---- changing something ------------------------------------------------------------


def _cited(s: RunState, env: Env, kind: str) -> list[dict[str, Any]]:
    ev = evidence(_rb(s, env), s["diagnostics"], env.policy)
    wanted = set(s.get("cited") or [])
    return [e for e in ev if e["id"] in wanted and e["kind"] == kind]


def bind(fix: FixStep, s: RunState, env: Env) -> dict[str, Any] | None:
    """Exact parameters for a fix, from inventory-checked tags or cited evidence."""
    tags, cat = _tags(s), s.get("category")
    params: dict[str, Any] = {"host": s["host"]}

    def one(values: list[Any]) -> Any:
        return values[0] if len(values) == 1 else None

    match fix.action:
        case Action.RESTART_WINDOWS_SERVICE:
            if cat == "service_leak":
                params["service_name"] = one([p["owner"] for p in _cited(s, env, "process")])
            elif cat == "service_crash":
                params["service_name"] = one([v["service"] for v in _cited(s, env, "service")])
            else:
                params["service_name"] = one(tags.get("service", []))
        case Action.START_APP_POOL:
            params["app_pool"] = one(tags.get("app_pool", []))
        case Action.RECYCLE_APP_POOL:
            params["app_pool"] = one([p["owner"] for p in _cited(s, env, "process")])
        case Action.CLEAR_TEMP_FILES:
            params["drive"] = one(tags.get("device", []))
        case Action.KILL_PROCESS:
            procs = _cited(s, env, "process")
            p = one(procs)
            if p is None or p["image"] not in _rb(s, env).allowed_processes:
                return None
            params["process_name"], params["pid"] = p["image"], p["pid"]
        case Action.RERUN_SCHEDULED_TASK:
            params["task_name"] = one(tags.get("task", []))
        case Action.RESTART_SYSTEMD_UNIT:
            params["unit"] = one(tags.get("unit", []))
        case Action.RESTART_QUEUE_CONSUMER:
            params["consumer"] = one(tags.get("consumer", []))
    if any(v is None for v in params.values()):
        return None
    params.update(fix.params)
    return params


def propose_fix(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    rb, host = _rb(s, env), _host(s, env)
    fixes, i = rb.fixes_for(s["category"] or "default", host.role), s.get("fix_index", 0)
    me = Node.PROPOSE_FIX
    if env.store.killswitch():
        return _stop(me, "killswitch")
    if i >= len(fixes):
        return _stop(me, "fixes_exhausted")
    params = bind(fixes[i], s, env)
    if params is None:
        return _stop(me, "cannot_bind_parameters")
    try:
        action, typed = act.parse(fixes[i].action.value, params)
        act.check_inventory(action, typed, env.inventory)
    except act.ActionRefused as e:
        return _stop(me, "proposal_refused", flags=[e.code])
    now = env.clock.now_ms()
    req = approvals.ApprovalRequest(
        approval_id=f"{s['run_id']}:fix{i}",
        run_id=s["run_id"],
        action=action.value,
        params=typed.model_dump(),
        requested_ms=now,
        expires_ms=now + env.policy.approval_ttl_minutes * 60_000,
    )
    env.store.add_approval(req)
    proposal = {"action": req.action, "params": req.params}
    return {
        "path": [me],
        "route": Node.APPROVAL,
        "proposal": proposal,
        "approval_id": req.approval_id,
    }


def approval(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    found = env.store.approval(s["approval_id"] or "")
    assert found is not None
    req, _ = found
    answer: dict[str, Any] = interrupt(
        {"kind": "approval", "approval_id": req.approval_id, "due_ms": req.expires_ms}
    )
    decision = answer.get("decision")
    me = Node.APPROVAL
    if decision == "withdrawn":  # the monitor recovered while waiting for a human
        return _stop(me, "recovered_webhook", Node.RESOLVE)
    if decision != "approve":
        return _stop(me, "approval_denied" if decision == "deny" else "approval_timeout")
    grant = approvals.Grant.model_validate(answer.get("grant"))
    problem = approvals.check(
        env.approval_key,
        grant,
        run_id=s["run_id"],
        action=req.action,
        params=req.params,
        now_ms=env.clock.now_ms(),
        approvers=env.policy.approvers,
    )
    if problem:
        return _stop(me, "approval_invalid", flags=[problem])
    return {"path": [me], "route": Node.EXECUTE}


def execute(s: RunState, runtime: Runtime[Env]) -> Out:
    env, run = runtime.context, s["run_id"]
    proposal, i = s["proposal"] or {}, s.get("fix_index", 0)
    jid = f"{run}:act{i}"
    grant_json = next(
        a["grant_json"] for a in env.store.approvals(run) if a["approval_id"] == s["approval_id"]
    )
    env.store.add_job(
        jid, run, "action", {**proposal, "run_id": run, "grant": grant_json}, env.clock.now_ms()
    )
    results: dict[str, dict[str, Any]] = interrupt({"kind": "jobs", "job_ids": [jid]})
    r = results[jid]
    done = [{**proposal, "result": r.get("status"), "code": r.get("code")}]
    if r.get("status") != "ok":
        reason = "executor_error" if r.get("status") == "error" else "executor_refused"
        return _stop(Node.EXECUTE, reason, actions=done)
    return {"path": [Node.EXECUTE], "route": Node.VERIFY, "actions": done}


def verify(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    if env.monitoring.status(s["monitor_id"], s["host"]) == "OK":
        return _stop(Node.VERIFY, "fixed", Node.RESOLVE)
    nxt = s.get("fix_index", 0) + 1
    if nxt >= len(_rb(s, env).fixes_for(s["category"] or "default", _host(s, env).role)):
        return _stop(Node.VERIFY, "fixes_exhausted")
    return {"path": [Node.VERIFY], "route": Node.PROPOSE_FIX, "fix_index": nxt}


# ---- endings -------------------------------------------------------------------------


def resolve(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    tid = s.get("ticket_id")
    if tid:
        env.tickets.comment(tid, f"Resolved by automation: {s.get('reason')}.")
        env.tickets.transition(tid, "resolved")
    return {"path": [Node.RESOLVE], "route": Node.FINALIZE, "final_state": FinalState.RESOLVED}


def escalate(s: RunState, runtime: Runtime[Env]) -> Out:
    env, run = runtime.context, s["run_id"]
    reason = s.get("reason") or "unknown"
    rb = env.runbooks.get(s.get("runbook") or "")
    if reason in SECURITY_REASONS:
        group = env.policy.security_escalation.group
    elif reason in NOC_REASONS or rb is None:
        group = env.policy.default_escalation.group
    else:
        group = rb.escalation.group
    host = env.inventory.host(s.get("host") or "")
    facts = summary_facts(
        host=s.get("host") or _alert(s).hostname,
        role=host.role if host else None,
        tier=host.tier if host else None,
        runbook=rb.key if rb else None,
        reason=reason,
        flags=sorted(set(s.get("flags", []))),
        category=s.get("category"),
        steps=list(s.get("path", [])),
        actions=[
            {"action": a["action"], "result": str(a.get("result"))} for a in s.get("actions", [])
        ],
    )
    summary, _ = _ask(env, run, Node.ESCALATE, LLMRequest("summarize", facts), parse_summary)
    note = summary or template_summary(facts)
    tid = s.get("ticket_id")
    if not tid:
        a = _alert(s)
        tid = env.tickets.create(
            title=a.title,
            body=a.message,
            host=a.hostname,
            priority=a.alert_priority or "P3",
            source_run=run,
        )
    env.tickets.comment(tid, f"{note.summary}\nNext: {note.next_step}")
    env.tickets.transition(tid, "escalated", group=group)
    return {
        "path": [Node.ESCALATE],
        "route": Node.FINALIZE,
        "final_state": FinalState.ESCALATED,
        "escalation_group": group,
        "ticket_id": tid,
        "summary": {
            "summary": note.summary,
            "next_step": note.next_step,
            "source": "model" if summary else "template",
        },
    }


def finalize(s: RunState, runtime: Runtime[Env]) -> Out:
    env = runtime.context
    final = s.get("final_state") or FinalState.ESCALATED
    if s.get("opened"):
        env.store.close_run(s["run_id"], str(final), env.clock.now_ms())
    return {"path": [Node.FINALIZE], "route": "__end__", "final_state": final}


NODES = {
    Node.INGEST: ingest,
    Node.SCREEN: screen,
    Node.DEDUPE: dedupe,
    Node.TICKET: ticket,
    Node.GATHER_CONTEXT: gather_context,
    Node.SELECT_RUNBOOK: select_runbook,
    Node.DIAGNOSTICS: diagnostics,
    Node.CLASSIFY: classify,
    Node.DECIDE_WAIT: decide_wait,
    Node.WAIT: wait,
    Node.RECHECK: recheck,
    Node.PROPOSE_FIX: propose_fix,
    Node.APPROVAL: approval,
    Node.EXECUTE: execute,
    Node.VERIFY: verify,
    Node.RESOLVE: resolve,
    Node.ESCALATE: escalate,
    Node.FINALIZE: finalize,
}
