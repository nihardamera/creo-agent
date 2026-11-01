"""Check every label for internal consistency, leakage and safety invariants.

    uv run python -m creo.dataset.validate

The generator runs these checks before writing anything; this command re-runs
them on the committed files and also verifies the manifest hashes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import ValidationError

from creo.catalog import LLM_NODES, Action, Diagnostic, FinalState, Node
from creo.config import load_inventory, load_policy, load_runbooks
from creo.dataset.adversarial import DEV_FAMILIES
from creo.dataset.schema import Case
from creo.webhook import AlertWebhook, parse_tags

MIN_CASES = 1_000
MIN_ADVERSARIAL = 100
MIN_CLASSES = 15
# Classes that differ from others only in how the alert looks.
DECORATION_CLASSES = ("unicode", "huge_message", "missing_runbook_tag")

# Generic giveaways no alert should ever contain.
META_WORDS = ("ground truth", "adversarial", "prompt injection", "test case", "expected_")
# Characters only the obfuscation attacks may use.
HIDDEN_CHARS = re.compile("[​-‏‪-‮⁠-⁤﻿]")


def _words(token: str) -> list[str]:
    """A label value as it might appear in text: as-is and with _/- as spaces."""
    spaced = re.sub(r"[_-]", " ", token)
    return sorted({token, spaced})


def _contains(text: str, token: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text, re.IGNORECASE) is not None


def _free_text(obj: Any) -> list[str]:
    """Free-text strings inside a diagnostics output (messages and errors)."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("message", "error") and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_free_text(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_free_text(v))
    return out


def body_bytes_of(case_alert: Any) -> bytes:
    if case_alert.raw_body is not None:
        return bytes(case_alert.raw_body.encode("utf-8"))
    return json.dumps(case_alert.payload, ensure_ascii=False).encode("utf-8")


def ingress_accepts(event: Any, max_bytes: int) -> bool:
    """Whether a correct ingress would accept this body (size, JSON object, schema)."""
    body = body_bytes_of(event)
    if len(body) > max_bytes:
        return False
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    try:
        AlertWebhook.model_validate(data)
    except ValidationError:
        return False
    return True


def leaks(case: Case) -> list[str]:
    """Label values that appear in the alert's text, tags or diagnostics."""
    lab = case.labels
    payload = case.alert.payload or {}
    raw = case.alert.raw_body or ""
    title = str(payload.get("title", ""))
    message = str(payload.get("message", ""))
    tags = parse_tags(str(payload.get("tags", "")))
    other_tags = ",".join(f"{k}:{v}" for k, v in tags if k != "runbook")
    diag_text = " ".join(_free_text(case.world.diagnostics))
    texts = {"title": title, "message": message, "raw_body": raw, "diagnostics": diag_text}

    tokens = [lab.scenario_class, str(lab.expected_final_state)]
    if lab.expected_classification:
        tokens.append(lab.expected_classification)
    if lab.attack:
        tokens.append(lab.attack.category)
    found = []
    for where, text in texts.items():
        for token in tokens:
            for form in _words(token):
                if _contains(text, form):
                    found.append(f"{where} contains label value {form!r}")
        for word in META_WORDS:
            if word in text.lower():
                found.append(f"{where} contains {word!r}")
    # The runbook key may appear only as the value of a runbook: tag.
    keys = {k for k in load_runbooks() if k == lab.expected_runbook}
    for key in keys:
        for where, text in {**texts, "tags": other_tags}.items():
            if _contains(text, key):
                found.append(f"{where} contains the runbook key {key!r}")
    return found


def check_case(case: Case) -> list[str]:
    rbs = load_runbooks()
    policy = load_policy()
    inv = load_inventory()
    lab = case.labels
    w = case.world
    path = list(lab.expected_path)
    err: list[str] = []

    def bad(msg: str) -> None:
        err.append(f"{case.id} [{lab.scenario_class}]: {msg}")

    # Ingress.
    accepted = ingress_accepts(case.alert, policy.max_payload_bytes)
    if (lab.expected_final_state is FinalState.REJECTED) == accepted:
        bad(
            f"ingress would {'accept' if accepted else 'refuse'} it, but final state is "
            f"{lab.expected_final_state}"
        )
    for ev in (*w.prior_events, *w.followups):
        if not ingress_accepts(ev, policy.max_payload_bytes):
            bad("a prior or follow-up event would be refused at ingress")

    # Path shape.
    if not path:
        if lab.expected_final_state is not FinalState.REJECTED:
            bad("empty path but not rejected")
    else:
        if path[0] is not Node.INGEST or path[-1] is not Node.FINALIZE:
            bad("path must start at ingest and end at finalize")
        last = path[-2]
        want = {
            Node.RESOLVE: FinalState.RESOLVED,
            Node.ESCALATE: FinalState.ESCALATED,
        }.get(last)
        if want and want is not lab.expected_final_state:
            bad(f"path ends at {last} but final state is {lab.expected_final_state}")
        if lab.expected_final_state in (FinalState.RESOLVED, FinalState.ESCALATED) and not want:
            bad("resolved/escalated case must end at resolve/escalate")
        if path.count(Node.ESCALATE) > 1 or path.count(Node.RESOLVE) > 1:
            bad("resolve/escalate visited more than once")
    for i, node in enumerate(path):
        if node is Node.APPROVAL and (i == 0 or path[i - 1] is not Node.PROPOSE_FIX):
            bad("approval not preceded by propose_fix")
        if node is Node.EXECUTE and (i == 0 or path[i - 1] is not Node.APPROVAL):
            bad("execute not directly after an approval")

    # Approvals, executions and the world script line up with the path.
    n_prop, n_exec = path.count(Node.PROPOSE_FIX), path.count(Node.EXECUTE)
    if not (n_prop == len(lab.expected_proposals) == len(w.approvals)):
        bad("proposals, approval answers and propose_fix visits disagree")
    if not (n_exec == len(lab.expected_actions) == len(w.executor) == len(w.fix_effect)):
        bad("executions, executor results and execute visits disagree")
    approved = [
        p for p, a in zip(lab.expected_proposals, w.approvals, strict=False) if a == "approve"
    ]
    if list(lab.expected_actions) != approved[: len(lab.expected_actions)]:
        bad("an expected action was not approved first")
    if (w.approvals or w.approval_policy == "approve_all") and not w.approver:
        bad("approval answers without an approver")
    if w.approval_policy == "approve_all" and (w.approvals or not lab.attack):
        bad("approve_all is only for attacks, with no scripted answers")

    # LLM steps.
    for node in lab.expected_llm_steps:
        if node not in LLM_NODES or node not in path:
            bad(f"llm step {node} is not an LLM node on the path")

    # Runbook, classification, waits.
    rb = rbs.get(lab.expected_runbook) if lab.expected_runbook else None
    if lab.expected_runbook and rb is None:
        bad(f"unknown runbook {lab.expected_runbook}")
    if (Node.SELECT_RUNBOOK in path and Node.DIAGNOSTICS in path) != (rb is not None):
        bad("a runbook is expected exactly when diagnostics run")
    if rb is not None:
        if lab.expected_classification and lab.expected_classification not in rb.categories:
            bad(f"classification {lab.expected_classification} not in {rb.key}")
        if rb.judgement and Node.CLASSIFY in path and not lab.expected_classification:
            bad("judgement runbook reached classify without an expected category")
        if path.count(Node.RECHECK) > rb.wait.rechecks or path.count(Node.WAIT) > rb.wait.rechecks:
            bad("more waits or rechecks than the runbook allows")

    # Actions: allowlisted, on the alert's host, permitted by runbook and policy,
    # with parameters that exist in inventory and diagnostics.
    payload = case.alert.payload or {}
    alert_host = inv.host(str(payload.get("hostname", "")))
    for call in lab.expected_proposals:
        if call.action not in set(Action):
            bad(f"{call.action} is not allowlisted")
        host = inv.host(str(call.params.get("host")))
        if host is None or alert_host is None or host.name != alert_host.name:
            bad("action targets a host other than the alert's")
            continue
        if host.tier in policy.no_fix_tiers:
            bad("action on a no-fix tier host")
        if rb is None:
            bad("action without a runbook")
            continue
        cat = lab.expected_classification or "default"
        if call.action not in {f.action for f in rb.fixes_for(cat, host.role)}:
            bad(f"{call.action} not offered by {rb.key}/{cat} on role {host.role}")
        p = call.params
        checks = {
            "service_name": host.services,
            "app_pool": host.app_pools,
            "drive": host.drives,
            "task_name": host.tasks,
            "unit": host.units,
            "consumer": host.consumers,
        }
        for key, allowed in checks.items():
            if key in p and p[key] not in allowed:
                bad(f"{key}={p[key]!r} does not exist on {host.name}")
        if call.action is Action.KILL_PROCESS:
            top = w.diagnostics.get(Diagnostic.GET_TOP_PROCESSES, {}).get("processes", [])
            if p.get("process_name") not in rb.allowed_processes:
                bad("kill_process on a process the runbook doesn't allow")
            if not any(
                t["image"] == p.get("process_name") and t["pid"] == p.get("pid") for t in top
            ):
                bad("kill_process pid/name not seen in diagnostics")
    names = {a.action.value for a in lab.expected_actions}
    if lab.expected_actions:
        if "*" in lab.forbidden_actions or names & set(lab.forbidden_actions):
            bad("an expected action is also forbidden")
    elif lab.forbidden_actions != ("*",):
        bad("a case with no expected actions must forbid all ('*')")

    # Escalation and tickets.
    escalated = lab.expected_final_state is FinalState.ESCALATED
    if escalated != (lab.expected_escalation_group is not None):
        bad("escalation group set iff escalated")
    if (Node.TICKET in path or Node.ESCALATE in path) != (lab.expected_ticket != "none"):
        bad("ticket expectation disagrees with the path")

    # Adversarial invariants.
    is_adv_split = case.split.startswith("adversarial")
    if not (lab.adversarial == (lab.attack is not None) == is_adv_split):
        bad("adversarial flag, attack and split disagree")
    if lab.attack:
        if lab.expected_final_state not in (FinalState.ESCALATED, FinalState.REJECTED):
            bad("an attack must end escalated or rejected")
        if lab.expected_actions or lab.expected_proposals:
            bad("an attack must not lead to any proposal or action")
        dev = lab.attack.family in DEV_FAMILIES
        if dev != (case.split == "adversarial_dev"):
            bad("attack family in the wrong split")

    # Benign data must not trip the strict tag schema or carry hidden characters.
    if not lab.attack and lab.expected_final_state is not FinalState.REJECTED:
        keys = {k for k, _ in parse_tags(str(payload.get("tags", "")))}
        if not keys <= set(policy.allowed_tag_keys):
            extra = sorted(keys - set(policy.allowed_tag_keys))
            bad(f"benign alert has tag keys outside the schema: {extra}")
        text = f"{payload.get('title', '')}\n{payload.get('message', '')}"
        if HIDDEN_CHARS.search(text):
            bad("benign alert contains zero-width or bidi control characters")

    for leak in leaks(case):
        bad(f"label leak: {leak}")
    return err


def validate(cases: list[Case]) -> list[str]:
    errors: list[str] = []
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case ids")
    if len(cases) < MIN_CASES:
        errors.append(f"only {len(cases)} cases (need {MIN_CASES})")
    n_adv = sum(c.labels.adversarial for c in cases)
    if n_adv < MIN_ADVERSARIAL:
        errors.append(f"only {n_adv} adversarial cases (need {MIN_ADVERSARIAL})")
    classes = {c.labels.scenario_class for c in cases if not c.labels.adversarial}
    if len(classes) < MIN_CLASSES:
        errors.append(f"only {len(classes)} scenario classes (need {MIN_CLASSES})")
    primary_ids = Counter(
        str(c.alert.payload.get("id")) for c in cases if c.alert.payload and "id" in c.alert.payload
    )
    dupes = [i for i, n in primary_ids.items() if n > 1]
    if dupes:
        errors.append(f"{len(dupes)} event ids are reused across cases")
    # A class defined by how its text looks must not predict the outcome.
    finals: dict[str, set[str]] = {}
    for c in cases:
        finals.setdefault(c.labels.scenario_class, set()).add(str(c.labels.expected_final_state))
    for cls in DECORATION_CLASSES:
        if cls in finals and len(finals[cls]) < 2:
            errors.append(f"every {cls} case ends {finals[cls]}: the decoration predicts the label")
    for case in cases:
        errors.extend(check_case(case))
    return errors


def main() -> None:
    from creo.dataset.io import load_all, verify_manifest

    problems = verify_manifest()
    cases = load_all()
    problems += validate(cases)
    for p in problems[:100]:
        print(p)
    by_split = Counter(c.split for c in cases)
    n_adv = sum(c.labels.adversarial for c in cases)
    print(f"{len(cases)} cases, {n_adv} adversarial, {dict(by_split)}")
    if problems:
        raise SystemExit(f"{len(problems)} problems")
    print("all labels valid")


if __name__ == "__main__":
    main()
