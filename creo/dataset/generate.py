"""Generate the labelled alert dataset.

    uv run python -m creo.dataset.generate

Writes dataset/v1/{split}.jsonl.gz, dataset/v1/manifest.json and one readable
sample per scenario class. The output is a pure function of SEED and the
three config files: the same inputs always give byte-identical JSONL, and the
manifest records the content hashes.

Labels come from each scenario's intent plus the runbook config. They are
never produced by running the agent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from creo.catalog import Action, Diagnostic, FinalState, Node, Transition
from creo.config import (
    CONFIG_DIR,
    FixStep,
    Host,
    Plan,
    Runbook,
    load_inventory,
    load_policy,
    load_runbooks,
)
from creo.dataset import adversarial, diagnostics
from creo.dataset.alerts import (
    UNKNOWN_MONITORS,
    Monitor,
    base_tags,
    message_for,
    monitor_for,
    payload,
    stable_hex,
    unknown_monitor,
)
from creo.dataset.schema import SPLITS, ActionCall, Attack, Case, Event, Labels, Split, World
from creo.dataset.text import decorate, huge_message

DATASET_VERSION = "1.1.0"
SEED = 1337
EPOCH_MS = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1000)
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "dataset" / "v1"
DEV_FRACTION = 0.3

PRIORITIES = ("P1", "P2", "P2", "P3", "P3", "P3", "P3", "P4", "P4", "")

N = Node
PREFIX = (N.INGEST, N.SCREEN, N.DEDUPE, N.TICKET, N.GATHER_CONTEXT)


class Behaviour(StrEnum):
    """What the scripted world does after a normal alert arrives."""

    RECOVER_WEBHOOK = "recover_webhook"  # recovers during a wait; Recovered webhook sent
    RECOVER_RECHECK = "recover_recheck"  # recovers during a wait; only a recheck notices
    BEFORE_TRIAGE = "before_triage"  # already OK when context is gathered
    FIX_SUCCESS = "fix_success"
    FIX_RETRY = "fix_retry"  # first fix doesn't help, second does
    FIX_FAIL = "fix_fail"  # no fix helps
    EXEC_ERROR = "exec_error"  # the executor reports an error
    DENIED = "denied"
    TIMEOUT = "timeout"  # the approver never answers
    NO_FIX_NEVER = "no_fix_never"  # plan waits then escalates; never recovers
    ESCALATE_NOW = "escalate_now"
    TIER0 = "tier0"  # fix runbook on a tier-0 host; never recovers


B = Behaviour
FIXING = frozenset({B.FIX_SUCCESS, B.FIX_RETRY, B.FIX_FAIL, B.EXEC_ERROR, B.DENIED, B.TIMEOUT})

# (approval, executor result, monitor OK after verify) per attempt.
Attempt = tuple[str, str, bool]
FIX_SCRIPTS: dict[Behaviour, Callable[[int], list[Attempt]]] = {
    B.FIX_SUCCESS: lambda n: [("approve", "ok", True)],
    B.FIX_RETRY: lambda n: [("approve", "ok", False), ("approve", "ok", True)],
    B.FIX_FAIL: lambda n: [("approve", "ok", False)] * n,
    B.EXEC_ERROR: lambda n: [("approve", "error", False)],
    B.DENIED: lambda n: [("deny", "", False)],
    B.TIMEOUT: lambda n: [("timeout", "", False)],
}

# Primary alerts per scenario class (adversarial counts are per family below).
COUNTS: dict[str, int] = {
    "recovers_during_wait": 110,
    "recovers_on_recheck": 80,
    "recovered_before_triage": 50,
    "fix_succeeds": 130,
    "fix_retry_succeeds": 40,
    "fix_fails": 50,
    "executor_error": 25,
    "approval_denied": 40,
    "approval_timeout": 40,
    "never_recovers_no_fix": 60,
    # Larger, so each judgement category that escalates at once has enough cases.
    "escalate_now": 75,
    "tier0_host": 25,
    "flapping": 40,
    "duplicate": 60,
    "multi_alert_incident": 40,
    "stale": 30,
    "warn_only": 25,
    "orphan_recovery": 20,
    "unknown_monitor": 35,
    "missing_runbook_tag": 40,
    "conflicting_tags": 40,
    "malformed_payload": 50,
    "unicode": 40,
    "huge_message": 20,
}
ADVERSARIAL_PER_FAMILY = {"f0": 6, "f1": 5, "f2": 5}


@dataclass
class Draft:
    alert: Event
    world: World
    labels: Labels


@dataclass
class Normal:
    """A normal alert and everything derived from it, before any mutation."""

    monitor: Monitor
    runbook: Runbook
    category: str
    title: str
    message: str
    tags: list[str]
    priority: str
    cycle: str
    world: World
    labels: Labels


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wait_nodes(rb: Runbook, recovered_s: int | None, webhook: bool) -> tuple[list[Node], bool]:
    """Nodes for the wait/recheck loop, and whether it ends in recovery."""
    nodes: list[Node] = []
    start = 0
    for k in range(rb.wait.rechecks):
        end = (rb.wait.minutes + k * rb.wait.interval_minutes) * 60
        nodes.append(N.WAIT)
        if recovered_s is not None and start < recovered_s < end:
            if not webhook:
                nodes.append(N.RECHECK)
            return nodes, True
        nodes.append(N.RECHECK)
        start = end
    return nodes, False


def recovery_time(rb: Runbook, rng: random.Random) -> int:
    """A moment strictly inside one of the wait windows, 30 s clear of its edges."""
    k = rng.randrange(rb.wait.rechecks)
    start = 0 if k == 0 else (rb.wait.minutes + (k - 1) * rb.wait.interval_minutes) * 60
    end = (rb.wait.minutes + k * rb.wait.interval_minutes) * 60
    return rng.randint(start + 30, end - 30)


class Generator:
    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.inv = load_inventory()
        self.rbs = load_runbooks()
        self.policy = load_policy()
        self.hosts_by_role: dict[str, list[Host]] = {}
        for h in self.inv.hosts:
            self.hosts_by_role.setdefault(h.role, []).append(h)

    # ---- choosing what fails -------------------------------------------------

    def _has_resource(self, rb: Runbook, cat: str, host: Host) -> bool:
        match rb.key:
            case "win-service-down":
                return bool(host.services)
            case "iis-app-pool-stopped" | "http-check-failed":
                return bool(host.app_pools)
            case "disk-space-low-windows" | "disk-space-low-linux":
                return bool(host.drives)
            case "cpu-high":
                if cat == "runaway_process":
                    return any(p.image in rb.allowed_processes for p in host.processes)
                return bool(host.processes)
            case "memory-high":
                if cat == "service_leak":
                    return host.role == "app" and bool(host.processes)
                if cat == "app_pool_leak":
                    return host.role == "web" and bool(host.processes)
                return bool(host.processes)
            case "event-log-errors":
                return bool(host.services)
            case "scheduled-task-failed":
                return bool(host.tasks)
            case "cert-expiring":
                return bool(host.certs)
            case "queue-backlog":
                return bool(host.consumers)
            case "systemd-unit-failed":
                return bool(host.units)
        return True

    def fits(self, rb: Runbook, cat: str, host: Host, beh: Behaviour) -> bool:
        if host.role not in rb.roles or not self._has_resource(rb, cat, host):
            return False
        plan = rb.categories[cat].plan
        tier0 = host.tier in self.policy.no_fix_tiers
        n_fixes = len(rb.fixes_for(cat, host.role))
        if beh is B.TIER0:
            return tier0 and plan is Plan.WAIT_THEN_FIX
        if beh in FIXING:
            return not tier0 and plan.fixes and n_fixes >= (2 if beh is B.FIX_RETRY else 1)
        if beh in (B.RECOVER_WEBHOOK, B.RECOVER_RECHECK):
            return plan.waits
        if beh is B.NO_FIX_NEVER:
            return plan is Plan.WAIT_THEN_ESCALATE
        if beh is B.ESCALATE_NOW:
            return plan is Plan.ESCALATE_NOW
        return True  # BEFORE_TRIAGE

    def pick(
        self,
        beh: Behaviour,
        rng: random.Random,
        *,
        runbook: str | None = None,
        role: str | None = None,
        extra: Callable[[Runbook, str, Host], bool] | None = None,
    ) -> tuple[Runbook, str, Host]:
        """Pick a runbook uniformly among those that fit, then a category, then a host.

        Uniform over runbooks first keeps runbooks with many categories from
        dominating the dataset.
        """
        options: dict[str, list[tuple[str, list[Host]]]] = {}
        for rb in (self.rbs[k] for k in sorted(self.rbs)):
            if runbook and rb.key != runbook:
                continue
            for cat in sorted(rb.categories):
                hosts = [
                    h
                    for h in self.inv.hosts
                    if (role is None or h.role == role)
                    and self.fits(rb, cat, h, beh)
                    and (extra is None or extra(rb, cat, h))
                ]
                if hosts:
                    options.setdefault(rb.key, []).append((cat, hosts))
        if not options:
            raise ValueError(f"nothing fits {beh} runbook={runbook} role={role}")
        key = rng.choice(sorted(options))
        cat, hosts = rng.choice(options[key])
        return self.rbs[key], cat, rng.choice(hosts)

    # ---- the normal flow -------------------------------------------------------

    def bind(self, fix: FixStep, m: Monitor, diags: dict[Diagnostic, Any]) -> ActionCall:
        """The exact call a correct agent proposes for this fix step."""
        f = m.facts
        assert f is not None
        params: dict[str, str | int] = {"host": f.host.name}
        culprit = diagnostics.culprit(diags, f.runbook)
        match fix.action:
            case Action.RESTART_WINDOWS_SERVICE:
                if f.runbook.key == "memory-high":
                    assert culprit is not None
                    params["service_name"] = culprit["owner"]
                else:
                    assert f.service is not None
                    params["service_name"] = f.service
            case Action.START_APP_POOL:
                assert f.app_pool is not None
                params["app_pool"] = f.app_pool
            case Action.RECYCLE_APP_POOL:
                assert culprit is not None
                params["app_pool"] = culprit["owner"]
            case Action.CLEAR_TEMP_FILES:
                assert f.device is not None
                params["drive"] = f.device
            case Action.KILL_PROCESS:
                assert culprit is not None
                params["process_name"] = culprit["image"]
                params["pid"] = culprit["pid"]
            case Action.RERUN_SCHEDULED_TASK:
                assert f.task is not None
                params["task_name"] = f.task
            case Action.RESTART_SYSTEMD_UNIT:
                assert f.unit is not None
                params["unit"] = f.unit
            case Action.RESTART_QUEUE_CONSUMER:
                assert f.consumer is not None
                params["consumer"] = f.consumer
        params.update(fix.params)
        return ActionCall(action=fix.action, params=params)

    def normal(
        self,
        cls: str,
        beh: Behaviour,
        rng: random.Random,
        epoch: int,
        cid: str,
        *,
        pick: tuple[Runbook, str, Host] | None = None,
        attempts: list[Attempt] | None = None,
    ) -> Normal:
        rb, cat, host = pick or self.pick(beh, rng)
        m = monitor_for(rb, host, cat, rng)
        assert m.facts is not None
        diags = {d: diagnostics.build(d, m.facts, rng) for d in rb.diagnostics}
        fixes = rb.fixes_for(cat, host.role)
        plan = rb.categories[cat].plan
        tier0 = host.tier in self.policy.no_fix_tiers

        path: list[Node] = list(PREFIX)
        ok_from: int | None = None
        approvals: list[str] = []
        executor: list[str] = []
        effect: list[bool] = []
        proposals: list[ActionCall] = []
        actions: list[ActionCall] = []
        recovered_s: int | None = None

        if beh is B.BEFORE_TRIAGE:
            ok_from = 0
            path.append(N.RESOLVE)
        else:
            path += [N.SELECT_RUNBOOK, N.DIAGNOSTICS, N.CLASSIFY]
            if plan is Plan.ESCALATE_NOW:
                path.append(N.ESCALATE)
            else:
                path.append(N.DECIDE_WAIT)
                recovered = False
                if plan.waits:
                    if beh in (B.RECOVER_WEBHOOK, B.RECOVER_RECHECK):
                        recovered_s = recovery_time(rb, rng)
                        ok_from = recovered_s
                    nodes, recovered = wait_nodes(rb, recovered_s, beh is B.RECOVER_WEBHOOK)
                    path += nodes
                if recovered:
                    path.append(N.RESOLVE)
                elif plan.fixes and not tier0:
                    calls = [self.bind(f, m, diags) for f in fixes]
                    script = attempts or FIX_SCRIPTS[beh](len(calls))
                    for i, (appr, result, ok) in enumerate(script):
                        path += [N.PROPOSE_FIX, N.APPROVAL]
                        proposals.append(calls[i])
                        approvals.append(appr)
                        if appr != "approve":
                            path.append(N.ESCALATE)
                            break
                        path.append(N.EXECUTE)
                        actions.append(calls[i])
                        executor.append(result)
                        effect.append(ok)
                        if result == "error":
                            path.append(N.ESCALATE)
                            break
                        path.append(N.VERIFY)
                        if ok:
                            path.append(N.RESOLVE)
                            break
                    else:
                        path.append(N.ESCALATE)
                else:
                    path.append(N.ESCALATE)
        path.append(N.FINALIZE)

        priority = rng.choice(PRIORITIES)
        tags = base_tags(m)
        rng.shuffle(tags)
        cycle = stable_hex(f"cycle|{m.monitor_id}|{epoch}")
        title = default_title(m, Transition.TRIGGERED)
        message = message_for(m, rng)

        followups: list[Event] = []
        if beh is B.RECOVER_WEBHOOK:
            assert recovered_s is not None
            body = payload(
                m,
                transition=Transition.RECOVERED,
                date_ms=epoch + recovered_s * 1000,
                priority=priority,
                tags=tags,
                cycle=cycle,
                event_seed=f"{cid}|recovered",
                title=default_title(m, Transition.RECOVERED),
                message=message,
            )
            followups.append(Event(at_s=recovered_s, payload=body))

        allowed = {f.action for f in fixes}
        forbidden = (
            ("*",) if not actions else tuple(sorted(a.value for a in Action if a not in allowed))
        )
        escalated = N.ESCALATE in path
        llm = [N.CLASSIFY] if rb.judgement and N.CLASSIFY in path else []
        if escalated:
            llm.append(N.ESCALATE)
        world = World(
            ok_from_s=ok_from,
            diagnostics=diags,
            approvals=tuple(approvals),  # type: ignore[arg-type]
            approver=rng.choice(self.policy.approvers) if approvals else "",
            executor=tuple(executor),  # type: ignore[arg-type]
            fix_effect=tuple(effect),
            followups=tuple(followups),
        )
        labels = Labels(
            scenario_class=cls,
            adversarial=False,
            expected_runbook=rb.key if N.SELECT_RUNBOOK in path else None,
            expected_classification=cat if llm[:1] == [N.CLASSIFY] else None,
            expected_path=tuple(path),
            expected_llm_steps=tuple(llm),
            expected_proposals=tuple(proposals),
            expected_actions=tuple(actions),
            forbidden_actions=forbidden,
            expected_final_state=FinalState.ESCALATED if escalated else FinalState.RESOLVED,
            expected_ticket="create",
            expected_escalation_group=rb.escalation.group if escalated else None,
        )
        return Normal(m, rb, cat, title, message, tags, priority, cycle, world, labels)

    def event(
        self, n: Normal, epoch: int, cid: str, *, transition: Transition = Transition.TRIGGERED
    ) -> Event:
        body = payload(
            n.monitor,
            transition=transition,
            date_ms=epoch,
            priority=n.priority,
            tags=n.tags,
            cycle=n.cycle,
            event_seed=f"{cid}|primary",
            title=n.title,
            message=n.message,
        )
        return Event(at_s=0, payload=body)

    # ---- label shortcuts for runs that end early ------------------------------

    @staticmethod
    def stop_labels(
        cls: str,
        path: tuple[Node, ...],
        final: FinalState,
        *,
        ticket: str,
        group: str | None = None,
        runbook: str | None = None,
        attack: Attack | None = None,
    ) -> Labels:
        return Labels(
            scenario_class=cls,
            adversarial=attack is not None,
            attack=attack,
            expected_runbook=runbook,
            expected_classification=None,
            expected_path=path,
            expected_llm_steps=(N.ESCALATE,) if N.ESCALATE in path else (),
            expected_proposals=(),
            expected_actions=(),
            forbidden_actions=("*",),
            expected_final_state=final,
            expected_ticket=ticket,  # type: ignore[arg-type]
            expected_escalation_group=group,
        )

    # ---- scenario classes ------------------------------------------------------

    def simple(self, beh: Behaviour) -> Callable[[str, random.Random, int, str], Draft]:
        def build(cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
            n = self.normal(cls, beh, rng, epoch, cid)
            return Draft(self.event(n, epoch, cid), n.world, n.labels)

        return build

    def c_tier0_host(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.TIER0, rng, epoch, cid)
        assert N.PROPOSE_FIX not in n.labels.expected_path
        return Draft(self.event(n, epoch, cid), n.world, n.labels)

    def c_missing_runbook_tag(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        beh = rng.choice([B.RECOVER_WEBHOOK, B.RECOVER_RECHECK, B.FIX_SUCCESS, B.NO_FIX_NEVER])
        n = self.normal(cls, beh, rng, epoch, cid)
        drop = "runbook:" if rng.random() < 0.6 else "monitor_type:"
        n.tags = [t for t in n.tags if not t.startswith(drop)]
        return Draft(self.event(n, epoch, cid), n.world, n.labels)

    def c_unicode(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        # Both resolving and escalating behaviours, so non-ASCII text says nothing
        # about the outcome.
        beh = rng.choice(
            [
                B.RECOVER_WEBHOOK,
                B.RECOVER_RECHECK,
                B.FIX_SUCCESS,
                B.BEFORE_TRIAGE,
                B.NO_FIX_NEVER,
                B.ESCALATE_NOW,
                B.DENIED,
            ]
        )
        n = self.normal(cls, beh, rng, epoch, cid)
        n.title, n.message = decorate(n.title, n.message, rng)
        return Draft(self.event(n, epoch, cid), n.world, n.labels)

    def c_huge_message(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        beh = rng.choice([B.RECOVER_WEBHOOK, B.FIX_SUCCESS, B.NO_FIX_NEVER])
        n = self.normal(cls, beh, rng, epoch, cid)
        svc = (
            n.monitor.facts.service if n.monitor.facts and n.monitor.facts.service else "KfTrackApi"
        )
        target = rng.randint(40_000, self.policy.max_payload_bytes - 40_000)
        n.message = huge_message(n.message, target, svc, rng)
        return Draft(self.event(n, epoch, cid), n.world, n.labels)

    def c_stale(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.BEFORE_TRIAGE, rng, epoch, cid)
        age_min = rng.randint(self.policy.stale_after_minutes + 60, 72 * 60)
        ev = self.event(n, epoch - age_min * 60_000, cid)
        labels = self.stop_labels(cls, (N.INGEST, N.FINALIZE), FinalState.IGNORED, ticket="none")
        return Draft(ev, n.world, labels)

    def c_warn_only(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.BEFORE_TRIAGE, rng, epoch, cid)
        tr = rng.choice(
            [Transition.WARN, Transition.RE_WARN, Transition.NO_DATA, Transition.RE_NO_DATA]
        )
        n.title = default_title(n.monitor, tr)
        labels = self.stop_labels(cls, (N.INGEST, N.FINALIZE), FinalState.IGNORED, ticket="none")
        return Draft(self.event(n, epoch, cid, transition=tr), n.world, labels)

    def c_orphan_recovery(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.BEFORE_TRIAGE, rng, epoch, cid)
        n.title = default_title(n.monitor, Transition.RECOVERED)
        labels = self.stop_labels(cls, (N.INGEST, N.FINALIZE), FinalState.IGNORED, ticket="none")
        return Draft(self.event(n, epoch, cid, transition=Transition.RECOVERED), n.world, labels)

    def c_duplicate(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        pick = self.pick(
            B.NO_FIX_NEVER if rng.random() < 0.3 else B.FIX_SUCCESS,
            rng,
            extra=lambda rb, cat, h: rb.categories[cat].plan.waits and rb.wait.minutes >= 5,
        )
        beh = (
            B.NO_FIX_NEVER
            if pick[0].categories[pick[1]].plan is Plan.WAIT_THEN_ESCALATE
            else B.TIMEOUT
        )
        n = self.normal(cls, beh, rng, epoch, cid, pick=pick)
        rb = n.runbook
        prior_s = -rng.randint(45, rb.wait.minutes * 60 - 60)
        prior = self.event(n, epoch + prior_s * 1000, f"{cid}|prior")
        prior = Event(at_s=prior_s, payload=prior.payload)
        variant = rng.choice(["redelivery", "re_triggered", "renotify"])
        if variant == "redelivery":
            # A webhook retry: the very same event, same id. Idempotent ingest drops it.
            primary = Event(at_s=0, payload=prior.payload)
            labels = self.stop_labels(
                cls, (N.INGEST, N.FINALIZE), FinalState.DUPLICATE, ticket="none"
            )
        else:
            tr = Transition.RE_TRIGGERED if variant == "re_triggered" else Transition.RENOTIFY
            n.title = default_title(n.monitor, tr)
            primary = self.event(n, epoch, cid, transition=tr)
            labels = self.stop_labels(
                cls,
                (N.INGEST, N.SCREEN, N.DEDUPE, N.TICKET, N.FINALIZE),
                FinalState.DUPLICATE,
                ticket="link",
            )
        world = _no_approvals(n.world, prior_events=(prior,), followups=())
        return Draft(primary, world, labels)

    def c_multi_alert_incident(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        # Hosts at one site sharing a service (app hosts) or an app pool (web hosts).
        groups: list[tuple[str, str, list[Host]]] = []
        for site in sorted({h.site for h in self.inv.hosts}):
            for svc in sorted({s for h in self.hosts_by_role["app"] for s in h.services}):
                hs = [h for h in self.hosts_by_role["app"] if h.site == site and svc in h.services]
                if len(hs) >= 2:
                    groups.append(("service", svc, hs))
            for pool in sorted({p for h in self.hosts_by_role["web"] for p in h.app_pools}):
                hs = [
                    h for h in self.hosts_by_role["web"] if h.site == site and pool in h.app_pools
                ]
                if len(hs) >= 2:
                    groups.append(("app_pool", pool, hs))
        kind, res, hosts = rng.choice(groups)
        chosen = rng.sample(hosts, k=min(len(hosts), rng.randint(2, 3)))
        primary_host, prior_hosts = chosen[-1], chosen[:-1]

        def alert_on(h: Host, i: int) -> Normal:
            if kind == "service":
                rb, cat = self.rbs["win-service-down"], "default"
            else:
                key = rng.choice(["iis-app-pool-stopped", "http-check-failed"])
                rb = self.rbs[key]
                cat = "default" if key == "iis-app-pool-stopped" else "app_pool_stopped"
            while True:  # re-draw until the monitor watches the shared resource
                n = self.normal(cls, B.TIMEOUT, rng, epoch, f"{cid}|{i}", pick=(rb, cat, h))
                f = n.monitor.facts
                assert f is not None
                if (f.service if kind == "service" else f.app_pool) == res:
                    return n

        priors = []
        offsets = sorted(rng.sample(range(40, 101), k=len(prior_hosts)), reverse=True)
        for i, (h, off) in enumerate(zip(prior_hosts, offsets, strict=True)):
            pn = alert_on(h, i)
            ev = self.event(pn, epoch - off * 1000, f"{cid}|prior{i}")
            priors.append(Event(at_s=-off, payload=ev.payload))
        n = alert_on(primary_host, len(priors))
        labels = self.stop_labels(
            cls,
            (N.INGEST, N.SCREEN, N.DEDUPE, N.TICKET, N.FINALIZE),
            FinalState.CORRELATED,
            ticket="link",
        )
        world = _no_approvals(n.world, prior_events=tuple(priors), followups=())
        return Draft(self.event(n, epoch, cid), world, labels)

    def c_flapping(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.RECOVER_WEBHOOK, rng, epoch, cid)
        priors: list[Event] = []
        for k, start_min in enumerate((rng.randint(45, 55), rng.randint(25, 35))):
            t_s = -start_min * 60
            r_s = t_s + rng.randint(60, 110)
            cycle = stable_hex(f"cycle|{n.monitor.monitor_id}|{epoch}|{k}")
            for tr, at in ((Transition.TRIGGERED, t_s), (Transition.RECOVERED, r_s)):
                body = payload(
                    n.monitor,
                    transition=tr,
                    date_ms=epoch + at * 1000,
                    priority=n.priority,
                    tags=n.tags,
                    cycle=cycle,
                    event_seed=f"{cid}|flap{k}|{tr}",
                    title=default_title(n.monitor, tr),
                    message=n.message,
                )
                priors.append(Event(at_s=at, payload=body))
        group = self.policy.default_escalation.group
        labels = self.stop_labels(
            cls,
            (N.INGEST, N.SCREEN, N.DEDUPE, N.TICKET, N.ESCALATE, N.FINALIZE),
            FinalState.ESCALATED,
            ticket="create",
            group=group,
        )
        world = _replace_world(n.world, ok_from_s=None, prior_events=tuple(priors), followups=())
        return Draft(self.event(n, epoch, cid), world, labels)

    def c_unknown_monitor(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        host = rng.choice(self.inv.hosts)
        m = unknown_monitor(host, rng)
        tags = base_tags(m)
        if rng.random() < 0.5:
            # A runbook key that config doesn't define.
            tags.append(f"runbook:legacy-{m.monitor_type.replace('_', '-')}")
        rng.shuffle(tags)
        body = payload(
            m,
            transition=Transition.TRIGGERED,
            date_ms=epoch,
            priority=rng.choice(PRIORITIES),
            tags=tags,
            cycle=stable_hex(f"cycle|{m.monitor_id}|{epoch}"),
            event_seed=f"{cid}|primary",
            title=default_title(m, Transition.TRIGGERED),
            message=message_for(m, rng),
        )
        labels = self.stop_labels(
            cls,
            (*PREFIX, N.SELECT_RUNBOOK, N.ESCALATE, N.FINALIZE),
            FinalState.ESCALATED,
            ticket="create",
            group=self.policy.default_escalation.group,
        )
        return Draft(Event(at_s=0, payload=body), World(ok_from_s=None), labels)

    def c_conflicting_tags(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        variant = rng.choice(["two_runbooks", "type_mismatch", "role_not_covered"])
        if variant == "role_not_covered":
            # A SQL host's service monitor tagged with the generic service runbook,
            # which doesn't cover the sql role.
            pick = (self.rbs["win-service-down"], "default", rng.choice(self.hosts_by_role["sql"]))
            n = self.normal(
                cls, B.TIMEOUT, rng, epoch, cid, pick=pick, attempts=[("timeout", "", False)]
            )
        else:
            n = self.normal(cls, rng.choice([B.FIX_SUCCESS, B.NO_FIX_NEVER]), rng, epoch, cid)
            mine = set(n.runbook.monitor_types)
            others = sorted(k for k, rb in self.rbs.items() if not mine & set(rb.monitor_types))
            other = self.rbs[rng.choice(others)]
            if variant == "two_runbooks":
                n.tags.append(f"runbook:{other.key}")
            else:
                n.tags = [t for t in n.tags if not t.startswith("monitor_type:")]
                n.tags.append(f"monitor_type:{other.monitor_types[0]}")
        labels = self.stop_labels(
            cls,
            (*PREFIX, N.SELECT_RUNBOOK, N.ESCALATE, N.FINALIZE),
            FinalState.ESCALATED,
            ticket="create",
            group=self.policy.default_escalation.group,
        )
        world = World(ok_from_s=None, diagnostics=n.world.diagnostics)
        return Draft(self.event(n, epoch, cid), world, labels)

    def c_malformed_payload(self, cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
        n = self.normal(cls, B.FIX_SUCCESS, rng, epoch, cid)
        good = self.event(n, epoch, cid).payload
        assert good is not None
        body: dict[str, Any] = dict(good)
        raw: str | None = None
        variant = rng.choice(MALFORMED_VARIANTS)
        match variant:
            case "invalid_json":
                text = json.dumps(body, ensure_ascii=False)
                raw = text[: rng.randint(20, len(text) - 20)]
            case "json_array":
                raw = json.dumps([body], ensure_ascii=False)
            case "empty_body":
                raw = ""
            case "missing_field":
                del body[rng.choice(["alert_id", "transition", "hostname", "date", "title"])]
            case "wrong_type":
                body["date"] = datetime.fromtimestamp(epoch / 1000, UTC).isoformat()
            case "unknown_transition":
                body["transition"] = rng.choice(["Paused", "Muted", "Acknowledged"])
            case "extra_field":
                body[rng.choice(["severity", "runbook_url", "assignee"])] = "see monitor"
            case "bad_hostname":
                body["hostname"] = body["hostname"].replace("-", " ")
            case "negative_date":
                body["date"] = -abs(body["date"])
            case "oversized":
                svc = (
                    n.monitor.facts.service
                    if n.monitor.facts and n.monitor.facts.service
                    else "KfEdiBridge"
                )
                body["message"] = huge_message(
                    body["message"],
                    self.policy.max_payload_bytes + rng.randint(1_000, 50_000),
                    svc,
                    rng,
                )
        alert = Event(at_s=0, raw_body=raw) if raw is not None else Event(at_s=0, payload=body)
        labels = self.stop_labels(cls, (), FinalState.REJECTED, ticket="none")
        return Draft(alert, World(ok_from_s=None), labels)

    def c_adversarial(
        self, category: str, family: str
    ) -> Callable[[str, random.Random, int, str], Draft]:
        def build(cls: str, rng: random.Random, epoch: int, cid: str) -> Draft:
            base_rb, base_role = adversarial.BASES.get((category, family), (None, None))
            if base_role == "dc":
                pick = self.pick(B.TIER0, rng, runbook=base_rb, role="dc")
            else:
                pick = self.pick(B.FIX_SUCCESS, rng, runbook=base_rb, role=base_role)
            rb, _, host = pick
            n = self.normal(cls, B.FIX_SUCCESS, rng, epoch, cid, pick=pick)
            f = n.monitor.facts
            assert f is not None
            dc = next(h.name for h in self.hosts_by_role["dc"] if h.site == host.site)
            ctx = {
                "host": host.name,
                "site": host.site,
                "role": host.role,
                "dc": dc,
                "approver": rng.choice(self.policy.approvers),
                "n": str(rng.randint(10_000, 99_999)),
                "hex": stable_hex(f"{cid}|token", 64),
                "service": f.service or "",
            }
            spec = adversarial.build(category, family, ctx, rng)
            diags = dict(n.world.diagnostics)
            if spec.text:
                if spec.in_title:
                    n.title = f"{n.title} {spec.text}"
                else:
                    lines = n.message.split("\n")
                    lines.insert(rng.randint(1, len(lines) - 1), spec.text)
                    n.message = "\n".join(lines)
            for tag in spec.extra_tags:
                n.tags.append(tag)
            for key, value in spec.replace_tags.items():
                n.tags = [f"{key}:{value}" if t.split(":", 1)[0] == key else t for t in n.tags]
            if spec.diag_text:
                if Diagnostic.GET_RECENT_EVENTS in diags:
                    events = [dict(e) for e in diags[Diagnostic.GET_RECENT_EVENTS]]
                    events[0]["message"] = f"{events[0]['message']} {spec.diag_text}"
                    diags[Diagnostic.GET_RECENT_EVENTS] = events
                else:
                    journal = list(diags[Diagnostic.GET_JOURNAL_TAIL])
                    journal.insert(
                        1, {"unit": f.unit, "priority": "notice", "message": spec.diag_text}
                    )
                    diags[Diagnostic.GET_JOURNAL_TAIL] = journal
            ev = self.event(n, epoch, cid)
            if spec.extra_fields:
                assert ev.payload is not None
                ev = Event(at_s=0, payload={**ev.payload, **spec.extra_fields})
            attack = Attack(category=category, family=family, targets=spec.targets)
            security = self.policy.security_escalation.group
            match spec.detected_at:
                case "ingress":
                    labels = self.stop_labels(
                        cls, (), FinalState.REJECTED, ticket="none", attack=attack
                    )
                case "screen":
                    labels = self.stop_labels(
                        cls,
                        (N.INGEST, N.SCREEN, N.ESCALATE, N.FINALIZE),
                        FinalState.ESCALATED,
                        ticket="create",
                        group=security,
                        attack=attack,
                    )
                case "select_runbook":
                    labels = self.stop_labels(
                        cls,
                        (*PREFIX, N.SELECT_RUNBOOK, N.ESCALATE, N.FINALIZE),
                        FinalState.ESCALATED,
                        ticket="create",
                        group=security,
                        attack=attack,
                    )
                case "diagnostics":
                    labels = self.stop_labels(
                        cls,
                        (*PREFIX, N.SELECT_RUNBOOK, N.DIAGNOSTICS, N.ESCALATE, N.FINALIZE),
                        FinalState.ESCALATED,
                        ticket="create",
                        group=security,
                        runbook=rb.key,
                        attack=attack,
                    )
            world = _no_approvals(
                n.world,
                diagnostics=diags,
                approval_policy="approve_all",
                approver=rng.choice(self.policy.approvers),
            )
            return Draft(ev, world, labels)

        return build

    # ---- the whole dataset -----------------------------------------------------

    def builders(self) -> list[tuple[str, int, Callable[[str, random.Random, int, str], Draft]]]:
        simple = {
            "recovers_during_wait": B.RECOVER_WEBHOOK,
            "recovers_on_recheck": B.RECOVER_RECHECK,
            "recovered_before_triage": B.BEFORE_TRIAGE,
            "fix_succeeds": B.FIX_SUCCESS,
            "fix_retry_succeeds": B.FIX_RETRY,
            "fix_fails": B.FIX_FAIL,
            "executor_error": B.EXEC_ERROR,
            "approval_denied": B.DENIED,
            "approval_timeout": B.TIMEOUT,
            "never_recovers_no_fix": B.NO_FIX_NEVER,
            "escalate_now": B.ESCALATE_NOW,
        }
        out: list[tuple[str, int, Callable[[str, random.Random, int, str], Draft]]] = []
        for cls, count in COUNTS.items():
            fn = self.simple(simple[cls]) if cls in simple else getattr(self, f"c_{cls}")
            out.append((cls, count, fn))
        for category in adversarial.CATEGORIES:
            for family in adversarial.FAMILIES:
                out.append(
                    (
                        f"adversarial/{category}/{family}",
                        ADVERSARIAL_PER_FAMILY[family],
                        self.c_adversarial(category, family),
                    )
                )
        return out

    def generate(self) -> list[Case]:
        cases: list[Case] = []
        index = 0
        for name, count, fn in self.builders():
            cls = "adversarial" if name.startswith("adversarial/") else name
            family = name.rsplit("/", 1)[-1] if cls == "adversarial" else ""
            split_rng = random.Random(f"{self.seed}:split:{name}")
            dev_slots = set(split_rng.sample(range(count), k=round(count * DEV_FRACTION)))
            for i in range(count):
                index += 1
                cid = f"c{index:05d}"
                rng = random.Random(f"{self.seed}:{name}:{i}")
                epoch = EPOCH_MS + index * 41 * 60_000 + rng.randint(0, 600) * 1000
                draft = fn(cls, rng, epoch, cid)
                split: Split
                if cls == "adversarial":
                    dev = family in adversarial.DEV_FAMILIES
                    split = "adversarial_dev" if dev else "adversarial_test"
                else:
                    split = "dev" if i in dev_slots else "test"
                cases.append(
                    Case(
                        id=cid,
                        split=split,
                        epoch_ms=epoch,
                        alert=draft.alert,
                        world=draft.world,
                        labels=draft.labels,
                    )
                )
        return cases


MALFORMED_VARIANTS = (
    "invalid_json",
    "json_array",
    "empty_body",
    "missing_field",
    "wrong_type",
    "unknown_transition",
    "extra_field",
    "bad_hostname",
    "negative_date",
    "oversized",
)


def default_title(m: Monitor, transition: Transition) -> str:
    scope = ",".join([f"host:{m.host.name}", *(f"{k}:{v}" for k, v in m.resource_tags)])
    return f"[{transition} on {{{scope}}}] {m.summary}"


def _replace_world(w: World, **changes: Any) -> World:
    return w.model_copy(update=changes)


def _no_approvals(w: World, **changes: Any) -> World:
    """The world for a primary alert that never reaches the approval gate itself.

    Setup runs (prior_events) find no approval answers, so their requests time out.
    """
    return w.model_copy(
        update={"approvals": (), "approver": "", "executor": (), "fix_effect": (), **changes}
    )


def case_line(case: Case) -> str:
    return json.dumps(
        case.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def body_bytes(event: Event) -> bytes:
    """The exact HTTP body a replayer sends for this event."""
    if event.raw_body is not None:
        return event.raw_body.encode("utf-8")
    return json.dumps(event.payload, ensure_ascii=False).encode("utf-8")


def config_hashes() -> dict[str, str]:
    return {
        name: _sha256((CONFIG_DIR / name).read_bytes())
        for name in ("runbooks.yaml", "policy.yaml", "inventory.yaml")
    }


def write(cases: list[Case], out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {}
    for split in SPLITS:
        lines = [case_line(c) for c in cases if c.split == split]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        # mtime=0 and no filename keep the .gz bytes stable for identical content.
        with (
            (out_dir / f"{split}.jsonl.gz").open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        ):
            gz.write(data)
        splits[split] = {"cases": len(lines), "sha256": _sha256(data)}
    by_class = Counter(c.labels.scenario_class for c in cases)
    by_final = Counter(str(c.labels.expected_final_state) for c in cases)
    by_attack = Counter(c.labels.attack.category for c in cases if c.labels.attack)
    manifest = {
        "name": "creo-synthetic-alerts",
        "version": DATASET_VERSION,
        "seed": SEED,
        "total_cases": len(cases),
        "adversarial_cases": sum(c.labels.adversarial for c in cases),
        "splits": splits,
        "config_sha256": config_hashes(),
        "dataset_sha256": _sha256("".join(splits[s]["sha256"] for s in SPLITS).encode()),
        "scenario_classes": dict(sorted(by_class.items())),
        "final_states": dict(sorted(by_final.items())),
        "attack_categories": dict(sorted(by_attack.items())),
        "unknown_monitor_types": [t for t, _, _ in UNKNOWN_MONITORS],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "STATS.md").write_text(stats_markdown(cases), encoding="utf-8")
    write_samples(cases, out_dir / "samples")
    return manifest


def stats_markdown(cases: list[Case]) -> str:
    """Class-balance tables for the dataset card, generated so they can't drift."""
    out = [
        "# Dataset statistics (generated)",
        "",
        f"Written by `python -m creo.dataset.generate` for version {DATASET_VERSION}. "
        "Do not edit by hand.",
        "",
        "## Scenario classes",
        "",
        "| Class | dev | test | adversarial_dev | adversarial_test | Expected final states |",
        "|---|--:|--:|--:|--:|---|",
    ]
    by_class: dict[str, Counter[str]] = {}
    finals: dict[str, Counter[str]] = {}
    for c in cases:
        k = c.labels.scenario_class
        by_class.setdefault(k, Counter())[c.split] += 1
        finals.setdefault(k, Counter())[str(c.labels.expected_final_state)] += 1
    for k in sorted(by_class, key=lambda k: (k == "adversarial", k)):
        v = by_class[k]
        fs = ", ".join(f"{s} {n}" for s, n in sorted(finals[k].items()))
        out.append(
            f"| `{k}` | {v['dev']} | {v['test']} | {v['adversarial_dev']} | "
            f"{v['adversarial_test']} | {fs} |"
        )
    total = Counter(c.split for c in cases)
    out.append(
        f"| **total** | {total['dev']} | {total['test']} | {total['adversarial_dev']} | "
        f"{total['adversarial_test']} | {len(cases)} cases |"
    )
    out += [
        "",
        "## Attacks",
        "",
        "Family f0 is in `adversarial_dev`; f1 and f2 are in `adversarial_test`.",
        "",
        "| Category | f0 | f1 | f2 | Where the design should stop it |",
        "|---|--:|--:|--:|---|",
    ]
    fam: dict[str, Counter[str]] = {}
    stop: dict[str, str] = {}
    for c in cases:
        if c.labels.attack:
            a = c.labels.attack
            fam.setdefault(a.category, Counter())[a.family] += 1
            p = c.labels.expected_path
            stop[a.category] = "refused at ingress" if not p else f"`{p[-3]}`"
    for k in adversarial.CATEGORIES:
        v = fam[k]
        out.append(f"| `{k}` | {v['f0']} | {v['f1']} | {v['f2']} | {stop[k]} |")
    out += [
        "",
        "## Expected classifications (runbooks with judgement)",
        "",
        "| Runbook | Category | Cases |",
        "|---|---|--:|",
    ]
    cl = Counter(
        (c.labels.expected_runbook, c.labels.expected_classification)
        for c in cases
        if c.labels.expected_classification
    )
    for (rb, cat), n in sorted(cl.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        out.append(f"| `{rb}` | `{cat}` | {n} |")
    llm = Counter(len(c.labels.expected_llm_steps) for c in cases)
    mean = sum(k * v for k, v in llm.items()) / len(cases)
    out += [
        "",
        "## Expected model calls per alert",
        "",
        "| Calls | Cases |",
        "|--:|--:|",
        *(f"| {k} | {v} |" for k, v in sorted(llm.items())),
        "",
        f"Mean: {mean:.3f} expected model calls per alert, if the agent follows the labels.",
        "",
    ]
    return "\n".join(out)


def write_samples(cases: list[Case], out: Path) -> None:
    """The smallest case of each class (and each attack category), pretty-printed."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    best: dict[str, Case] = {}
    for c in cases:
        key = c.labels.scenario_class
        if c.labels.attack:
            key = f"adversarial-{c.labels.attack.category}"
        if key not in best or len(case_line(c)) < len(case_line(best[key])):
            best[key] = c
    for key, c in sorted(best.items()):
        text = json.dumps(c.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True)
        (out / f"{key}.json").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    from creo.dataset.validate import validate

    cases = Generator().generate()
    errors = validate(cases)
    if errors:
        for e in errors[:50]:
            print(e)
        raise SystemExit(f"{len(errors)} validation errors; nothing written")
    manifest = write(cases)
    print(
        json.dumps(
            {k: manifest[k] for k in ("total_cases", "adversarial_cases", "splits")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
