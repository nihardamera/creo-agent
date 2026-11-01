"""Monitors and webhook payloads for the fictional fleet, in Datadog's shape."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Any

from creo.catalog import Transition
from creo.config import Host, Runbook
from creo.dataset.diagnostics import Facts
from creo.dataset.fleet import COMPANY, DOMAIN, HTTP_CHECKS

ORG_ID = "48213"
MONITORING = f"https://monitoring.{DOMAIN}"

TEAMS = {
    "KfRateCalc": "rating",
    "KfLabelSpooler": "labels",
    "KfManifestSync": "manifests",
    "KfDockSched": "docks",
    "KfEdiBridge": "edi",
    "KfTrackApi": "tracking",
    "KfBatchAgent": "batch",
    "TrackHubPool": "tracking",
    "CustomerPortalPool": "portal",
    "CarrierApiPool": "carriers",
    "AdminPool": "platform",
}
ROLE_TEAMS = {
    "web": "platform",
    "app": "platform",
    "batch": "batch",
    "sql": "data",
    "fs": "infra",
    "dc": "identity",
    "proxy": "edge",
    "queue": "messaging",
}
DNS_NAMES = tuple(f"{n}.svc.{DOMAIN}" for n in ("rates", "docks", "edi", "labels", "tracking"))


def stable_int(text: str, lo: int, hi: int) -> int:
    """A deterministic integer in [lo, hi] (Python's hash() is salted per process)."""
    digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return lo + digest % (hi - lo + 1)


def stable_hex(text: str, n: int = 32) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


@dataclass(frozen=True)
class Monitor:
    monitor_id: str
    monitor_type: str
    runbook_key: str | None
    summary: str
    metric: str
    query: str
    status: str
    resource_tags: tuple[tuple[str, str], ...]
    team: str
    host: Host
    # None for monitors no runbook covers.
    facts: Facts | None


def monitor_for(rb: Runbook, host: Host, category: str, rng: random.Random) -> Monitor:
    """Pick the failing resource on `host` and describe the monitor watching it."""
    h = host.name
    f = Facts(host=host, runbook=rb, category=category)
    res: tuple[tuple[str, str], ...] = ()
    team = ROLE_TEAMS[host.role]
    match rb.key:
        case "win-service-down":
            svc = rng.choice(host.services)
            f = replace(f, service=svc)
            res = (("service", svc),)
            team = TEAMS.get(svc, team)
            metric, summary = "windows_service.state", f"Windows service {svc} is not running"
            query = (
                f'"windows_service.state".over("host:{h}","windows_service:{svc.lower()}")'
                ".last(3).count_by_status()"
            )
            status = (
                f"windows_service.state over host:{h},windows_service:{svc.lower()} "
                "was CRITICAL on the last 3 checks."
            )
        case "iis-app-pool-stopped":
            pool = rng.choice(host.app_pools)
            f = replace(f, app_pool=pool)
            res = (("app_pool", pool),)
            team = TEAMS[pool]
            metric, summary = "iis.app_pool_up", f"IIS application pool {pool} is stopped"
            query = (
                f'"iis.app_pool_up".over("host:{h}","app_pool:{pool}").last(2).count_by_status()'
            )
            status = (
                f"iis.app_pool_up over host:{h},app_pool:{pool} was CRITICAL on the last 2 checks."
            )
        case "disk-space-low-windows" | "disk-space-low-linux":
            dev = rng.choice(host.drives)
            f = replace(f, device=dev)
            res = (("device", dev),)
            metric, summary = "system.disk.in_use", f"Disk {dev} is more than 90% full"
            query = f"avg(last_10m):avg:system.disk.in_use{{host:{h},device:{dev}}} > 0.9"
            status = (
                f"system.disk.in_use over host:{h},device:{dev} was > 0.9 on average "
                "during the last 10m."
            )
        case "cpu-high":
            metric, summary = "system.cpu.user", "CPU above 90% for 15 minutes"
            query = f"avg(last_15m):avg:system.cpu.user{{host:{h}}} > 90"
            status = f"system.cpu.user over host:{h} was > 90 on average during the last 15m."
        case "memory-high":
            metric, summary = "system.mem.pct_usable", "Usable memory below 10%"
            query = f"avg(last_10m):avg:system.mem.pct_usable{{host:{h}}} < 0.1"
            status = (
                f"system.mem.pct_usable over host:{h} was < 0.1 on average during the last 10m."
            )
        case "http-check-failed":
            pool = rng.choice(host.app_pools)
            check = next(c for c in host.http_checks if _check_pool(c) == pool)
            f = replace(f, app_pool=pool, check=check)
            res = (("check", check), ("app_pool", pool))
            team = TEAMS[pool]
            metric, summary = "http.can_connect", f"HTTP check {check} is failing"
            query = (
                f'"http.can_connect".over("instance:{check}","host:{h}").by("url")'
                ".last(3).count_by_status()"
            )
            status = (
                f"http.can_connect over instance:{check},host:{h} "
                "was CRITICAL on the last 3 checks."
            )
        case "event-log-errors":
            if category == "service_crash":
                f = replace(f, service=rng.choice(host.services))
            metric, summary = "", "Error burst in the Windows event logs"
            query = (
                f'logs("source:windows.events host:{h} status:error")'
                '.index("*").rollup("count").last("15m") > 50'
            )
            status = f"Log count for host:{h} status:error was > 50 during the last 15m."
        case "scheduled-task-failed":
            task = rng.choice(host.tasks)
            f = replace(f, task=task)
            res = (("task", task),)
            metric, summary = "", f"Scheduled task {task} failed"
            query = (
                f'logs("source:windows.events host:{h} @evt.id:103 @task:{task}")'
                '.index("*").rollup("count").last("30m") > 0'
            )
            status = f"Log count for host:{h} task:{task} was > 0 during the last 30m."
        case "cert-expiring":
            cert = rng.choice(host.certs)
            f = replace(f, cert=cert)
            res = (("cert", cert),)
            metric, summary = "tls.days_left", f"TLS certificate {cert} expires in under 14 days"
            query = f"min(last_1h):min:tls.days_left{{host:{h},server_name:{cert}}} < 14"
            status = (
                f"tls.days_left over host:{h},server_name:{cert} "
                "was < 14 at all times during the last 1h."
            )
        case "host-unreachable":
            metric, summary = "datadog.agent.up", "Host is not reporting"
            query = f'"datadog.agent.up".over("host:{h}").last(2).count_by_status()'
            status = f"datadog.agent.up over host:{h} was CRITICAL on the last 2 checks."
        case "queue-backlog":
            q = rng.choice(host.consumers)
            f = replace(f, consumer=q)
            res = (("consumer", q),)
            metric, summary = "rabbitmq.queue.messages", f"Queue {q} backlog above 5000 messages"
            query = (
                f"avg(last_10m):avg:rabbitmq.queue.messages{{host:{h},rabbitmq_queue:{q}}} > 5000"
            )
            status = (
                f"rabbitmq.queue.messages over host:{h},rabbitmq_queue:{q} "
                "was > 5000 on average during the last 10m."
            )
        case "systemd-unit-failed":
            unit = rng.choice(host.units)
            f = replace(f, unit=unit)
            res = (("unit", unit),)
            metric, summary = "systemd.unit.state", f"systemd unit {unit} has failed"
            query = f'"systemd.unit.state".over("host:{h}","unit:{unit}").last(2).count_by_status()'
            status = (
                f"systemd.unit.state over host:{h},unit:{unit} was CRITICAL on the last 2 checks."
            )
        case "sql-blocking":
            metric, summary = (
                "sqlserver.activity.blocked_connections",
                "SQL sessions blocked for over 5 minutes",
            )
            query = f"min(last_5m):max:sqlserver.activity.blocked_connections{{host:{h}}} > 0"
            status = (
                f"sqlserver.activity.blocked_connections over host:{h} "
                "was > 0 at all times during the last 5m."
            )
        case "dns-resolution-failed":
            name = rng.choice(DNS_NAMES)
            f = replace(f, dns_name=name)
            metric, summary = "dns.can_resolve", f"Cannot resolve {name}"
            query = (
                f'"dns.can_resolve".over("host:{h}","resolved_hostname:{name}")'
                ".last(3).count_by_status()"
            )
            status = (
                f"dns.can_resolve over host:{h},resolved_hostname:{name} "
                "was CRITICAL on the last 3 checks."
            )
        case _:
            raise ValueError(f"no monitor template for runbook {rb.key}")
    resource = ",".join(f"{k}:{v}" for k, v in res)
    return Monitor(
        monitor_id=str(stable_int(f"{rb.key}|{h}|{resource}", 10_000_000, 99_999_999)),
        monitor_type=rb.monitor_types[0],
        runbook_key=rb.key,
        summary=summary,
        metric=metric,
        query=query,
        status=status,
        resource_tags=res,
        team=team,
        host=host,
        facts=f,
    )


def _check_pool(check: str) -> str:
    return next(pool for pool, c in HTTP_CHECKS.items() if c == check)


# Monitor types no runbook covers (the unknown_monitor class).
UNKNOWN_MONITORS: tuple[tuple[str, str, str], ...] = (
    ("backup_job_failed", "veeam.job.result", "Nightly backup job did not complete"),
    ("ntp_drift", "ntp.offset", "Clock offset above 500 ms"),
    ("printer_offline", "snmp.printer.status", "Label printer queue is offline"),
    ("license_expiring", "kestrel.license.days_left", "Software licence expires in under 30 days"),
    ("replication_lag", "dfsr.backlog_files", "DFS replication backlog above 5000 files"),
    ("smart_disk_warning", "smart.reallocated_sectors", "Disk reports reallocated sectors"),
)


def unknown_monitor(host: Host, rng: random.Random) -> Monitor:
    mtype, metric, summary = rng.choice(UNKNOWN_MONITORS)
    h = host.name
    return Monitor(
        monitor_id=str(stable_int(f"unknown|{mtype}|{h}", 10_000_000, 99_999_999)),
        monitor_type=mtype,
        runbook_key=None,
        summary=summary,
        metric=metric,
        query=f"avg(last_15m):avg:{metric}{{host:{h}}} > 0",
        status=f"{metric} over host:{h} was > 0 on average during the last 15m.",
        resource_tags=(),
        team=ROLE_TEAMS[host.role],
        host=host,
        facts=None,
    )


def base_tags(m: Monitor, *, runbook_tag: bool = True, monitor_type_tag: bool = True) -> list[str]:
    host = m.host
    tags = [
        f"env:{host.env}",
        f"site:{host.site}",
        f"role:{host.role}",
        f"tier:{host.tier}",
        f"team:{m.team}",
        *(f"{k}:{v}" for k, v in m.resource_tags),
    ]
    if monitor_type_tag:
        tags.append(f"monitor_type:{m.monitor_type}")
    if runbook_tag and m.runbook_key:
        tags.append(f"runbook:{m.runbook_key}")
    return tags


ALERT_TYPE = {
    Transition.TRIGGERED: "error",
    Transition.RE_TRIGGERED: "error",
    Transition.RENOTIFY: "error",
    Transition.RECOVERED: "success",
    Transition.WARN: "warning",
    Transition.RE_WARN: "warning",
    Transition.NO_DATA: "info",
    Transition.RE_NO_DATA: "info",
}


def message_for(m: Monitor, rng: random.Random) -> str:
    """The monitor's human-written notification text ($TEXT_ONLY_MSG)."""
    slug = m.team
    lines = [
        f"{m.summary} on {m.host.name}.",
        "",
        rng.choice(
            [
                f"Owner: {m.team} team. Dashboard: {MONITORING}/dash/{slug}",
                f"Escalation contacts are on the {m.team} team page: {MONITORING}/teams/{slug}",
                f"See the {m.team} overview: {MONITORING}/dash/{slug}?live=true",
            ]
        ),
        "",
        m.status,
        "",
        "@webhook-creo-ingest",
    ]
    return "\n".join(lines)


def payload(
    m: Monitor,
    *,
    transition: Transition,
    date_ms: int,
    priority: str,
    tags: list[str],
    cycle: str,
    event_seed: str,
    title: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    host = m.host.name
    scope = ",".join([f"host:{host}", *(f"{k}:{v}" for k, v in m.resource_tags)])
    event_id = str(stable_int(f"event|{event_seed}", 1_000_000_000, 9_999_999_999))
    return {
        "id": event_id,
        "alert_id": m.monitor_id,
        "alert_cycle_key": cycle,
        "aggreg_key": stable_hex(f"aggreg|{m.monitor_id}"),
        "title": title if title is not None else f"[{transition} on {{{scope}}}] {m.summary}",
        "message": message if message is not None else "",
        "transition": str(transition),
        "alert_type": ALERT_TYPE[transition],
        "alert_priority": priority,
        "date": date_ms,
        "last_updated": date_ms,
        "hostname": host,
        "metric": m.metric,
        "query": m.query,
        "scope": scope,
        "status": m.status,
        "tags": ",".join(tags),
        "org_id": ORG_ID,
        "org_name": COMPANY,
        "link": f"{MONITORING}/event/jump_to?event_id={event_id}",
    }
