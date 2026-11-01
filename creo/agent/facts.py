"""The redaction boundary: the only data that may reach the model.

The model never sees an alert's title, message, raw tags or the free text
that diagnostics bring back. It gets a handful of typed facts: numbers,
enums, and identifiers that match a strict pattern. A string that doesn't
match, or that the screen would flag, is replaced with "<redacted>".
"""

from __future__ import annotations

import re
from typing import Any

from creo.agent.screen import screen_text
from creo.config import Host, Policy, Runbook

IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
# Event log and source names contain spaces and slashes ("Service Control Manager").
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./\-]{0,63}$")
REDACTED = "<redacted>"


def _ident(v: Any) -> str:
    return v if isinstance(v, str) and IDENT.match(v) else REDACTED


def _label(v: Any, policy: Policy) -> str:
    if not isinstance(v, str) or not LABEL.match(v) or screen_text(v, "label", policy):
        return REDACTED
    return v


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return v


def _series(v: Any) -> list[float] | None:
    if not isinstance(v, list) or not v or len(v) > 60:
        return None
    nums = [_num(x) for x in v]
    return [float(x) for x in nums if x is not None] if None not in nums else None


def _change(series: list[float]) -> float:
    """Change across the window on a least-squares line, so noise at either end counts less."""
    n = len(series)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, sum(series) / n
    slope = sum((i - mx) * (y - my) for i, y in enumerate(series)) / sum(
        (i - mx) ** 2 for i in range(n)
    )
    return slope * (n - 1)


def _trend(change: float, threshold: float) -> str:
    return "rising" if change > threshold else "falling" if change < -threshold else "steady"


# A trend counts when the fitted change exceeds 20 CPU points, or a quarter of the mean memory.
CPU_TREND_POINTS = 20.0
MEM_TREND_SHARE = 0.25


def _enum(v: Any, allowed: tuple[str, ...]) -> str:
    return v if v in allowed else "other"


def evidence(runbook: Runbook, diagnostics: dict[str, Any], policy: Policy) -> list[dict[str, Any]]:
    """Typed evidence records from the runbook's diagnostics, each with a stable id."""
    out: list[dict[str, Any]] = []
    for name in runbook.diagnostics:
        data = diagnostics.get(name.value)
        if name.value == "get_top_processes" and isinstance(data, dict):
            out.append(
                {
                    "id": "host",
                    "kind": "host_totals",
                    "total_mem_mb": _num(data.get("total_mem_mb")),
                    "logical_cpus": _num(data.get("logical_cpus")),
                }
            )
            for i, p in enumerate(data.get("processes", [])):
                cpu, mem = _series(p.get("cpu_pct_series_5m")), _series(p.get("mem_mb_series_30m"))
                if cpu is None or mem is None or _num(p.get("pid")) is None:
                    continue
                mem_mean = sum(mem) / len(mem)
                out.append(
                    {
                        "id": f"proc{i + 1}",
                        "kind": "process",
                        "image": _ident(p.get("image")),
                        "owner": _ident(p.get("owner")),
                        "pid": p["pid"],
                        "cpu_pct_last_5m": cpu,
                        "cpu_pct_avg": round(sum(cpu) / len(cpu), 1),
                        "cpu_trend": _trend(_change(cpu), CPU_TREND_POINTS),
                        "mem_mb_last_30m": mem,
                        "mem_mb_change_30m": round(_change(mem)),
                        "mem_trend": _trend(_change(mem), MEM_TREND_SHARE * mem_mean),
                    }
                )
        elif name.value == "get_recent_events" and isinstance(data, list):
            for i, e in enumerate(data):
                out.append(
                    {
                        "id": f"evt{i + 1}",
                        "kind": "event",
                        "log": _label(e.get("log"), policy),
                        "source": _label(e.get("source"), policy),
                        "event_id": _num(e.get("event_id")),
                        "level": _enum(e.get("level"), ("Error", "Warning", "Information")),
                        "count": _num(e.get("count")),
                    }
                )
        elif name.value == "http_probe" and isinstance(data, dict):
            results = data.get("recent_results")
            out.append(
                {
                    "id": "probe",
                    "kind": "http_probe",
                    "status_code": _num(data.get("status_code")),
                    "latency_ms": _num(data.get("latency_ms")),
                    "recent_results": [_enum(r, ("ok", "fail")) for r in results][:10]
                    if isinstance(results, list)
                    else [],
                }
            )
        elif name.value == "get_app_pool_state" and isinstance(data, list):
            for i, p in enumerate(data):
                out.append(
                    {
                        "id": f"pool{i + 1}",
                        "kind": "app_pool",
                        "app_pool": _ident(p.get("app_pool")),
                        "state": _enum(
                            p.get("state"), ("Started", "Stopped", "Starting", "Stopping")
                        ),
                        "worker_processes": _num(p.get("worker_processes")),
                        "rapid_fail_count_1h": _num(p.get("rapid_fail_count_1h")),
                    }
                )
        elif name.value == "get_service_status" and isinstance(data, list):
            for i, s in enumerate(data):
                out.append(
                    {
                        "id": f"svc{i + 1}",
                        "kind": "service",
                        "service": _ident(s.get("service")),
                        "state": _enum(
                            s.get("state"), ("Running", "Stopped", "Paused", "Starting")
                        ),
                        "last_exit_code": _num(s.get("last_exit_code")),
                        "restarts_24h": _num(s.get("restarts_24h")),
                    }
                )
    return out


def classify_facts(
    runbook: Runbook, host: Host, diagnostics: dict[str, Any], policy: Policy
) -> dict[str, Any]:
    return {
        "task": "classify",
        "runbook": runbook.key,
        "runbook_description": runbook.description,
        "host_role": host.role,
        "categories": sorted(runbook.categories),
        "category_meanings": {k: c.means for k, c in sorted(runbook.categories.items())},
        "evidence": evidence(runbook, diagnostics, policy),
    }


def summary_facts(
    *,
    host: str,
    role: str | None,
    tier: int | None,
    runbook: str | None,
    reason: str,
    flags: list[str],
    category: str | None,
    steps: list[str],
    actions: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "task": "summarize",
        "host": _ident(host),
        "host_role": role,
        "host_tier": tier,
        "runbook": runbook,
        "reason": reason,
        "flags": flags,
        "category": category,
        "steps_taken": steps,
        "actions": actions,
    }
