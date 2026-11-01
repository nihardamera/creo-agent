"""A hand-written rules classifier, scored on the same requests the model answered.

Honest caveat: these rules were written by the author of the data generator,
so on this synthetic data they are close to an upper bound. They show the
classify step *can* be done by rules on clean, typed evidence. They do not
show the model is unnecessary on messy real evidence. Reported side by side
with the model so a reader can judge what the model adds here.
"""

from __future__ import annotations

import json
from typing import Any

from creo.dataset.schema import Case
from creo.eval.metrics import bootstrap

HARDWARE_SOURCES = {"disk", "Ntfs", "Microsoft-Windows-WHEA-Logger", "volmgr"}
NOISE_SOURCES = {"Schannel", "Microsoft-Windows-DistributedCOM"}


def classify(facts: dict[str, Any]) -> str:
    rb, ev = facts["runbook"], facts["evidence"]
    procs = [e for e in ev if e["kind"] == "process"]
    if rb == "cpu-high":
        if any(min(p["cpu_pct_last_5m"]) > 90 for p in procs):
            return "runaway_process"
        if any(
            p["cpu_pct_last_5m"][0] >= 85
            and p["cpu_pct_last_5m"][-1] < p["cpu_pct_last_5m"][0] - 30
            for p in procs
        ):
            return "transient_spike"
        return "sustained_load"
    if rb == "memory-high":
        growing = [p for p in procs if p["mem_mb_last_30m"][-1] - p["mem_mb_last_30m"][0] > 1_500]
        if len(growing) == 1:
            return "app_pool_leak" if growing[0]["image"] == "w3wp.exe" else "service_leak"
        if any(
            p["mem_mb_last_30m"][0] > 5_000
            and p["mem_mb_last_30m"][-1] < p["mem_mb_last_30m"][0] - 2_000
            for p in procs
        ):
            return "transient"
        return "unknown"
    if rb == "http-check-failed":
        probe: dict[str, Any] = next((e for e in ev if e["kind"] == "http_probe"), {})
        if probe.get("status_code") == 503 or any(
            e.get("state") == "Stopped" for e in ev if e["kind"] == "app_pool"
        ):
            return "app_pool_stopped"
        if probe.get("status_code") == 200:
            return "transient"
        return "upstream_or_network"
    if rb == "event-log-errors":
        events = [e for e in ev if e["kind"] == "event"]
        if any(e["event_id"] in (7031, 7034, 1000) for e in events):
            return "service_crash"
        if any(e["source"] in HARDWARE_SOURCES for e in events):
            return "hardware_or_os"
        if events and all(e["source"] in NOISE_SOURCES for e in events):
            return "transient"
        return "unknown"
    return "default"


def compare(pairs: list[tuple[Case, dict[str, Any]]]) -> dict[str, Any]:
    """Model vs rules on every classify request of a primary run with a labelled category."""
    model, rules = [], []
    for case, out in pairs:
        want = case.labels.expected_classification
        if not want:
            continue
        for call in out["llm_calls"]:
            if call["node"] != "classify":
                continue
            facts = json.loads(call["request_json"])
            rules.append(float(classify(facts) == want))
            model.append(float(out.get("category") == want))
    return {
        "note": "rules by the data generator's author: near an upper bound on synthetic data",
        "model_accuracy": bootstrap(model).as_dict(),
        "rules_accuracy": bootstrap(rules).as_dict(),
    }
