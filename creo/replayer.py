"""The `replayer`: posts dataset cases to the live agent, Datadog-style, on the shared clock.

    python -m creo.replayer --split dev --n 50 --rate 2 --auto-approve

For each case it prefixes monitor and event ids (so two replayed cases never
share a monitor), shifts every timestamp to replay time, registers the case's
world with the sim service, then posts setup alerts, the alert and any
follow-ups at their scripted times. With --auto-approve it also plays the
approver, answering each request as the case's script says (a "timeout"
answer means it never answers). At the end it compares each run with its
labels and writes a report.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from creo.dataset.generate import body_bytes
from creo.dataset.io import load
from creo.dataset.schema import Case, Event
from creo.settings import Settings, clock


def _prefix(i: int, value: str) -> str:
    return f"9{i:04d}{value}"[:20]


def localise(case: Case, i: int, shift_ms: int) -> Case:
    """The case as replay i: prefixed ids, timestamps moved by shift_ms."""

    def fix(ev: Event) -> Event:
        if ev.payload is None:
            return ev
        p = dict(ev.payload)
        for key in ("id", "alert_id"):
            if isinstance(p.get(key), str) and p[key].isdigit():
                p[key] = _prefix(i, p[key])
        for key in ("date", "last_updated"):
            if isinstance(p.get(key), int):
                p[key] = p[key] + shift_ms
        return Event(at_s=ev.at_s, payload=p, raw_body=ev.raw_body)

    w = case.world.model_copy(
        update={
            "prior_events": tuple(fix(e) for e in case.world.prior_events),
            "followups": tuple(fix(e) for e in case.world.followups),
        }
    )
    return case.model_copy(
        update={"alert": fix(case.alert), "world": w, "epoch_ms": case.epoch_ms + shift_ms}
    )


def pick(cases: list[Case], n: int, seed: int) -> list[Case]:
    """n cases, spread over scenario classes, preferring distinct hosts."""
    rng = random.Random(seed)
    by_class: dict[str, list[Case]] = {}
    for c in cases:
        by_class.setdefault(c.labels.scenario_class, []).append(c)
    for v in by_class.values():
        rng.shuffle(v)
    out: list[Case] = []
    hosts: set[str] = set()
    while len(out) < n and any(by_class.values()):
        for cls in sorted(by_class):
            if not by_class[cls] or len(out) >= n:
                continue
            fresh = [
                c for c in by_class[cls] if (c.alert.payload or {}).get("hostname") not in hosts
            ]
            c = (fresh or by_class[cls])[0]
            by_class[cls].remove(c)
            out.append(c)
            hosts.add(str((c.alert.payload or {}).get("hostname")))
    return out


class Approver(threading.Thread):
    """Answers approval requests for replayed runs, as each case's script says."""

    def __init__(self, agent: httpx.Client, s: Settings) -> None:
        super().__init__(daemon=True)
        self.agent, self.s = agent, s
        self.scripts: dict[str, tuple[Case, list[str]]] = {}
        self.seen: set[str] = set()
        self.stop = threading.Event()

    def watch(self, run_id: str, case: Case) -> None:
        self.scripts[run_id] = (case, list(case.world.approvals))

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                for req in self.agent.get("/api/approvals").json():
                    aid, run_id = req["approval_id"], req["run_id"]
                    if aid in self.seen or run_id not in self.scripts:
                        continue
                    self.seen.add(aid)
                    case, answers = self.scripts[run_id]
                    w = case.world
                    answer = (
                        "approve"
                        if w.approval_policy == "approve_all"
                        else (answers.pop(0) if answers else "timeout")
                    )
                    if answer == "timeout":
                        continue
                    self.agent.post(
                        f"/api/approvals/{aid}",
                        json={"decision": answer, "approver": w.approver or "a.mehta"},
                        headers={"X-Creo-Approver-Token": self.s.approver_token},
                    )
            except httpx.HTTPError:
                pass
            self.stop.wait(0.3)


def compare(case: Case, run: dict[str, Any] | None, rejected: str | None) -> dict[str, Any]:
    lab = case.labels
    if run is None:
        got_final, got_path = ("rejected", []) if rejected else (None, [])
    else:
        got_final, got_path = run.get("final_state"), run.get("path", [])
    return {
        "case": case.id,
        "class": lab.scenario_class,
        "run_id": run["run_id"] if run else None,
        "final_state": got_final,
        "expected_final_state": str(lab.expected_final_state),
        "final_ok": got_final == str(lab.expected_final_state),
        "path_ok": got_path == [str(n) for n in lab.expected_path],
        "executed": [
            a["action"] for a in (run or {}).get("actions", []) if a.get("result") == "ok"
        ],
        "expected_actions": [a.action.value for a in lab.expected_actions],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--split", default="dev", choices=["dev", "test", "adversarial_dev", "adversarial_test"]
    )
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--rate", type=float, default=2.0, help="primary alerts per wall-clock second")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument(
        "--timeout", type=float, default=600, help="wall seconds to wait for runs to finish"
    )
    ap.add_argument("--report", type=Path, default=Path("var/replay-report.json"))
    args = ap.parse_args()

    s = Settings()
    clk = clock(s)
    agent = httpx.Client(base_url=s.agent_url, timeout=30.0)
    sim = httpx.Client(base_url=s.sim_url, timeout=30.0)
    cases = pick(load(args.split, purpose="eval"), args.n, args.seed)

    # Every primary alert gets a slot; setup alerts go before it on the scaled clock.
    lead_s = max([-e.at_s for c in cases for e in c.world.prior_events] + [0]) / s.time_scale + 2
    wall0 = time.time() + lead_s
    plan: list[tuple[float, int, str, Case, Event]] = []
    replayed: list[Case] = []
    for i, c in enumerate(cases):
        wall = wall0 + i / args.rate
        shift = int(clk.epoch_ms + (wall - clk.wall_epoch) * 1000 * s.time_scale) - c.epoch_ms
        lc = localise(c, i, shift)
        replayed.append(lc)
        sim.post(
            "/cases",
            json={
                "case": lc.model_dump(mode="json"),
                "monitor_id": (lc.alert.payload or {}).get("alert_id", ""),
            },
        ).raise_for_status()
        for e in lc.world.prior_events:
            plan.append((wall + e.at_s / s.time_scale, 0, "prior", lc, e))
        plan.append((wall, 1, "primary", lc, lc.alert))
        for e in lc.world.followups:
            plan.append((wall + e.at_s / s.time_scale, 2, "followup", lc, e))
    plan.sort(key=lambda p: (p[0], p[1]))

    approver = Approver(agent, s)
    if args.auto_approve:
        approver.start()
    primary_run: dict[str, str | None] = {}
    rejected: dict[str, str] = {}
    headers = {"X-Creo-Secret": s.webhook_secret, "Content-Type": "application/json"}
    print(f"replaying {len(cases)} cases from {args.split} (time x{s.time_scale:g})", flush=True)
    for at, _, kind, lc, ev in plan:
        time.sleep(max(0.0, at - time.time()))
        r = agent.post("/webhook", content=body_bytes(ev), headers=headers)
        if kind == "primary":
            data = r.json()
            primary_run[lc.id] = data.get("run_id")
            if r.status_code >= 400:
                rejected[lc.id] = data.get("reason", str(r.status_code))
            elif data.get("run_id"):
                approver.watch(data["run_id"], lc)
            print(f"{lc.id} {lc.labels.scenario_class:<24} -> {r.status_code} {data}", flush=True)

    deadline = time.time() + args.timeout
    results: dict[str, dict[str, Any] | None] = {}
    while time.time() < deadline:
        for lc in replayed:
            rid = primary_run.get(lc.id)
            if rid and lc.id not in results:
                run = agent.get(f"/api/runs/{rid}").json()
                if run.get("final_state"):
                    results[lc.id] = run
        if all(lc.id in results or not primary_run.get(lc.id) for lc in replayed):
            break
        time.sleep(1)
    approver.stop.set()

    rows = [compare(lc, results.get(lc.id), rejected.get(lc.id)) for lc in replayed]
    unfinished = [r["case"] for r in rows if r["final_state"] is None]
    summary = {
        "cases": len(rows),
        "finished": len(rows) - len(unfinished),
        "final_state_matches": sum(r["final_ok"] for r in rows),
        "path_matches": sum(r["path_ok"] for r in rows),
        "unfinished": unfinished,
        "llm": agent.get("/healthz").json().get("llm"),
        "time_scale": s.time_scale,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
