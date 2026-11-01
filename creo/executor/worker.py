"""The `executor` service: polls the agent for jobs, runs them, reports back.

Outbound only: it opens no port. It honours two kill switches, the agent's
(reported with each poll) and a local one (the file CREO_KILLSWITCH_FILE),
so a person on the jump host can stop changes even if the agent is
compromised or unreachable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from creo.clients import HttpHosts
from creo.config import load_inventory, load_policy
from creo.executor.core import Executor
from creo.settings import Settings, clock


def main() -> None:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    local_switch = Path(os.environ.get("CREO_KILLSWITCH_FILE", str(s.data_dir / "KILLSWITCH")))
    remote_switch = {"on": False}

    def killswitch() -> bool:
        return remote_switch["on"] or local_switch.exists()

    ex = Executor(
        inventory=load_inventory(),
        policy=load_policy(),
        approval_key=s.approval_key,
        backend=HttpHosts(s.sim_url),
        clock=clock(s),
        db=s.data_dir / "executor.db",
        killswitch=killswitch,
    )
    agent = httpx.Client(
        base_url=s.agent_url, timeout=10.0, headers={"X-Creo-Executor-Token": s.executor_token}
    )
    print(f"executor polling {s.agent_url}", flush=True)
    while True:
        try:
            r = agent.post("/api/jobs/claim")
            r.raise_for_status()
            data = r.json()
            remote_switch["on"] = bool(data.get("killswitch"))
            for job in data["jobs"]:
                result = ex.run(job)
                agent.post(f"/api/jobs/{job['job_id']}/result", json=result).raise_for_status()
                print(f"{job['job_id']}: {result['status']} {result.get('code', '')}", flush=True)
            if not data["jobs"]:
                time.sleep(0.5)
        except httpx.HTTPError as e:
            print(f"agent unreachable: {e!r}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
