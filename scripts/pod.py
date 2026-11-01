"""Drive one RunPod GPU pod for P4 through the REST API (key from the Keychain).

    uv run python scripts/pod.py create      # RTX 4090, community cloud, CUDA 13.0 hosts
    uv run python scripts/pod.py status      # state, $/h, ssh command
    uv run python scripts/pod.py stop        # stops billing for the GPU; the volume is kept
    uv run python scripts/pod.py start
    uv run python scripts/pod.py terminate   # deletes the pod and its volume

The pod id is kept in tune/pod/pod.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx

API = "https://rest.runpod.io/v1"
STATE = Path(__file__).resolve().parents[1] / "tune" / "pod" / "pod.json"
KEY = Path.home() / ".ssh" / "runpod_ed25519"
IMAGE = "runpod/pytorch:1.5.1-cu1300-torch2130-ubuntu2404"  # torch 2.13.0, as vllm 0.32.0 pins


def _headers() -> dict[str, str]:
    key = subprocess.run(
        ["security", "find-generic-password", "-s", "runpod-api-key", "-w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"Authorization": f"Bearer {key}"}


def _call(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    r = httpx.request(method, f"{API}{path}", headers=_headers(), json=body, timeout=60)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path}: {r.status_code} {r.text[:500]}")
    return r.json() if r.content else {}


def _pod_id() -> str:
    return str(json.loads(STATE.read_text())["id"])


def create(args: argparse.Namespace) -> None:
    body = {
        "name": "creo-p4",
        "imageName": IMAGE,
        "cloudType": args.cloud,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuCount": 1,
        "allowedCudaVersions": ["13.0"],
        "containerDiskInGb": 40,
        "volumeInGb": 60,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp", "8000/http"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()},
        "interruptible": False,
    }
    pod = _call("POST", "/pods", body)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"id": pod["id"]}) + "\n")
    print(pod["id"], pod.get("costPerHr"))


def status(_: argparse.Namespace) -> None:
    p = _call("GET", f"/pods/{_pod_id()}")
    ssh = p.get("portMappings", {}).get("22")
    print(
        json.dumps(
            {
                k: p.get(k)
                for k in ("id", "desiredStatus", "costPerHr", "adjustedCostPerHr", "publicIp")
            }
        )
    )
    if ssh and p.get("publicIp"):
        print(f"ssh -i {KEY} -p {ssh} -o StrictHostKeyChecking=accept-new root@{p['publicIp']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["create", "status", "stop", "start", "terminate"])
    ap.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    args = ap.parse_args()
    if args.command == "create":
        create(args)
    elif args.command == "status":
        status(args)
    elif args.command == "terminate":
        _call("DELETE", f"/pods/{_pod_id()}")
        STATE.unlink()
        print("terminated")
    else:
        _call("POST", f"/pods/{_pod_id()}/{args.command}")
        print(args.command, "sent")


if __name__ == "__main__":
    main()
