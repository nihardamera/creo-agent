"""Drive one RunPod GPU pod for P4 through the REST API (key from the Keychain).

    uv run python scripts/pod.py create [--gpu 'NVIDIA GeForce RTX 5090']  # community, CUDA 12.9+
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
# torch 2.13.0 (vllm 0.32.0's pin) built for CUDA 12.9, so hosts with 12.9 or 13.0 drivers
# both qualify; no 4090 with a 13.0 driver was free on 2026-11-02.
IMAGE = "runpod/pytorch:1.5.1-cu1290-torch2130-ubuntu2404"


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
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "containerDiskInGb": 40,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp", "8000/http"],
        "env": {"PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()},
        "interruptible": False,
    }
    # No allowedCudaVersions or supportPublicIp: with either set, every create on
    # 2026-11-02 failed ("no instances"), though the pod we got has a CUDA 13.2
    # driver and a public IP. The setup step checks the driver instead.
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
    # On 2026-11-02 community cloud had no stock; Nihar chose a Secure Cloud 5090 ($1.19/h).
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
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
