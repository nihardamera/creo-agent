"""The fictional fleet of Kestrel Freight, an invented logistics company.

Every name here (company, sites, hosts, services, app pools, tasks) is made up
for this repo. `python -m creo.dataset.fleet` writes config/inventory.yaml;
a test checks the committed file is up to date.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import yaml

from creo.config import CONFIG_DIR, Host, Inventory, Process

COMPANY = "Kestrel Freight (fictional)"
DOMAIN = "kestrel-freight.example"  # .example is reserved (RFC 2606); it can't resolve
SITES = ("ash", "elm")
SEED = 20261101

Env = Literal["prod", "staging"]

APP_SERVICES: dict[str, tuple[str, ...]] = {
    # service -> the processes it runs (the first is the service's own image)
    "KfRateCalc": ("KfRateCalc.exe", "KfRateCalcWorker.exe"),
    "KfLabelSpooler": ("KfLabelSpooler.exe", "KfLabelRender.exe"),
    "KfManifestSync": ("KfManifestSync.exe", "KfManifestDiff.exe"),
    "KfDockSched": ("KfDockSched.exe",),
    "KfEdiBridge": ("KfEdiBridge.exe",),
    "KfTrackApi": ("KfTrackApi.exe",),
}
APP_POOLS = ("TrackHubPool", "CustomerPortalPool", "CarrierApiPool", "AdminPool")
HTTP_CHECKS = {
    "TrackHubPool": "trackhub-home",
    "CustomerPortalPool": "portal-login",
    "CarrierApiPool": "carrier-api-health",
    "AdminPool": "admin-ping",
}
BATCH_TASKS = ("NightlyManifestExport", "InvoiceBatch", "RateTableRefresh", "EdiArchiveSweep")
QUEUE_UNITS = ("kf-shipment-consumer.service", "kf-tracking-consumer.service")
QUEUE_CONSUMERS = ("shipment-events", "tracking-events", "invoice-events")


def _web(name: str, site: str, rng: random.Random, tier: int, env: Env) -> Host:
    pools = tuple(sorted(rng.sample(APP_POOLS, k=rng.randint(2, 3))))
    return Host(
        name=name,
        site=site,
        role="web",
        os="windows",
        tier=tier,
        env=env,
        services=("W3SVC", "WAS"),
        app_pools=pools,
        drives=("C:", "D:"),
        processes=tuple(Process(image="w3wp.exe", owner=p) for p in pools),
        http_checks=tuple(HTTP_CHECKS[p] for p in pools),
        certs=(f"{pools[0].removesuffix('Pool').lower()}.{DOMAIN}",),
    )


def _app(name: str, site: str, rng: random.Random, tier: int, env: Env) -> Host:
    services = tuple(sorted(rng.sample(sorted(APP_SERVICES), k=rng.randint(2, 3))))
    return Host(
        name=name,
        site=site,
        role="app",
        os="windows",
        tier=tier,
        env=env,
        services=services,
        drives=("C:", "D:"),
        processes=tuple(Process(image=img, owner=s) for s in services for img in APP_SERVICES[s]),
    )


def build_inventory() -> Inventory:
    rng = random.Random(SEED)
    hosts: list[Host] = []
    for site in SITES:
        n_web, n_app, n_bat, n_sql, n_fs, n_prx, n_mq = (
            (6, 8, 3, 3, 2, 3, 3) if site == "ash" else (4, 6, 2, 2, 1, 2, 2)
        )
        for i in range(1, n_web + 1):
            tier = 1 if i <= 2 else 2
            hosts.append(_web(f"{site}-web-{i:02d}", site, rng, tier=tier, env="prod"))
        for i in range(1, n_app + 1):
            tier = 1 if i <= 3 else 2
            hosts.append(_app(f"{site}-app-{i:02d}", site, rng, tier=tier, env="prod"))
        for i in range(1, n_bat + 1):
            tasks = tuple(sorted(rng.sample(BATCH_TASKS, k=rng.randint(2, 3))))
            hosts.append(
                Host(
                    name=f"{site}-bat-{i:02d}",
                    site=site,
                    role="batch",
                    os="windows",
                    tier=2,
                    env="prod",
                    services=("KfBatchAgent",),
                    drives=("C:", "D:"),
                    tasks=tasks,
                    processes=(Process(image="KfBatchAgent.exe", owner="KfBatchAgent"),),
                )
            )
        for i in range(1, n_sql + 1):
            hosts.append(
                Host(
                    name=f"{site}-sql-{i:02d}",
                    site=site,
                    role="sql",
                    os="windows",
                    tier=1,
                    env="prod",
                    services=("MSSQLSERVER", "SQLSERVERAGENT"),
                    drives=("C:", "D:", "E:"),
                    processes=(Process(image="sqlservr.exe", owner="MSSQLSERVER"),),
                )
            )
        for i in range(1, n_fs + 1):
            hosts.append(
                Host(
                    name=f"{site}-fs-{i:02d}",
                    site=site,
                    role="fs",
                    os="windows",
                    tier=2,
                    env="prod",
                    services=("LanmanServer", "DFSR"),
                    drives=("C:", "D:"),
                )
            )
        hosts.append(
            Host(
                name=f"{site}-dc-01",
                site=site,
                role="dc",
                os="windows",
                tier=0,
                env="prod",
                services=("NTDS", "DNS", "Netlogon"),
                drives=("C:",),
                processes=(Process(image="lsass.exe", owner="NTDS"),),
            )
        )
        for i in range(1, n_prx + 1):
            hosts.append(
                Host(
                    name=f"{site}-prx-{i:02d}",
                    site=site,
                    role="proxy",
                    os="linux",
                    tier=1,
                    env="prod",
                    units=("nginx.service", "node_exporter.service"),
                    drives=("/", "/var"),
                    certs=(f"edge-{site}.{DOMAIN}",),
                )
            )
        for i in range(1, n_mq + 1):
            hosts.append(
                Host(
                    name=f"{site}-mq-{i:02d}",
                    site=site,
                    role="queue",
                    os="linux",
                    tier=1,
                    env="prod",
                    units=("rabbitmq-server.service", *QUEUE_UNITS),
                    drives=("/", "/var"),
                    consumers=QUEUE_CONSUMERS,
                )
            )
    # A small staging footprint, tier 3, at the ash site.
    hosts.append(_web("stg-web-01", "ash", rng, tier=3, env="staging"))
    hosts.extend(_app(f"stg-app-{i:02d}", "ash", rng, tier=3, env="staging") for i in (1, 2))
    return Inventory(company=COMPANY, domain=DOMAIN, hosts=tuple(hosts))


def render_inventory(inv: Inventory) -> str:
    header = (
        "# GENERATED by `python -m creo.dataset.fleet`. Do not edit by hand.\n"
        "# The fictional fleet of an invented company. No real hosts, names or data.\n"
    )
    body = yaml.safe_dump(
        inv.model_dump(mode="json", exclude_defaults=True), sort_keys=False, width=100
    )
    return header + body


def main(out: Path = CONFIG_DIR / "inventory.yaml") -> None:
    out.write_text(render_inventory(build_inventory()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
