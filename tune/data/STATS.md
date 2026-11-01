# P4 data (generated)

Written by `python -m creo.tune.data`. Do not edit by hand.

| File | Items | Categories (gold) | Probe families |
|---|--:|---|---|
| `train.jsonl` | 1525 | 11 distinct, 104-324 each | - |
| `train_probes.jsonl` | 300 | 11 distinct, 16-62 each | f0 300 |
| `grpo.jsonl` | 821 | 11 distinct, 56-171 each | - |
| `grpo_probes.jsonl` | 205 | 11 distinct, 10-43 each | f0 205 |
| `val.jsonl` | 277 | 11 distinct, 19-58 each | - |
| `val_probes.jsonl` | 60 | 11 distinct, 2-13 each | f0 60 |
| `test.jsonl` | 185 | 11 distinct, 3-39 each | - |
| `test_probes.jsonl` | 185 | 11 distinct, 3-39 each | f1 93, f2 92 |

Exact-request overlap with `test` (SHA-256 of the canonical facts): train 0, grpo 0, val 0. Exact repeats dropped while building (of a test request or of an earlier train/val item): train 15, val 3, grpo 19.

## Test categories

| Runbook | Category | Items |
|---|---|--:|
| `cpu-high` | `runaway_process` | 17 |
| `cpu-high` | `sustained_load` | 10 |
| `cpu-high` | `transient_spike` | 15 |
| `event-log-errors` | `hardware_or_os` | 3 |
| `event-log-errors` | `service_crash` | 28 |
| `event-log-errors` | `transient` | 13 |
| `event-log-errors` | `unknown` | 6 |
| `http-check-failed` | `app_pool_stopped` | 26 |
| `http-check-failed` | `transient` | 10 |
| `http-check-failed` | `upstream_or_network` | 16 |
| `memory-high` | `app_pool_leak` | 8 |
| `memory-high` | `service_leak` | 9 |
| `memory-high` | `transient` | 16 |
| `memory-high` | `unknown` | 8 |
