# Failure analysis (dev split only)

What went wrong in the "before" run, what was changed because of it, and what
each change did. Every number here was measured on the **dev** split. The
held-out splits (test, adversarial_test) were not looked at while choosing or
tuning fixes; `creo.eval.failures` reads dev through the tuning guard.

Model: qwen3.5:4b on local Ollama, temperature 0, seed 0.

## Before (prompt v1)

Run: [`runs/before-qwen3.5-4b`](runs/before-qwen3.5-4b/). 51 of 410 dev cases
(dev + adversarial_dev) ended on a different path or final state than labelled.

| Count | First thing that went wrong |
|---|---|
| 10 | wrong category: service_crash → hardware_or_os |
| 10 | wrong category: runaway_process → sustained_load |
| 7 | right category, but the cited evidence couldn't bind the fix |
| 6 | wrong category: transient → hardware_or_os |
| 5 | wrong category: transient_spike → sustained_load |
| 5 | wrong category: transient → app_pool_leak |
| 3 | wrong category: transient → service_leak |
| 3 | wrong category: unknown → service_leak |
| 1 | wrong category: unknown → app_pool_leak |
| 1 | path differs (runbook_escalates) |

By runbook: event-log-errors 18, memory-high 17, cpu-high 15.

Nothing unsafe happened in any of them. A wrong category mostly turned a fix
into an escalation, which is why the final-state accuracy (94.6% on dev) is
much higher than the classification accuracy (45.2%).

## Causes, from reading the requests

1. **The model saw category names and nothing else.** `transient_spike`,
   `runaway_process` and `sustained_load` were bare names. Shown Service
   Control Manager 7031 events (a service exiting unexpectedly), it answered
   `hardware_or_os`; shown a memory curve going *down*, it answered "leak".
2. **It cited the events but not the service.** A restart needs one service
   record to name the service. Citing only the 7031 event left nothing to bind
   (the 7 "couldn't bind" cases), so the agent escalated instead.
3. **It doesn't read trends from raw number series.** A process at 98–99%
   CPU in every sample and one falling from 90 to 26 both came back
   `sustained_load`.
4. **The runbook's own description prejudged the answer.** cpu-high was
   described as "Sustained high CPU…". With that one word changed to neutral
   wording, three probe cases that had all come back `sustained_load` came back
   right.

## Changes, each measured on the 93 dev cases that need the model to choose

| Version | Change | Category right | Exact path | Final state right |
|---|---|---|---|---|
| v1 (before) | none | 42 | 43 | 75 |
| v2a | a `means` line for every judgement category in `config/runbooks.yaml`, passed to the model as `category_meanings`; prompt asks it to cite the process or service record | 68 | 68 | 88 |
| v2b | + per-process `cpu_pct_avg`, `cpu_trend`, `mem_mb_change_30m`, `mem_trend`, computed in code at the redaction boundary (`creo/agent/facts.py`); the meanings name these fields | 71 | 71 | 87 |
| **v2c (frozen)** | + neutral cpu-high description | **84** | **84** | **92** |

Runs: [`runs/tune-v2a-dev`](runs/tune-v2a-dev/), [`runs/tune-v2b-dev`](runs/tune-v2b-dev/),
[`runs/tune-v2c-dev`](runs/tune-v2c-dev/). No version produced an unsafe
action or an invalid model output.

Tuning stopped at v2c, before the held-out run, whatever it scored. The 9
dev failures left:

| Count | Wrong category |
|---|---|
| 4 | unknown → transient |
| 2 | runaway_process → sustained_load |
| 2 | transient → app_pool_leak |
| 1 | service_leak → app_pool_leak |

They are left as found, so the after run measures a frozen system.

The config check now refuses a judgement category without a meaning
(`tests/test_config.py`).

## What this does not show

- **The author wrote both sides.** I wrote the dataset generator and the
  category meanings. The held-out split guards against tuning on particular
  cases, not against me describing the evidence my own generator produces.
  The meanings use ordinary Windows facts (event 7031 is a service exit, Ntfs 55
  is file-system corruption), but the test score is still an optimistic
  estimate of how this would do on real alerts.
- **93 cases is small.** On dev, a difference of three cases (v2a → v2b) is
  noise. Each change was made because the requests showed a cause, not to
  chase a score, and the held-out test split gives the number with a CI.
- **The model's confidence is not calibrated.** In the before run it reported
  0.95 on 338 of 357 classifications, right or wrong. Nothing in CREO reads it.

## Provenance

- The before run was recorded at commit `6876ee1`, before reports carried
  the model digest. The tag `qwen3.5:4b` was not re-pulled between runs; the
  tuning runs record digest `7f3251fa…2d13`.
- The tuning runs are marked `-dirty`: each ran on the uncommitted change
  described above, and the changes were committed together after v2c.
