# Dataset statistics (generated)

Written by `python -m creo.dataset.generate` for version 1.1.0. Do not edit by hand.

## Scenario classes

| Class | dev | test | adversarial_dev | adversarial_test | Expected final states |
|---|--:|--:|--:|--:|---|
| `approval_denied` | 12 | 28 | 0 | 0 | escalated 40 |
| `approval_timeout` | 12 | 28 | 0 | 0 | escalated 40 |
| `conflicting_tags` | 12 | 28 | 0 | 0 | escalated 40 |
| `duplicate` | 18 | 42 | 0 | 0 | duplicate 60 |
| `escalate_now` | 22 | 53 | 0 | 0 | escalated 75 |
| `executor_error` | 8 | 17 | 0 | 0 | escalated 25 |
| `fix_fails` | 15 | 35 | 0 | 0 | escalated 50 |
| `fix_retry_succeeds` | 12 | 28 | 0 | 0 | resolved 40 |
| `fix_succeeds` | 39 | 91 | 0 | 0 | resolved 130 |
| `flapping` | 12 | 28 | 0 | 0 | escalated 40 |
| `huge_message` | 6 | 14 | 0 | 0 | escalated 5, resolved 15 |
| `malformed_payload` | 15 | 35 | 0 | 0 | rejected 50 |
| `missing_runbook_tag` | 12 | 28 | 0 | 0 | escalated 13, resolved 27 |
| `multi_alert_incident` | 12 | 28 | 0 | 0 | correlated 40 |
| `never_recovers_no_fix` | 18 | 42 | 0 | 0 | escalated 60 |
| `orphan_recovery` | 6 | 14 | 0 | 0 | ignored 20 |
| `recovered_before_triage` | 15 | 35 | 0 | 0 | resolved 50 |
| `recovers_during_wait` | 33 | 77 | 0 | 0 | resolved 110 |
| `recovers_on_recheck` | 24 | 56 | 0 | 0 | resolved 80 |
| `stale` | 9 | 21 | 0 | 0 | ignored 30 |
| `tier0_host` | 8 | 17 | 0 | 0 | escalated 25 |
| `unicode` | 12 | 28 | 0 | 0 | escalated 12, resolved 28 |
| `unknown_monitor` | 10 | 25 | 0 | 0 | escalated 35 |
| `warn_only` | 8 | 17 | 0 | 0 | ignored 25 |
| `adversarial` | 0 | 0 | 60 | 100 | escalated 144, rejected 16 |
| **total** | 350 | 815 | 60 | 100 | 1325 cases |

## Attacks

Family f0 is in `adversarial_dev`; f1 and f2 are in `adversarial_test`.

| Category | f0 | f1 | f2 | Where the design should stop it |
|---|--:|--:|--:|---|
| `direct_override` | 6 | 5 | 5 | `screen` |
| `fake_approval_text` | 6 | 5 | 5 | `screen` |
| `fake_approval_fields` | 6 | 5 | 5 | refused at ingress |
| `tag_reserved_key` | 6 | 5 | 5 | `screen` |
| `tag_contradicts_inventory` | 6 | 5 | 5 | `select_runbook` |
| `exfiltration` | 6 | 5 | 5 | `screen` |
| `tool_call_text` | 6 | 5 | 5 | `screen` |
| `obfuscated` | 6 | 5 | 5 | `screen` |
| `param_injection` | 6 | 5 | 5 | `screen` |
| `indirect_diagnostics` | 6 | 5 | 5 | `diagnostics` |

## Expected classifications (runbooks with judgement)

| Runbook | Category | Cases |
|---|---|--:|
| `cpu-high` | `runaway_process` | 27 |
| `cpu-high` | `sustained_load` | 19 |
| `cpu-high` | `transient_spike` | 20 |
| `event-log-errors` | `hardware_or_os` | 5 |
| `event-log-errors` | `service_crash` | 42 |
| `event-log-errors` | `transient` | 19 |
| `event-log-errors` | `unknown` | 9 |
| `http-check-failed` | `app_pool_stopped` | 34 |
| `http-check-failed` | `transient` | 14 |
| `http-check-failed` | `upstream_or_network` | 18 |
| `memory-high` | `app_pool_leak` | 17 |
| `memory-high` | `service_leak` | 14 |
| `memory-high` | `transient` | 27 |
| `memory-high` | `unknown` | 13 |

## Expected model calls per alert

| Calls | Cases |
|--:|--:|
| 0 | 613 |
| 1 | 542 |
| 2 | 170 |

Mean: 0.666 expected model calls per alert, if the agent follows the labels.
