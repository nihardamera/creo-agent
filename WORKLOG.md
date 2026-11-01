# Worklog

A new chat can resume from this file: read it top to bottom, then continue
with **Next**.

## Ground rules

- Synthetic data only. Everything here (the company, its fleet, hostnames,
  runbooks, ticket formats, alerts) is invented for this repo.
- Never write "production", "deployed" or "handles N alerts/day". Every number
  in the README comes from a committed eval artifact.
- Work in milestones M0–M6. Each ends with an explain-back gate; the
  "Design decisions" section of the README is written by Nihar, not generated.
- Check current library versions and docs before adding a dependency.
- Small commits, each passing `ruff`, `mypy` and `pytest`.

## Milestones

| | Milestone | State |
|---|---|---|
| M0 | Skeleton: uv, ruff, mypy, pytest, pre-commit + gitleaks, CI, license | done; CI green |
| M1 | Synthetic dataset (≥ 1,000 alerts, ≥ 100 adversarial, labels, card) | next |
| M2 | The workflow (graph, runbooks, approvals, executor, tickets, replayer) | |
| M3 | Trajectory evals as a CI gate | |
| M4 | Tracing and cost | |
| M5 | MCP + security (OWASP Agentic Top 10 mapping) | |
| M6 | Ship it (README, demo video, fresh-clone test) | |

## Log

### 2026-11-01 — M0

Done:
- Repo skeleton: `uv` project (Python 3.12, `uv_build`), package `creo/`,
  `tests/`, MIT license.
- Dev tools pinned to the latest releases as of today: pytest 9.2.0,
  hypothesis 6.170.14, ruff 0.17.3, mypy 2.4.1 (strict), pre-commit 4.7.0.
- pre-commit: ruff check + format, gitleaks v8.30.1, basic file hygiene hooks.
- GitHub Actions: lint, format check, mypy, pytest; a separate gitleaks job
  over the full history. Actions pinned: checkout@v7, setup-uv@v10.3.3
  (setup-uv has no floating major tag), gitleaks-action@v3.
- iCloud: `.venv` is a symlink to `.venv.nosync`; runtime state will live in
  `var` → `var.nosync`.
- Remote: github.com/nihardamera/creo-agent (private until M6). First CI run
  green on both jobs (checks, secrets).

Next:
- M1: name the fictional company and fleet, define the alert and label
  schemas, write the generator.

Open questions:
- The fictional company's name and fleet layout (decide at the start of M1).
