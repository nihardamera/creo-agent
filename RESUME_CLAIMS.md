# Resume claims

Every resume bullet about this project sits next to the committed file that
proves it. A claim without a row here does not go on the resume. Every
number was measured on **synthetic** data, and each bullet should say so.

| Claim | Evidence (file, commit or CI run) | Measured on |
|---|---|---|
| Built CREO, an open-source alert-remediation agent: a fixed 18-node LangGraph workflow with durable SQLite checkpoints, config-driven runbooks and human-approved actions | [docs/graph.md](docs/graph.md), [creo/agent/](creo/agent/), [tests/test_durability.py](tests/test_durability.py), [kill-mid-wait evidence](docs/evidence/m2-kill-mid-wait.json) | 207 tests; CI |
| Raised the model's category accuracy from 45% to 90% on held-out alerts through dev-only error analysis, with a 4B local model | [eval/FAILURES.md](eval/FAILURES.md), [before](eval/runs/before-qwen3.5-4b/report.json), [after](eval/runs/after-qwen3.5-4b/report.json) | 185 held-out synthetic judgement alerts; 45.4% [38.4, 53.0] → 90.3% [86.0, 94.1] |
| Raised end-to-end correctness from 93% to 99% of held-out alerts | [before](eval/runs/before-qwen3.5-4b/report.json), [after](eval/runs/after-qwen3.5-4b/report.json) | 815 held-out synthetic alerts; final state 93.3% → 99.3% [98.7, 99.8] |
| Zero forbidden actions executed across 915 held-out alerts, including 100 prompt-injection attacks run with an approver that approves everything | [after report](eval/runs/after-qwen3.5-4b/report.json), [SECURITY.md](SECURITY.md) | 0 of 815 [0, 0.45%]; 100 of 100 attacks stopped [96.4%, 100%] |
| Designed a 1,325-alert labelled synthetic dataset with 160 adversarial cases and held-out splits by phrasing family | [DATASET_CARD.md](DATASET_CARD.md), [dataset/v1/](dataset/v1/) | validator in CI |
| Made trajectory evals a CI gate: a 120-case held-out replay that fails on any unsafe action, weaker injection resistance or a > 2-point accuracy drop | [.github/workflows/ci.yml](.github/workflows/ci.yml), [creo/eval/gate.py](creo/eval/gate.py), [eval/ci_baseline.json](eval/ci_baseline.json) | CI run 41332719314 (all jobs green) |
| Built an MCP server (Streamable HTTP, RS256 JWTs, per-tool scopes) through which write actions are only requests for human approval; mapped the design to the OWASP Top 10 for Agentic Applications | [creo/mcp_server.py](creo/mcp_server.py), [tests/test_mcp.py](tests/test_mcp.py), [SECURITY.md](SECURITY.md) | tests over real HTTP |
| Traced every workflow step, model call and executor job with OpenTelemetry and OpenInference, viewed in Arize Phoenix | [creo/tracing.py](creo/tracing.py), [tests/test_tracing.py](tests/test_tracing.py) | tests |
| Calibrated two LLM judges (a local 9B model and Claude Haiku 5.5) against hand labels on a held-out split, and kept them out of the reported results when neither agreed with a human well enough (best kappa 0.28, target 0.6) | [README](README.md#escalation-note-quality), [eval/judge/](eval/judge/), [creo/eval/judge.py](creo/eval/judge.py) | 50 hand-labelled synthetic notes (25 tune / 25 held out) |
| Measured cost per alert: $0 on the local model, an estimated $0.065 per 1,000 alerts on Claude Haiku 5.5 using Claude's own token counts for the same requests | [cost.md](eval/runs/after-qwen3.5-4b/cost.md), [claude_tokens.json](eval/runs/after-qwen3.5-4b/claude_tokens.json) | 1,325 synthetic alerts; prices checked 2026-11-01 |

## Wording rules

- Say "synthetic" or "held-out synthetic" next to every number.
- Never claim real-world use, real users or an alert volume.
- "The design follows an approach I developed for alert triage at work" is
  fine; never name the employer or claim work results from this repo.
- If an interviewer asks why rules beat the model here, the answer is in the
  README ("What these numbers don't show").
