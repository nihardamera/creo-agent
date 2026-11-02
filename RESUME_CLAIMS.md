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

## P4: a tuned small model for the classify step

| Claim | Evidence (file, commit or CI run) | Measured on |
|---|---|---|
| Distilled Claude Haiku 5.5 into Qwen3-0.6B (rejection-sampled SFT) and GRPO-tuned it on verifiable rewards with an unsafe-action penalty, for the agent's classify step | [model on Hugging Face](https://huggingface.co/nihar-damera/creo-classify-0.6b) (revision 254829047ce3), [docs/P4_REPORT.md](docs/P4_REPORT.md), [creo/tune/](creo/tune/), [tune/gpu/](tune/gpu/), [rewards and their tests](tests/test_tune.py) | 185 held-out synthetic requests: 100% [98.0, 100] correct, 0 unsafe ([report](tune/runs/grpo-sft-0.6b/report.json)) |
| In the agent behind a local-first router with API fallback: 100% final-state accuracy against 99.1% for the API model, 0 unsafe actions, 100/100 attacks stopped, and no API calls ($0 vs $0.10 per 1,000 alerts) | [router run](eval/runs/p4-router-tuned-0.6b/report.json), [API run](eval/runs/p4-haiku-only/report.json), [cost](eval/runs/p4-cost.json), [creo/agent/llm.py](creo/agent/llm.py) (`RouterLLM`) | 815 held-out synthetic alerts + 100 attacks; final state 100% [99.6, 100] vs 99.1% [98.4, 99.8]; 0 of 185 classify calls fell back |
| Cut classify latency from 8.8 s (4B local model) to 0.71 s p50 / 1.32 s p95 on a laptop with a GGUF Q8_0 build, at no quality loss | [gguf-q8_0](tune/runs/gguf-q8_0/report.json), [qwen3.5-4b](tune/runs/qwen3.5-4b/report.json) | 185 held-out synthetic requests; Apple M4, llama.cpp b12032 |
| Held the tuned model to 0% on held-out injection probes, where the agent's current 4B model follows the injected instruction 28% of the time | [tune/runs/grpo-sft-0.6b](tune/runs/grpo-sft-0.6b/report.json), [tune/runs/qwen3.5-4b](tune/runs/qwen3.5-4b/report.json) | 185 synthetic probes, phrasing families never trained on; 0% [0, 2.0] vs 28.1% [21.6, 34.6] |
| Found and fixed reward hacking in GRPO: the policy dodged the unsafe penalty by never proposing fixes (68% over-caution); a caution cost raised accuracy 66% → 82% with unsafe choices still at 0 | [eval/REWARD_HACKING.md](eval/REWARD_HACKING.md), [curves](tune/curves/grpo.png) | 185 held-out synthetic requests; teacher-free GRPO from base Qwen3-0.6B |
| Benchmarked serving: vLLM on an RTX 5090 (≈70 ms per call; prefix caching 1.3–1.4× throughput on agent requests) vs llama.cpp on an M4; measured quantization quality (GGUF Q8/Q4 lossless, AWQ lost one safety item) | [tune/bench/](tune/bench/), [serving chart](tune/bench/serving.png), [AWQ](tune/runs/grpo-sft-0.6b-awq/report.json) | agent-shaped requests, each unique, concurrency 1–64 |

P4 wording rules:
- Say plainly that SFT did nearly all the work (99.5% before GRPO) and that
  GRPO's own gain on top of SFT was one test item, inside the interval.
- "No API calls" is the measured fallback count on this data, not a guarantee.
- The cost saving is $0.10 per 1,000 alerts at Haiku prices: say "no API cost
  in the normal path", not "Nx cheaper".

## Wording rules

- Say "synthetic" or "held-out synthetic" next to every number.
- Never claim real-world use, real users or an alert volume.
- "The design follows an approach I developed for alert triage at work" is
  fine; never name the employer or claim work results from this repo.
- If an interviewer asks why rules beat the model here, the answer is in the
  README ("What these numbers don't show").
