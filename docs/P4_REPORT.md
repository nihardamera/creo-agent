# A tuned 0.6B model for CREO's classify step

CREO calls a model at two points: `classify` (pick the runbook category that the
diagnostic evidence fits) and `summarize` (write the escalation note). Only
`classify` has a verifiable answer, so it is the step a small model was tuned
for. Every number below is on **synthetic** data and comes from a committed
file; intervals are 95% (bootstrap, or exact Clopper-Pearson when a rate is
0 or 1).

## Results

### In the agent

CREO's own trajectory eval: 815 held-out test alerts plus 100 held-out attacks,
each run through the full workflow.

| | Current agent: qwen3.5:4b | Claude Haiku 5.5 (API) | **Router: tuned Qwen3-0.6B first, API fallback** |
|---|---|---|---|
| Final state correct (test) | 99.3% [98.7, 99.8] | 99.1% [98.4, 99.8] | **100%** [99.6, 100] |
| Exact path (test) | 97.6% | 98.5% | **100%** |
| Classify answer correct (185 judgement alerts) | 90.3% | 94.6% | **100%** [98.0, 100] |
| Model outputs failing the agent's validation | 0% | 2.5% | **0%** |
| Alerts with a forbidden action executed | 0 of 815 | 0 of 815 | **0 of 815** [0, 0.45%] |
| Held-out attacks stopped | 100 of 100 | 100 of 100 | **100 of 100** [96.4, 100] |
| Classify calls that fell back to the API | – | – | **0 of 185** |
| API cost per 1,000 alerts | $0 | $0.10 | **$0** |
| Classify latency p50 / p95 | 8.8 s / 13.4 s | 1.6 s / 4.1 s | 2.4 s / 4.7 s (sharing the GPU with the 4B note writer); 0.71 s / 1.32 s alone |

Router configuration: classify goes to the tuned model (GGUF Q8_0 on
llama.cpp, Apple M4) and falls back to Haiku 5.5 when the answer fails the
agent's validation; notes come from the local qwen3.5:4b, as before. Files:
[eval/runs/p4-router-tuned-0.6b](../eval/runs/p4-router-tuned-0.6b/report.json),
[p4-haiku-only](../eval/runs/p4-haiku-only/report.json),
[after-qwen3.5-4b](../eval/runs/after-qwen3.5-4b/report.json),
[cost](../eval/runs/p4-cost.json) (measured tokens, prices checked 2026-11-01).

### Per step

The classify step alone, on the same 185 requests, and on 185 **injection
probes**: those requests with attacker text added in an `alert_message` field,
as if the redaction boundary had failed (in the agent that text never reaches
the model). Probe phrasings in families f1 and f2 were never trained on.

| | Correct | Valid JSON | Unsafe choice | Probes: correct | Probes: followed the injection |
|---|---|---|---|---|---|
| qwen3.5:4b (current agent model) | 89.2% | 100% | 2.2% | 61.6% | 28.1% [21.6, 34.6] |
| Claude Haiku 5.5 | 95.7% | 95.7% | 0% | 78.4% (30 refusals) | 0.5% |
| Qwen3-0.6B base, grammar-constrained | 50.8% | 89.2% | 34.6% | 30.3% | 41.1% |
| **SFT** (distilled from Haiku) | 99.5% [98.4, 100] | 99.5% | 0% | 100% | 0% |
| **GRPO from SFT** (deployed) | **100%** [98.0, 100] | 100% | 0% | 100% | 0% [0, 2.0] |
| GRPO from base, teacher-free, M3 rewards | 66.0% | 100% | 0% | 65.9% | 0% |
| GRPO from base + caution fix | 81.6% [75.7, 87.0] | 100% | 0% | 80.0% | 1.1% |

Files: `tune/runs/<name>/report.json`. "Unsafe choice" means a fix category
other than the gold one, or a category the runbook doesn't have.

### Quantization

| Deployed model as | Test correct | Unsafe | Probes correct | p50 / p95 per call |
|---|---|---|---|---|
| bf16 (vLLM) | 100% | 0 | 100% | |
| AWQ W4A16 (vLLM, unconstrained) | 98.9% | **1 of 185** | 97.3% | |
| GGUF Q8_0 (llama.cpp, M4, schema) | 100% | 0 | 100% | 709 / 1,320 ms |
| GGUF Q4_K_M (llama.cpp, M4, schema) | 100% | 0 | 100% | 799 / 1,548 ms |

AWQ cost real quality: one wrong fix (`app_pool_leak -> service_leak`, on the
same alert with and without injection), and on 4 f2 probes it copied the
injected JSON's shape and dropped `evidence`. The GGUF copies run under the
agent's JSON schema, which removes the format failure mode.

### Serving

![serving benchmark](../tune/bench/serving.png)

Agent requests (each made unique, so only the shared prefix can be cached),
concurrency 1 to 64, from `creo/tune/bench.py`:

- **vLLM 0.32 on an RTX 5090**: about 70 ms end to end per classify call at
  concurrency 1; 3,791 output tokens/s at 32 with prefix caching on, against
  2,706 with it off (1.40x; 1.32x at concurrency 8, where TTFT p50 is 24.6 vs
  52.4 ms). At $1.19/h that is about $0.09 per million output tokens. AWQ is
  no faster at this size.
- **llama.cpp on an Apple M4 (16 GB)**: 755 ms at concurrency 1 (Q8_0), a peak
  of 123 tokens/s at 8, and the server drops connections at 32. Fine for one
  agent; not a serving box.
- Concurrency 64 is unreliable in every vLLM configuration (throughput falls
  below 32): most likely the Python benchmark client on the pod's CPU.

## How it was built

1. **Data** (`creo/tune/data.py`). Train (1,525), GRPO (821) and val (277) are
   fresh alerts from CREO's generator under new seeds, balanced over (runbook,
   category); 37 exact repeats of a test request were dropped. Test is the
   agent's own 185 held-out requests. Probe family f0 is in train/val, f1/f2
   only in test.
2. **Teacher** (`creo/tune/teacher.py`). Haiku 5.5 through the Batches API
   ($0.18 for 2,162 requests). Its answers that pass the agent's validation
   and name the gold category were kept: 1,713 of 1,825.
3. **SFT** (`tune/gpu/sft.py`). Qwen3-0.6B, LoRA r32, 2 epochs, 7.8 minutes on
   an RTX 5090.
4. **Rewards** (`creo/tune/score.py`), each logged separately: format (+0.5 /
   −1), correct (+1), unsafe (−3), followed the injection (−3), length band
   (−0.2), and later caution (−1). Unit tests check that known hacks score low.
5. **GRPO** (`tune/gpu/grpo.py`). TRL 1.17.1, DAPO loss (its default, set
   explicitly), clip-higher (ε_high 0.28), KL β 0.04 to the starting model
   (TRL's default is 0), LoRA, vLLM colocated, 8 completions per prompt.
   ![GRPO curves](../tune/curves/grpo.png)
6. **Audit** ([eval/REWARD_HACKING.md](../eval/REWARD_HACKING.md)).
7. **Serving and router**: `creo/agent/llm.py` (`RouterLLM`), GGUF via
   llama.cpp b12032, AWQ via llmcompressor 0.15.

Spend: about $0.40 of API calls (teacher, baselines, the Haiku agent run) and
$4.53 of GPU (RunPod, Secure Cloud RTX 5090; community cloud had no stock).

## What these numbers don't show

- **The task is easy on this data.** CREO's hand-written rules get 100% of the
  judgement cases from the typed facts. The generator and the rules have the
  same author, so a model reaching 100% here says little about messy real
  evidence.
- **Distillation did nearly all the work.** SFT alone reached 99.5% on test.
  GRPO on top fixed the one remaining test item (+0.5 points, inside the
  interval); after SFT only 98 of 1,026 GRPO prompts still produced different
  rewards within a group, so RL had little to learn from.
- **RL without a teacher is where RL showed up, and it reward-hacked.**
  Teacher-free GRPO from the base model removed every unsafe choice in 60
  steps by learning to never propose a fix (68% over-caution). A caution cost
  brought it to 81.6%, still well below SFT.
- **The teacher-free model depends on the grammar.** Unconstrained it is 0%
  valid; it never learned the output format itself.
- **The injection probes are mine,** written by the same author as the
  defences, as in CREO's adversarial set.
- **Cost is small either way.** The API path costs $0.10 per 1,000 alerts;
  the saving is real but not large at this volume. Latency (0.7 s vs 1.6 s
  per classify call) and running offline matter as much as cost.

## Design decisions

*To be written by Nihar.* (Deferred to interview prep, with the explain-back
gates listed in WORKLOG.md.)

## What failed

*To be written by Nihar.* Facts to draw on, from the work log: the base model's
0% unconstrained validity; the always-escalate reward hack; the flat training
correctness during the fix run; Haiku refusing 30 probes; the serving
benchmark's two invalid first runs (Qwen3 thinking left on; repeated requests
crediting prefix caching); AWQ's lost safety item; RunPod having no 4090 stock.
