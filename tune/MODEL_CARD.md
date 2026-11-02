---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
language: en
pipeline_tag: text-generation
tags:
  - classification
  - grpo
  - distillation
  - synthetic-data
  - agents
---

# creo-classify-0.6b

A Qwen3-0.6B fine-tune that does one job: the `classify` step of
[CREO](https://github.com/nihardamera/creo-agent), an open-source
alert-remediation agent. Given typed diagnostic evidence for a server alert and
the categories its runbook allows, it returns
`{"category": ..., "evidence": [...], "confidence": ...}`.

**Trained and evaluated only on synthetic data** (a fictional company and
fleet). It has never seen a real alert. Do not use it to make decisions about
real systems.

## How it was made

1. SFT (LoRA r32, 2 epochs) on 1,713 answers from Claude Haiku 5.5 that passed
   the agent's validation and named the gold category (rejection sampling).
2. GRPO (TRL 1.17.1, DAPO loss, clip-higher, KL β 0.04 to the SFT model, LoRA,
   60 steps) on the 98 prompts where the SFT model's sampled answers still got
   different rewards. Rewards: valid JSON, correct category, a −3 penalty for a
   wrong fix category or a category the runbook doesn't have, −3 for following
   injected instructions, a length band.
3. LoRA merged into the base weights.

Data, code, rewards and every eval: the
[p4-small-model branch](https://github.com/nihardamera/creo-agent) of the
CREO repo, [report](https://github.com/nihardamera/creo-agent/blob/p4-small-model/docs/P4_REPORT.md).

## Results (synthetic, held out)

| | Correct | Unsafe choice | Followed injected instructions |
|---|---|---|---|
| 185 test requests | 100% [98.0, 100] | 0 | – |
| 185 injection probes (phrasings never trained on) | 100% | 0 | 0 [0, 2.0%] |

In the agent (815 test alerts + 100 attacks) behind a router with an API
fallback: final state 100% [99.6, 100], 0 unsafe actions, 100/100 attacks
stopped, 0 of 185 calls fell back to the API.

Quantized: GGUF Q8_0 and Q4_K_M keep 100% under the agent's JSON schema; an AWQ
W4A16 copy lost one safety item (not published).

## Use

Send the agent's exact request: system prompt `SYSTEM["classify"]` from
`creo/agent/llm.py`, the facts JSON (`sort_keys=True`) as the user message.
**Turn Qwen3's thinking off** (`enable_thinking=False` in the chat template, or
`--default-chat-template-kwargs '{"enable_thinking": false}'` in vLLM,
`--chat-template-kwargs` in llama.cpp); with thinking on, it writes a `<think>`
block until max_tokens. Validate every answer before use, as CREO does.

## Limits

- The task is easy on this synthetic data: hand-written rules also get 100%.
- The training and test data come from one generator and one author.
- The injection probes test text that the agent's redaction boundary normally
  removes before the model sees it.
