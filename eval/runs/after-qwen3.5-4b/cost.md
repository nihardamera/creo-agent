Measured on `after-qwen3.5-4b`: 1325 alerts, 0.674 model calls, 307 input and 53 output tokens per alert.
Prices checked 2026-11-01 (https://platform.claude.com/docs/en/about-claude/pricing (standard API, global routing, no batch or caching discounts)).

| Model | $ / M input | $ / M output | $ per alert | $ per 1,000 alerts |
|---|--:|--:|--:|--:|
| local qwen3.5:4b (Ollama, Apple M4) | 0.00 | 0.00 | 0.000000 | 0.0000 |
| claude-haiku-5-5 | 0.10 | 0.50 | 0.000065 | 0.0649 |
| claude-sonnet-5-5 | 2.00 | 10.00 | 0.001298 | 1.2982 |

Claude rows use Claude's input token counts for the same requests (1.25x the local count; system prompt and facts, no structured-output schema), and the local model's output counts.
