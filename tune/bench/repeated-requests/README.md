# Superseded sweeps (repeated requests)

These sweeps cycled through the 185 test requests with the vLLM cache kept between
concurrency levels, so with prefix caching on, a later level could reuse a whole
earlier request, which real alerts never repeat. They overstate prefix caching
(e.g. 4,201 vs 2,634 output tokens/s at concurrency 32). The sweeps in the parent
folder (`*-unique.json`) give every request a unique last field, so only the shared
prefix (system prompt, categories, meanings) can be cached. Kept for the record.

An earlier run with Qwen3's thinking left on (every answer ran to the 128-token
cap) was discarded and not kept.
