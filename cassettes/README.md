# Recorded model responses

`default.jsonl` holds responses recorded from a real model, one JSON object
per line, keyed by a hash of the request (prompt version, task and typed
facts). Mock-LLM mode (`CREO_LLM=mock`, the default) replays them, so tests,
CI and the demo need no model server and no API key.

A request with no recording fails like any invalid model output: the agent
takes its deterministic fallback (a judgement step escalates; an escalation
note uses the template). Until the file is filled by a recorded run of a
real model, judgement runbooks escalate in mock mode.

`v1-before.jsonl` is the "before" run (prompt v1, qwen3.5:4b, all 1,325
cases), recorded at commit `6876ee1`. Prompt v2 changed the request key, so
the current code can't replay it; it is kept so the before numbers in
`eval/runs/before-qwen3.5-4b/` can be checked by checking out that commit
and replaying them with `--llm mock --cassette cassettes/v1-before.jsonl`.
