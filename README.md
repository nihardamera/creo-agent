# CREO

An alert-remediation agent built workflow-first. A LangGraph state machine
owns the whole flow, runbooks are configuration rather than retrieval, the LLM
is called only at the few steps that need judgement, and no server changes
without a named human approving that exact action.

> **Status: in development (M2 of M6 done).** CREO runs only on synthetic
> data: a fictional company, a fictional fleet, generated alerts and invented
> runbooks. It is not deployed anywhere and has never handled a real alert.
> Every number this README reports comes from a committed artifact.

## Quickstart

Needs Docker. No API keys: the model runs in mock mode (recorded responses).

```bash
docker compose up -d                  # agent, executor, sim, mock tickets
docker compose run --rm replayer      # replay 50 synthetic alerts, auto-approving as scripted
open http://localhost:8000            # dashboard: runs, waits, approvals
```

Time runs 60× faster in the demo, so a 10-minute runbook wait takes 10 seconds.
`docker compose down -v` removes everything.

**Known gap:** the recorded-response file (`cassettes/default.jsonl`) is still
empty, so in mock mode every runbook that needs the model's judgement
escalates through the deterministic fallback. On the 50-alert replay, all 50
runs finished and 41 took their exact labelled path; the other 9 are those
fallbacks ([evidence](docs/evidence/m2-replay-50.json)).

## What's here so far

- [`config/`](config/): 15 runbooks, the global policy, the fictional fleet.
- [`DATASET_CARD.md`](DATASET_CARD.md): 1,325 labelled synthetic alerts,
  including 160 attacks.
- [`docs/graph.md`](docs/graph.md): the workflow graph, generated from code.
- [`WORKLOG.md`](WORKLOG.md): what was done, decisions, what's next.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
