# CREO

An alert-remediation agent built workflow-first. A LangGraph state machine
owns the whole flow, runbooks are configuration rather than retrieval, the LLM
is called only at the few steps that need judgement, and no server changes
without a named human approving that exact action.

> **Status: early development (M0, skeleton).** CREO runs only on synthetic
> data: a fictional company, a fictional fleet, generated alerts and invented
> runbooks. It is not deployed anywhere and has never handled a real alert.
> Every number this README will report comes from a committed eval artifact.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
