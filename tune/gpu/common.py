"""Shared by the GPU scripts: the student's prompt, exactly as the agent's request renders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creo.agent.llm import SYSTEM

DATA = Path(__file__).resolve().parents[1] / "data"

# The answer's shape, as a grammar for constrained decoding (what Ollama's `format`
# does for the agent's local model): field names, order, 1-6 citations, confidence
# in [0, 1]. The category and the evidence ids stay the model's choice, so a category
# the runbook doesn't allow or an invented id is still possible and still penalised.
ID = r'"[a-z]+[0-9]*"'
SHAPE = (
    r'\{"category": "[a-z_]{2,40}", "evidence": \[' + ID + r"(, " + ID + r"){0,5}\], "
    r'"confidence": (0|1|0\.[0-9]{1,2}|1\.0)\}'
)


def read(name: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (DATA / f"{name}.jsonl").read_text().splitlines() if x]


def messages(facts: dict[str, Any]) -> list[dict[str, str]]:
    """The same two messages creo.agent.llm.LLMRequest.messages() sends."""
    return [
        {"role": "system", "content": SYSTEM["classify"]},
        {"role": "user", "content": json.dumps(facts, sort_keys=True)},
    ]


def prompt(tok: Any, facts: dict[str, Any]) -> str:
    # Qwen3 without thinking: the template closes an empty <think> block itself.
    return str(
        tok.apply_chat_template(
            messages(facts), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    )
