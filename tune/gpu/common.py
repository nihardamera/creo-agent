"""Shared by the GPU scripts: the student's prompt, exactly as the agent's request renders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creo.agent.llm import SYSTEM

DATA = Path(__file__).resolve().parents[1] / "data"


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
