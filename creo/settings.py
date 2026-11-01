"""Service settings from environment variables, with demo defaults.

The defaults make `docker compose up` work with no .env. They are demo
values on purpose (all data is synthetic); a real deployment must set its
own secrets. Every service reads the same CREO_CLOCK_EPOCH_MS and
CREO_TIME_SCALE, so they all agree on the (possibly sped-up) time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("CREO_DATA_DIR", "var")))
    webhook_secret: str = field(
        default_factory=lambda: _env("CREO_WEBHOOK_SECRET", "demo-webhook-secret")
    )
    approver_token: str = field(
        default_factory=lambda: _env("CREO_APPROVER_TOKEN", "demo-approver-token")
    )
    executor_token: str = field(
        default_factory=lambda: _env("CREO_EXECUTOR_TOKEN", "demo-executor-token")
    )
    approval_key: bytes = field(
        default_factory=lambda: _env("CREO_APPROVAL_KEY", "demo-approval-signing-key").encode()
    )
    agent_url: str = field(default_factory=lambda: _env("CREO_AGENT_URL", "http://localhost:8000"))
    tickets_url: str = field(
        default_factory=lambda: _env("CREO_TICKETS_URL", "http://localhost:8001")
    )
    sim_url: str = field(default_factory=lambda: _env("CREO_SIM_URL", "http://localhost:8002"))
    # mock (recorded responses), ollama, or openai (any OpenAI-compatible API).
    llm_mode: str = field(default_factory=lambda: _env("CREO_LLM", "mock"))
    cassette: Path = field(
        default_factory=lambda: Path(_env("CREO_CASSETTE", "cassettes/default.jsonl"))
    )
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen3:4b"))
    openai_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", ""))
    openai_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    clock_epoch_ms: int = field(
        default_factory=lambda: int(_env("CREO_CLOCK_EPOCH_MS", str(int(time.time() * 1000))))
    )
    time_scale: float = field(default_factory=lambda: float(_env("CREO_TIME_SCALE", "1")))


class ScaledClock:
    """Wall time, sped up by `scale` from a shared epoch. scale=1 is plain wall time."""

    def __init__(self, epoch_ms: int, scale: float) -> None:
        self.epoch_ms, self.scale = epoch_ms, scale
        self.wall_epoch = epoch_ms / 1000

    def now_ms(self) -> int:
        return int(self.epoch_ms + (time.time() - self.wall_epoch) * 1000 * self.scale)

    def wall_seconds_until(self, t_ms: int) -> float:
        return max(0.0, (t_ms - self.now_ms()) / 1000 / self.scale)


def clock(settings: Settings) -> ScaledClock:
    return ScaledClock(settings.clock_epoch_ms, settings.time_scale)
