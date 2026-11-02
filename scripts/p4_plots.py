"""Charts for the P4 report, from committed logs only.

    uv run --with matplotlib==3.11.2 python scripts/p4_plots.py

- tune/curves/grpo.png: GRPO per-step curves (reward components, KL, entropy,
  completion length) for the three runs, one metric per panel.
- tune/bench/serving.png: output throughput and p95 latency against concurrency
  for vLLM on the RTX 5090 and llama.cpp on the M4, one measure per panel.

Colours: the dataviz skill's reference categorical slots 1-5 in fixed order
(validated there; the validator needs Node, which this Mac doesn't have).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, MUTED, GRID, BASE, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def log(name: str) -> list[dict[str, float]]:
    d = json.loads((ROOT / "tune" / "curves" / f"{name}.json").read_text())
    return [e for e in d["log"] if "reward" in e]


def smooth(xs: list[float], k: int = 10) -> list[float]:
    return [
        sum(xs[max(0, i - k + 1) : i + 1]) / len(xs[max(0, i - k + 1) : i + 1])
        for i in range(len(xs))
    ]


def grpo() -> None:
    base, caution, sft = log("grpo-base-0.6b"), log("grpo-base-0.6b-caution"), log("grpo-sft-0.6b")
    runs = [
        ("GRPO from base (M3 rewards)", base, 0),
        ("+ caution cost (continues from step 300)", caution, len(base)),
        ("GRPO from SFT (98 prompts)", sft, 0),
    ]
    panels = [
        ("Total reward", "reward", False),
        ("Correct (component)", "rewards/correct/mean", False),
        ("Unsafe + injection penalties", None, False),
        ("KL to reference (log scale)", "kl", True),
        ("Entropy", "entropy", False),
        ("Completion length (tokens)", "completions/mean_length", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    for ax, (title, key, logy) in zip(axes.flat, panels, strict=True):
        for i, (label, rows, offset) in enumerate(runs):
            steps = [r["step"] + offset for r in rows]
            if key is None:
                ys = [r["rewards/unsafe/mean"] + r["rewards/injection/mean"] for r in rows]
            else:
                ys = [max(r[key], 1e-4) if logy else r[key] for r in rows]
            ax.plot(steps, ys, color=SLOTS[i], linewidth=0.6, alpha=0.25)
            ax.plot(steps, smooth(ys), color=SLOTS[i], linewidth=2, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("optimizer step")
        if logy:
            ax.set_yscale("log")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "GRPO on CREO's classify step, Qwen3-0.6B (thin: per step; thick: 10-step mean)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.savefig(ROOT / "tune" / "curves" / "grpo.png", dpi=150)


def serving() -> None:
    configs = [
        ("vLLM bf16, prefix cache on (RTX 5090)", "vllm-5090-bf16-prefix-unique"),
        ("vLLM bf16, prefix cache off (RTX 5090)", "vllm-5090-bf16-noprefix-unique"),
        ("vLLM AWQ W4A16, prefix cache on (RTX 5090)", "vllm-5090-awq-prefix-unique"),
        ("llama.cpp Q8_0 (Apple M4, Metal)", "llamacpp-m4-q8_0-unique"),
        ("llama.cpp Q4_K_M (Apple M4, Metal)", "llamacpp-m4-q4_k_m-unique"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for i, (label, name) in enumerate(configs):
        d = json.loads((ROOT / "tune" / "bench" / f"{name}.json").read_text())
        rows = [r for r in d["levels"] if "error" not in r]
        c = [r["concurrency"] for r in rows]
        a.plot(
            c,
            [r["output_tokens_per_s"] for r in rows],
            color=SLOTS[i],
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
        b.plot(
            c,
            [r["latency_ms_p95"] for r in rows],
            color=SLOTS[i],
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
    for ax, title, ylabel in (
        (a, "Output throughput", "output tokens / s (log)"),
        (b, "p95 end-to-end latency per classify call", "ms (log)"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    a.legend(loc="center right", fontsize=8)
    fig.suptitle(
        "Serving the tuned 0.6B classifier: agent-shaped requests, each unique"
        " (missing points: the server failed at that level)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.savefig(ROOT / "tune" / "bench" / "serving.png", dpi=150)


if __name__ == "__main__":
    grpo()
    serving()
