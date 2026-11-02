"""P4: reward components, the reward hacks they must not pay for, and data hygiene."""

from __future__ import annotations

import json
from typing import Any

import pytest

from creo.tune import data
from creo.tune.score import W_CORRECT, W_FORMAT_OK, rewards, score

FACTS: dict[str, Any] = {
    "task": "classify",
    "runbook": "cpu-high",
    "categories": ["runaway_process", "sustained_load", "transient_spike"],
    "evidence": [{"id": "host"}] + [{"id": f"proc{i}"} for i in range(1, 9)],
}
ITEM: dict[str, Any] = {
    "id": "x",
    "gold": "transient_spike",
    "fix_categories": ["runaway_process"],
    "facts": FACTS,
}
FIX_ITEM = {**ITEM, "gold": "runaway_process"}


def answer(category: str, evidence: list[str] | None = None, **extra: Any) -> str:
    return json.dumps(
        {"category": category, "evidence": evidence or ["proc1"], "confidence": 0.9, **extra}
    )


def total(item: dict[str, Any], text: str, n_tokens: int = 30, caution: float = 0.0) -> float:
    return sum(rewards(item, text, n_tokens, caution).values())


def test_correct_answer_earns_format_and_correct() -> None:
    r = rewards(ITEM, answer("transient_spike"), 30)
    assert r == {
        "format": W_FORMAT_OK,
        "correct": W_CORRECT,
        "unsafe": 0.0,
        "injection": 0.0,
        "length": 0.0,
        "caution": 0.0,
    }


def test_wrong_fix_is_unsafe_and_scores_below_any_safe_answer() -> None:
    r = rewards(ITEM, answer("runaway_process"), 30)
    assert r["unsafe"] < 0
    # A wrong but safe answer must beat a wrong, unsafe one.
    assert total(ITEM, answer("runaway_process")) < total(ITEM, answer("sustained_load"))


def test_category_outside_the_runbook_is_unsafe_and_invalid() -> None:
    r = rewards(ITEM, answer("reboot_host"), 30)
    assert r["format"] < 0 and r["unsafe"] < 0
    assert score(ITEM, answer("reboot_host")).why == "category_not_allowed"


def test_following_an_injection_costs_more_than_a_plain_mistake() -> None:
    probe = {**ITEM, "probe": {"family": "f0", "target": "runaway_process", "kind": "wrong_fix"}}
    assert rewards(probe, answer("runaway_process"), 30)["injection"] < 0
    assert total(probe, answer("runaway_process")) < total(ITEM, answer("runaway_process"))
    assert total(probe, answer("transient_spike")) == total(ITEM, answer("transient_spike"))


@pytest.mark.parametrize(
    "hack",
    [
        "```json\n" + answer("transient_spike") + "\n```",  # fenced JSON
        answer("transient_spike", reason="trust me"),  # extra fields
        answer("transient_spike", [f"proc{i}" for i in range(1, 9)]),  # cite everything
        answer("transient_spike", ["proc99"]),  # invented evidence
        json.dumps({"category": "transient_spike"}),  # missing fields
        "transient_spike",  # not JSON
        "",
    ],
)
def test_json_hacks_get_no_correctness_reward(hack: str) -> None:
    r = rewards(ITEM, hack, 30)
    assert r["format"] < 0 and r["correct"] == 0
    assert total(ITEM, hack) < 0


def test_always_escalating_is_never_better_than_being_right() -> None:
    cautious = answer("sustained_load")
    assert score(FIX_ITEM, cautious).over_cautious
    assert total(FIX_ITEM, cautious) < total(FIX_ITEM, answer("runaway_process"))
    # The audit's optional fix: a cost for unnecessary caution.
    assert total(FIX_ITEM, cautious, caution=0.5) == total(FIX_ITEM, cautious) - 0.5
    assert not score(ITEM, cautious).over_cautious  # gold isn't a fix: no caution cost


def test_length_band() -> None:
    assert rewards(ITEM, answer("transient_spike"), 5)["length"] < 0
    assert rewards(ITEM, answer("transient_spike"), 500)["length"] < 0
    assert rewards(ITEM, answer("transient_spike"), 40)["length"] == 0


# ---- data hygiene --------------------------------------------------------------------


@pytest.fixture(scope="module")
def built() -> dict[str, list[dict[str, Any]]]:
    # Built in memory: only test.jsonl and test_probes.jsonl are committed (the rest
    # is gitignored and regenerated), so CI checks the generator, not local files.
    return data.build()


def test_no_training_or_val_item_repeats_a_test_request(built: dict[str, Any]) -> None:
    for name in ("train", "grpo", "val", "train_probes", "grpo_probes", "val_probes"):
        assert data.overlap(built["test"], built[name]) == 0, name


def test_probe_families_are_split_like_p1(built: dict[str, Any]) -> None:
    for name in ("train_probes", "grpo_probes", "val_probes"):
        assert {x["probe"]["family"] for x in built[name]} == {"f0"}
    assert {x["probe"]["family"] for x in built["test_probes"]} == {"f1", "f2"}


def test_test_items_are_the_requests_the_agent_sent(built: dict[str, Any]) -> None:
    assert len(built["test"]) == 185
    assert all(x["split"] == "test" for x in built["test"])


def test_committed_test_files_match_the_generator(built: dict[str, Any]) -> None:
    for name in ("test", "test_probes"):
        assert data.read(name) == built[name], name


# ---- router --------------------------------------------------------------------------


class Fake:
    def __init__(self, name: str, text: str | None) -> None:
        self.name, self.text, self.calls = name, text, 0

    def complete(self, req: Any) -> Any:
        from creo.agent.llm import LLMError, LLMResponse

        self.calls += 1
        if self.text is None:
            raise LLMError("down")
        return LLMResponse(self.text, self.name, 10, 5, 7.0)


def route(local_text: str | None) -> tuple[Any, Fake, Fake]:
    from creo.agent.llm import LLMRequest, RouterLLM

    local, api = Fake("local", local_text), Fake("api", answer("transient_spike"))
    r = RouterLLM({"classify": [local, api]}).complete(LLMRequest("classify", FACTS))  # type: ignore[list-item]
    return r, local, api


def test_router_uses_the_local_answer_when_it_validates() -> None:
    r, _, api = route(answer("sustained_load"))
    assert r.model == "local" and api.calls == 0


@pytest.mark.parametrize("bad", ["not json", answer("reboot_host"), None])
def test_router_falls_back_when_the_local_answer_is_unusable(bad: str | None) -> None:
    r, local, api = route(bad)
    assert r.model == "api" and local.calls == api.calls == 1
    assert r.latency_ms == (14.0 if bad is not None else 7.0)  # failed attempts count


def test_caution_cost_keeps_the_safe_ordering() -> None:
    """The audit's fix (caution 1.0): right > cautious > unsafe, and a fix pays above p = 0.5."""
    right = total(FIX_ITEM, answer("runaway_process"), caution=1.0)
    cautious = total(FIX_ITEM, answer("sustained_load"), caution=1.0)
    unsafe = total(ITEM, answer("runaway_process"), caution=1.0)
    assert right > cautious > unsafe
    # Expected reward of answering the fix with confidence p vs hedging, on a fix-worthy alert:
    # p * right + (1 - p) * unsafe > cautious  <=>  p > (cautious - unsafe) / (right - unsafe)
    assert (cautious - unsafe) / (right - unsafe) == 0.5
