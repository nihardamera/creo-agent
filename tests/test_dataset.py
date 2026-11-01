import hashlib
import json
from collections.abc import Callable
from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st

from creo.catalog import FinalState, Node
from creo.config import load_runbooks
from creo.dataset.generate import Generator, case_line, recovery_time, wait_nodes
from creo.dataset.io import DATASET_DIR, HeldOutSplitError, load, load_all, verify_manifest
from creo.dataset.schema import SPLITS, Case
from creo.dataset.validate import check_case, validate


@cache
def generated() -> tuple[Case, ...]:
    return tuple(Generator().generate())


@cache
def committed() -> tuple[Case, ...]:
    return tuple(load_all())


def _split_hashes(cases: tuple[Case, ...]) -> dict[str, str]:
    out = {}
    for split in SPLITS:
        lines = [case_line(c) for c in cases if c.split == split]
        out[split] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    return out


def test_committed_dataset_matches_the_generator() -> None:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    want = {s: manifest["splits"][s]["sha256"] for s in SPLITS}
    assert _split_hashes(generated()) == want, "run: uv run python -m creo.dataset.generate"


def test_generation_is_deterministic() -> None:
    again = tuple(Generator().generate())
    assert _split_hashes(again) == _split_hashes(generated())


def test_committed_files_match_manifest_and_config() -> None:
    assert verify_manifest() == []


def test_committed_dataset_has_no_label_errors() -> None:
    assert validate(list(committed())) == []


def test_dataset_meets_size_requirements() -> None:
    cases = committed()
    assert len(cases) >= 1_000
    assert sum(c.labels.adversarial for c in cases) >= 100
    assert len({c.labels.scenario_class for c in cases if not c.labels.adversarial}) >= 15
    assert {c.labels.expected_final_state for c in cases} == set(FinalState)


def test_tuning_code_cannot_read_held_out_splits() -> None:
    assert load("dev", purpose="tune")
    assert load("adversarial_dev", purpose="tune")
    for split in ("test", "adversarial_test"):
        with pytest.raises(HeldOutSplitError):
            load(split, purpose="tune")


def test_attack_phrasings_in_test_never_appear_in_dev() -> None:
    def families(split: str) -> set[tuple[str, str]]:
        return {
            (c.labels.attack.category, c.labels.attack.family)
            for c in committed()
            if c.split == split and c.labels.attack
        }

    dev, test = families("adversarial_dev"), families("adversarial_test")
    assert dev and test and not dev & test
    assert {cat for cat, _ in dev} == {cat for cat, _ in test}


def test_attacks_never_expect_an_action() -> None:
    for c in committed():
        if c.labels.adversarial:
            assert c.labels.expected_actions == ()
            assert c.labels.forbidden_actions == ("*",)
            assert c.labels.expected_final_state in (FinalState.ESCALATED, FinalState.REJECTED)


# ---- the validator catches what it should --------------------------------------


def _first(cls: str) -> Case:
    if cls == "classified":  # any case whose model step has an expected category
        return next(c for c in generated() if c.labels.expected_classification)
    return next(c for c in generated() if c.labels.scenario_class == cls)


def _with(
    case: Case,
    *,
    labels: dict | None = None,
    world: dict | None = None,
    payload: dict | None = None,
) -> Case:
    update: dict = {}
    if labels:
        update["labels"] = case.labels.model_copy(update=labels)
    if world:
        update["world"] = case.world.model_copy(update=world)
    if payload is not None:
        update["alert"] = case.alert.model_copy(update={"payload": payload})
    return case.model_copy(update=update)


def _other_host(c: Case) -> Case:
    call = c.labels.expected_actions[0]
    moved = call.model_copy(update={"params": {**call.params, "host": "elm-sql-01"}})
    return _with(c, labels={"expected_actions": (moved,), "expected_proposals": (moved,)})


def _leak(field: str, text: Callable[[Case], str]) -> Callable[[Case], Case]:
    def mutate(c: Case) -> Case:
        p = dict(c.alert.payload or {})
        p[field] = f"{p[field]} {text(c)}"
        return _with(c, payload=p)

    return mutate


# name -> (class to start from, mutation, text the targeted rule's error must contain).
# Checking the message matters: a defect caught by some other rule would hide
# a rule that never fires.
PLANTED: dict[str, tuple[str, Callable[[Case], Case], str]] = {
    "final state contradicts path": (
        "fix_succeeds",
        lambda c: _with(c, labels={"expected_final_state": FinalState.ESCALATED}),
        "path ends at resolve but final state is escalated",
    ),
    "action on another host": ("fix_succeeds", _other_host, "host other than the alert's"),
    "execution without approval": (
        "fix_succeeds",
        lambda c: _with(c, world={"approvals": ("deny",)}),
        "not approved first",
    ),
    "scenario class in title": (
        "fix_succeeds",
        _leak("title", lambda c: "(fix succeeds)"),
        "label leak: title contains label value 'fix succeeds'",
    ),
    "runbook key in message": (
        "fix_succeeds",
        _leak("message", lambda c: f"see {c.labels.expected_runbook}"),
        "label leak: message contains the runbook key",
    ),
    "category in message": (
        "classified",
        _leak("message", lambda c: str(c.labels.expected_classification)),
        "label leak: message contains label value",
    ),
    "unknown runbook": (
        "fix_succeeds",
        lambda c: _with(c, labels={"expected_runbook": "no-such-runbook"}),
        "unknown runbook no-such-runbook",
    ),
    "tag key outside schema": (
        "fix_succeeds",
        _leak("tags", lambda c: ",approver:bob"),
        "tag keys outside the schema: ['approver']",
    ),
    "hidden character": (
        "unicode",
        _leak("message", lambda c: "\u200b"),
        "zero-width or bidi control characters",
    ),
    "path missing finalize": (
        "fix_succeeds",
        lambda c: _with(c, labels={"expected_path": c.labels.expected_path[:-1]}),
        "must start at ingest and end at finalize",
    ),
    "malformed but not rejected": (
        "malformed_payload",
        lambda c: _with(c, labels={"expected_final_state": FinalState.ESCALATED}),
        "ingress would refuse it",
    ),
    "llm step not on path": (
        "recovers_during_wait",
        lambda c: _with(c, labels={"expected_llm_steps": (Node.ESCALATE,)}),
        "llm step escalate is not an LLM node on the path",
    ),
}


def test_clean_cases_pass() -> None:
    for cls in {cls for cls, _, _ in PLANTED.values()}:
        assert check_case(_first(cls)) == []


@pytest.mark.parametrize("name", sorted(PLANTED))
def test_validator_catches_planted_defect_with_the_right_rule(name: str) -> None:
    cls, mutate, expected = PLANTED[name]
    errors = check_case(mutate(_first(cls)))
    assert any(expected in e for e in errors), f"{name}: wanted {expected!r}, got {errors}"


def test_duplicate_case_ids_are_caught() -> None:
    a = _first("fix_succeeds")
    b = _first("stale").model_copy(update={"id": a.id})
    assert "duplicate case ids" in validate([a, b])


# ---- the wait loop -------------------------------------------------------------

WAITING = sorted(k for k, rb in load_runbooks().items() if rb.wait.rechecks)


@given(key=st.sampled_from(WAITING), seed=st.integers(0, 10_000), webhook=st.booleans())
def test_recovery_inside_a_wait_always_resolves_within_the_rechecks(
    key: str, seed: int, webhook: bool
) -> None:
    import random

    rb = load_runbooks()[key]
    nodes, recovered = wait_nodes(rb, recovery_time(rb, random.Random(seed)), webhook)
    assert recovered
    assert nodes.count(Node.WAIT) <= rb.wait.rechecks
    # A Recovered webhook wakes the run mid-wait; without one, a recheck notices.
    assert nodes[-1] is (Node.WAIT if webhook else Node.RECHECK)


@pytest.mark.parametrize("key", WAITING)
def test_no_recovery_runs_every_recheck(key: str) -> None:
    rb = load_runbooks()[key]
    nodes, recovered = wait_nodes(rb, None, webhook=False)
    assert not recovered
    assert nodes == [Node.WAIT, Node.RECHECK] * rb.wait.rechecks


def test_dataset_card_headline_numbers_match_the_manifest() -> None:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    card = (DATASET_DIR.parents[1] / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert f"**{manifest['total_cases']:,} labelled" in card
    assert f"including {manifest['adversarial_cases']}\nadversarial" in card
    assert manifest["version"] in card.splitlines()[0]
    for split in SPLITS:
        assert f"| `{split}` | {manifest['splits'][split]['cases']} |" in card
