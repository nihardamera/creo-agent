"""The eval's own arithmetic: CIs, unsafe-action detection, the gate's rules."""

from __future__ import annotations

import dataclasses
from typing import Any

from creo.dataset.io import load
from creo.dataset.schema import Case
from creo.eval import baseline
from creo.eval.gate import check
from creo.eval.metrics import bootstrap, per_case, summarise
from creo.harness import run_case
from tests.oracle import LabelOracleLLM


def _pair(cls: str) -> tuple[Case, dict[str, Any]]:
    c = next(x for x in load("dev", purpose="tune") if x.labels.scenario_class == cls)
    return c, dataclasses.asdict(run_case(c, LabelOracleLLM(c)))


def test_bootstrap_brackets_the_mean_and_is_deterministic() -> None:
    xs = [1.0] * 80 + [0.0] * 20
    r = bootstrap(xs)
    assert r.value == 0.8 and r.lo is not None and r.hi is not None
    assert r.lo < 0.8 < r.hi and r.lo > 0.7 and r.hi < 0.9
    assert bootstrap(xs) == r
    assert bootstrap([]).value is None


def test_an_all_zero_rate_still_has_an_upper_bound() -> None:
    none = bootstrap([0.0] * 815)
    assert (none.value, none.lo) == (0.0, 0.0) and none.hi == 0.0045  # exact, not [0, 0]
    every = bootstrap([1.0] * 100)
    assert every.hi == 1.0 and every.lo == 0.9638


def test_a_forbidden_execution_counts_as_unsafe() -> None:
    case, out = _pair("approval_denied")  # forbids everything
    assert not per_case(case, out)["unsafe"]
    tampered = {**out, "executed": [{"action": "restart_iis", "params": {"host": "x"}}]}
    row = per_case(case, tampered)
    assert row["unsafe"] and row["unsafe_actions"] == ["restart_iis"]


def test_a_correct_run_scores_perfectly() -> None:
    case, out = _pair("fix_retry_succeeds")
    row = per_case(case, out)
    assert row["final_ok"] and row["path_exact"] and row["proposal_exact"]
    assert row["tool_args_valid"] == [1.0, 1.0]
    assert row["step_precision"] == row["step_recall"] == 1.0


def test_gate_fails_on_unsafe_actions_weaker_injection_resistance_or_a_two_point_drop() -> None:
    case, out = _pair("fix_succeeds")
    good = summarise([per_case(case, out)])
    assert check(good, good) == []
    worse = {**good, "final_state_accuracy": {"value": 0.97}}
    assert check(worse, {**good, "final_state_accuracy": {"value": 1.0}})
    unsafe = {**good, "unsafe_action_rate": {"value": 0.01}, "unsafe_alerts": ["c1"]}
    assert check(unsafe, good)
    base = {**good, "injection_resistance": {"value": 1.0}}
    assert check({**good, "injection_resistance": {"value": 0.94}}, base)
    assert check({**good, "injection_resistance": {"value": 1.0}}, base) == []


def test_rules_baseline_matches_dev_labels() -> None:
    """The rules classifier really reads the typed facts (it is not a label lookup)."""
    from creo.agent.facts import classify_facts
    from creo.config import load_inventory, load_policy, load_runbooks

    rbs, inv, pol = load_runbooks(), load_inventory(), load_policy()
    n = right = 0
    for c in load("dev", purpose="tune"):
        if not c.labels.expected_classification or not c.labels.expected_runbook:
            continue
        host = inv.host(str((c.alert.payload or {})["hostname"]))
        assert host is not None
        diags = {k.value: v for k, v in c.world.diagnostics.items()}
        facts = classify_facts(rbs[c.labels.expected_runbook], host, diags, pol)
        n += 1
        right += baseline.classify(facts) == c.labels.expected_classification
    assert n > 30 and right / n > 0.9


def test_cohen_kappa_known_values() -> None:
    from creo.eval.judge import cohen_kappa

    assert cohen_kappa(["pass", "fail"] * 5, ["pass", "fail"] * 5) == 1.0
    # Agreement no better than chance gives 0.
    assert cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "fail", "pass", "fail"]) == 0.0
    a = ["pass"] * 20 + ["fail"] * 10
    b = ["pass"] * 15 + ["fail"] * 5 + ["pass"] * 3 + ["fail"] * 7
    assert cohen_kappa(a, b) == 0.4286  # (22/30 - 16/30) / (1 - 16/30)


def test_kappa_ci_brackets_kappa_and_is_wide_for_small_samples() -> None:
    from creo.eval.judge import cohen_kappa, kappa_ci

    a = ["pass"] * 12 + ["fail"] * 13
    b = ["pass"] * 9 + ["fail"] * 3 + ["pass"] * 4 + ["fail"] * 9
    k = cohen_kappa(a, b)
    lo, hi = kappa_ci(a, b)
    assert k is not None and lo is not None and hi is not None
    assert lo < k < hi and hi - lo > 0.3  # 25 labels can't pin kappa down
    assert kappa_ci(a, b) == (lo, hi)  # seeded
