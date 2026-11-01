import pytest
from pydantic import ValidationError

from creo.catalog import Action
from creo.config import (
    CONFIG_DIR,
    Category,
    Runbook,
    load_inventory,
    load_policy,
    load_runbooks,
)
from creo.dataset.fleet import build_inventory, render_inventory


def test_committed_inventory_is_up_to_date() -> None:
    committed = (CONFIG_DIR / "inventory.yaml").read_text(encoding="utf-8")
    assert committed == render_inventory(build_inventory()), (
        "run: uv run python -m creo.dataset.fleet"
    )


def test_inventory_hostnames_are_unique() -> None:
    names = [h.name for h in load_inventory().hosts]
    assert len(names) == len(set(names))


def test_runbook_roles_exist_in_inventory() -> None:
    roles = {h.role for h in load_inventory().hosts}
    for rb in load_runbooks().values():
        assert set(rb.roles) <= roles, rb.key
        for cat in rb.categories.values():
            for fix in cat.fixes:
                assert fix.roles is None or set(fix.roles) <= set(rb.roles), rb.key


def test_monitor_type_and_role_pick_at_most_one_runbook() -> None:
    # The missing-runbook-tag fallback relies on this: it must never guess.
    seen: dict[tuple[str, str], str] = {}
    for rb in load_runbooks().values():
        for mt in rb.monitor_types:
            for role in rb.roles:
                assert (mt, role) not in seen, f"{rb.key} overlaps {seen[(mt, role)]}"
                seen[(mt, role)] = rb.key


def test_every_allowlisted_action_is_used_by_a_runbook() -> None:
    used = {
        f.action for rb in load_runbooks().values() for c in rb.categories.values() for f in c.fixes
    }
    assert used == set(Action)


def test_killable_processes_exist_in_the_fleet() -> None:
    images = {p.image for h in load_inventory().hosts for p in h.processes}
    for rb in load_runbooks().values():
        assert set(rb.allowed_processes) <= images, rb.key


def test_tier_zero_hosts_are_protected_by_policy() -> None:
    policy = load_policy()
    assert 0 in policy.no_fix_tiers
    assert all(h.role == "dc" for h in load_inventory().hosts if h.tier == 0)


def test_routing_tags_are_in_the_tag_schema() -> None:
    keys = set(load_policy().allowed_tag_keys)
    assert {"runbook", "monitor_type", "service", "app_pool"} <= keys


def test_fix_plan_without_fixes_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Category.model_validate({"plan": "wait_then_fix"})


def test_judgement_free_runbook_needs_single_default_category() -> None:
    rb = load_runbooks()["win-service-down"].model_dump()
    rb["categories"]["other"] = {"plan": "escalate_now"}
    with pytest.raises(ValidationError):
        Runbook.model_validate(rb)
