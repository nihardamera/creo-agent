import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from creo.catalog import Transition
from creo.webhook import PAYLOAD_TEMPLATE, AlertWebhook, parse_tags

GOOD = {
    "id": "5309152963",
    "alert_id": "44250693",
    "alert_cycle_key": "0a58",
    "aggreg_key": "6301",
    "title": "[Triggered on {host:elm-app-04}] Cannot resolve a name",
    "message": "text",
    "transition": "Triggered",
    "alert_type": "error",
    "alert_priority": "P3",
    "date": 1788853152000,
    "last_updated": 1788853152000,
    "hostname": "elm-app-04",
    "metric": "dns.can_resolve",
    "query": "q",
    "scope": "host:elm-app-04",
    "status": "s",
    "tags": "env:prod,role:app",
    "org_id": "48213",
    "org_name": "Kestrel Freight (fictional)",
    "link": "https://monitoring.kestrel-freight.example/event/1",
}


def test_good_payload_parses() -> None:
    assert AlertWebhook.model_validate(GOOD).transition is Transition.TRIGGERED


def test_template_covers_every_field() -> None:
    assert set(PAYLOAD_TEMPLATE) == set(AlertWebhook.model_fields)


@pytest.mark.parametrize(
    "change",
    [
        {"approved_by": "ops-lead"},  # smuggled field
        {"hostname": "elm app 04"},
        {"date": -1},
        {"transition": "Paused"},
        {"id": "12a"},
        {"title": ""},
    ],
)
def test_bad_payloads_are_refused(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AlertWebhook.model_validate({**GOOD, **change})


def test_tags_split_on_first_colon() -> None:
    assert parse_tags("device:C:, monitor ,env:prod") == [
        ("device", "C:"),
        ("monitor", ""),
        ("env", "prod"),
    ]


_key = st.text("abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)
_value = st.text("abcdefghijklmnopqrstuvwxyzABC0123456789_.:/-", max_size=20)


@given(st.lists(st.tuples(_key, _value), max_size=10))
def test_tags_round_trip(pairs: list[tuple[str, str]]) -> None:
    text = ",".join(f"{k}:{v}" if v else k for k, v in pairs)
    assert parse_tags(text) == pairs
