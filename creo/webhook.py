"""The ingress contract: the JSON body the monitoring system posts to the agent.

It mirrors a Datadog webhook with a custom payload template; each field is
filled from one Datadog template variable (noted beside it). Unknown fields
are refused, so a payload cannot smuggle in extra keys such as "approved_by".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creo.catalog import Transition

# The template a Datadog webhook integration would use to produce this body.
PAYLOAD_TEMPLATE = {
    "id": "$ID",
    "alert_id": "$ALERT_ID",
    "alert_cycle_key": "$ALERT_CYCLE_KEY",
    "aggreg_key": "$AGGREG_KEY",
    "title": "$ALERT_TITLE",
    "message": "$TEXT_ONLY_MSG",
    "transition": "$ALERT_TRANSITION",
    "alert_type": "$ALERT_TYPE",
    "alert_priority": "$ALERT_PRIORITY",
    "date": "$DATE",
    "last_updated": "$LAST_UPDATED",
    "hostname": "$HOSTNAME",
    "metric": "$ALERT_METRIC",
    "query": "$ALERT_QUERY",
    "scope": "$ALERT_SCOPE",
    "status": "$ALERT_STATUS",
    "tags": "$TAGS",
    "org_id": "$ORG_ID",
    "org_name": "$ORG_NAME",
    "link": "$LINK",
}

_DIGITS = r"^[0-9]{1,20}$"


class AlertWebhook(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_DIGITS)
    alert_id: str = Field(pattern=_DIGITS)
    alert_cycle_key: str = Field(max_length=64)
    aggreg_key: str = Field(max_length=64)
    title: str = Field(min_length=1, max_length=2_000)
    # Bounded overall by policy.max_payload_bytes, checked before parsing.
    message: str
    transition: Transition
    alert_type: Literal["error", "warning", "success", "info"]
    alert_priority: Literal["P1", "P2", "P3", "P4", "P5", ""]
    date: int = Field(gt=0)  # epoch milliseconds
    last_updated: int = Field(gt=0)
    hostname: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
    metric: str = Field(max_length=200)
    query: str = Field(max_length=2_000)
    scope: str = Field(max_length=2_000)
    status: str = Field(max_length=2_000)
    tags: str = Field(max_length=4_000)  # comma-separated, as Datadog sends $TAGS
    org_id: str = Field(max_length=32)
    org_name: str = Field(max_length=200)
    link: str = Field(max_length=500)


def parse_tags(tags: str) -> list[tuple[str, str]]:
    """Split Datadog's comma-separated $TAGS into (key, value) pairs.

    A tag splits on its first colon, so `device:C:` is ("device", "C:").
    A bare tag such as `monitor` has an empty value.
    """
    pairs: list[tuple[str, str]] = []
    for raw in tags.split(","):
        tag = raw.strip()
        if not tag:
            continue
        key, _, value = tag.partition(":")
        pairs.append((key, value))
    return pairs
