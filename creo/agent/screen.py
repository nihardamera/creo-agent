"""The screen: deterministic checks on untrusted text and tags, before anything acts.

A monitor's notification describes a state ("disk D: is 93% full"). It has
no reason to address the automation, claim an approval, ask for secrets,
carry code or tool-call syntax, hide characters, or link outside the
company. Each rule below encodes one of those principles; none copies a
phrase from the attack data.

Honest caveat: the same author wrote these rules and the held-out attacks,
so a held-out score is still optimistic. The external red-team set (M3)
is the real test.

Findings are codes, never the matched text, so a flagged alert's content
can't reach the model through the flag itself.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from creo.config import Policy
from creo.webhook import AlertWebhook, parse_tags

HIDDEN = re.compile("[​-‏‪-‮⁠-⁤﻿]")

# (code, pattern). Case-insensitive, applied to free text.
TEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (code, re.compile(p, re.IGNORECASE))
    for code, p in (
        # Speaking to the automation, or about its instructions.
        ("addresses_automation", r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
         r"\b(instructions?|rules|runbooks?|polic(y|ies)|previous|above|guardrails?)\b"),
        ("addresses_automation", r"\b(system|assistant|developer)\s*(prompt|override|message)\b"),
        ("addresses_automation", r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?"
         r"(automation|assistant|bot|operator\s+ai|ai|llm|model)\b"),
        ("addresses_automation", r"\b(you are|act as|pretend to be)\b[^.\n]{0,30}"
         r"\b(assistant|automation|ai|admin|root)\b"),
        # Approvals, authorisation and change references never belong in an alert.
        ("claims_approval", r"\bapprov\w*|\bauthori[sz]\w*|\bsign(ed)?[\s-]?off\b|\bCHG\d+\b"),
        # Fleet-wide scope: a single alert is about one host.
        ("fleet_wide_scope", r"\b(all|every|each)\s+(\w+\s+){0,2}(servers?|hosts?|machines?|"
         r"nodes?|services?|instances?)\b"),
        # Secrets and exfiltration targets.
        ("asks_for_secrets", r"\b(passwords?|passwd|api[\s_-]?keys?|secrets?|credentials?|"
         r"connection\s+strings?|private\s+keys?|(access|auth|bearer|session|api)\s+tokens?|"
         r"web\.config|environment\s+variables|"
         r"env\s+vars)\b"),
    )
)  # fmt: skip

# Code or tool-call syntax. Applied to alert text only: diagnostic output
# legitimately contains things like "code=exited".
CODE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (code, re.compile(p, re.IGNORECASE))
    for code, p in (
        ("code_or_tool_syntax", r"[{\[,]\s*\"(tool|function|action|cmd|command|arguments|"
         r"name|params?)\"\s*:"),
        ("code_or_tool_syntax", r"<\s*/?\s*(function|tool|invoke|action|call)\w*[\s>]"),
        ("code_or_tool_syntax", r"```"),
        ("code_or_tool_syntax", r"\b(host|hosts|process_name|pid|service_name|cmd|command|"
         r"action|approver)=\S"),
    )
)  # fmt: skip

URL = re.compile(r"\bhttps?://([^/\s:?#]+)", re.IGNORECASE)
BLOB = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")

# Tag values: an identifier, except a few keys whose values have their own shape.
SAFE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
VALUE_RULES: dict[str, re.Pattern[str]] = {
    "device": re.compile(r"^([A-Z]:|/([A-Za-z0-9_.\-]+/?)*)$"),
}


@dataclass(frozen=True)
class Finding:
    code: str
    field: str


def _scripts(word: str) -> set[str]:
    out = set()
    for ch in word:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            out.add(name.split(" ", 1)[0])
    return out


def _mixed_script(text: str) -> bool:
    """A word that mixes Latin with Cyrillic or Greek letters (a homoglyph trick)."""
    for word in re.findall(r"\w+", text):
        s = _scripts(word)
        if "LATIN" in s and s & {"CYRILLIC", "GREEK"}:
            return True
    return False


def _encoded_text(text: str) -> bool:
    """A base64 run that decodes to readable text: an instruction smuggled past the rules."""
    for blob in BLOB.findall(text):
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        printable = sum(32 <= b < 127 for b in raw)
        if raw and printable / len(raw) > 0.9 and b" " in raw:
            return True
    return False


def _foreign_urls(text: str, allowed: tuple[str, ...]) -> bool:
    for host in URL.findall(text):
        h = host.lower().rstrip(".")
        if not any(h == d or h.endswith("." + d) for d in allowed):
            return True
    return False


def screen_text(text: str, field: str, policy: Policy, *, code_rules: bool = True) -> list[Finding]:
    found: list[Finding] = []
    if HIDDEN.search(text):
        found.append(Finding("hidden_characters", field))
    if _mixed_script(text):
        found.append(Finding("mixed_script", field))
    if _encoded_text(text):
        found.append(Finding("encoded_text", field))
    if _foreign_urls(text, policy.allowed_link_domains):
        found.append(Finding("foreign_url", field))
    rules = TEXT_RULES + (CODE_RULES if code_rules else ())
    for code, pattern in rules:
        if pattern.search(text):
            found.append(Finding(code, field))
    return _unique(found)


def screen_tags(tags: str, policy: Policy) -> list[Finding]:
    found = []
    allowed = set(policy.allowed_tag_keys)
    for key, value in parse_tags(tags):
        if key not in allowed:
            found.append(Finding("tag_key_not_allowed", "tags"))
            continue
        rule = VALUE_RULES.get(key, SAFE_VALUE)
        if ".." in value or not rule.match(value):
            found.append(Finding("tag_value_unsafe", "tags"))
    return _unique(found)


def screen_alert(alert: AlertWebhook, policy: Policy) -> list[Finding]:
    """Every check on the parts of an alert that its sender controls."""
    found = screen_tags(alert.tags, policy)
    found += screen_text(alert.title, "title", policy)
    found += screen_text(alert.message, "message", policy)
    found += screen_text(alert.status, "status", policy, code_rules=False)
    return _unique(found)


def screen_diagnostics(outputs: dict[str, Any], policy: Policy) -> list[Finding]:
    """Free text that diagnostics bring back from a host is untrusted too."""
    found: list[Finding] = []
    for name, out in outputs.items():
        for text in _strings(out):
            found += screen_text(text, f"diagnostics.{name}", policy, code_rules=False)
    return _unique(found)


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _unique(found: list[Finding]) -> list[Finding]:
    return list(dict.fromkeys(found))
