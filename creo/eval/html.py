"""A self-contained HTML view of report.json."""

from __future__ import annotations

import html
import json
from typing import Any

HEADLINE = (
    ("final_state_accuracy", "Final-state accuracy"),
    ("path_exact", "Path exact match"),
    ("step_precision", "Step precision"),
    ("step_recall", "Step recall"),
    ("proposal_exact", "Proposals exact"),
    ("tool_args_valid", "Tool arguments valid"),
    ("unsafe_action_rate", "Unsafe-action rate"),
    ("injection_resistance", "Injection resistance"),
    ("classification_accuracy", "Classification accuracy"),
    ("escalation_route_accuracy", "Escalation route accuracy"),
    ("llm_validation_failure_rate", "Model output failed validation"),
    ("llm_calls_per_alert", "Model calls per alert"),
)


def _fmt(m: dict[str, Any] | None, pct: bool = True) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    v, (lo, hi), n = m["value"], m["ci95"], m["n"]
    if pct:
        return f"{100 * v:.1f}% <small>[{100 * lo:.1f}, {100 * hi:.1f}] n={n}</small>"
    return f"{v:.3f} <small>[{lo:.3f}, {hi:.3f}] n={n}</small>"


def render(report: dict[str, Any]) -> str:
    e = html.escape
    meta, s = report["meta"], report["summary"]
    rows = "".join(
        f"<tr><td>{e(label)}</td><td>{_fmt(s.get(key), key != 'llm_calls_per_alert')}</td></tr>"
        for key, label in HEADLINE
    )
    lat = s["llm_latency_ms"]
    rows += f"<tr><td>Model latency p50 / p95</td><td>{lat['p50']} / {lat['p95']} ms</td></tr>"
    tokens = f"{s['tokens_in_per_alert']} / {s['tokens_out_per_alert']}"
    rows += f"<tr><td>Tokens per alert (in / out)</td><td>{tokens}</td></tr>"
    rows += f"<tr><td>$ per 1k alerts</td><td>{s['usd_per_1k_alerts']}</td></tr>"
    classes = "".join(
        f"<tr><td>{e(k)}</td><td>{v['n']}</td><td>{_fmt(v['final_state_accuracy'])}</td>"
        f"<td>{_fmt(v['path_exact'])}</td><td>{v['unsafe']}</td><td>{v['llm_calls_per_alert']}</td></tr>"
        for k, v in report["by_class"].items()
    )
    b = report["classification_baseline"]
    fails = "".join(
        f"<tr><td>{e(f['case'])}</td><td>{e(f['class'])}</td><td>{e(str(f['final_state']))}</td>"
        f"<td>{e(f['expected'])}</td><td>{e(str(f['reason']))}</td></tr>"
        for f in report["failures"][:200]
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>CREO eval report</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;max-width:1100px;color:#222;
background:#fff}}table{{border-collapse:collapse;margin:8px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}}small{{color:#666}}
code{{font-size:12px}}</style></head><body>
<h1>CREO eval report: {e(meta["name"])}</h1>
<p>Synthetic data only. <code>{e(json.dumps(meta))}</code></p>
<h2>Headline (95% bootstrap CI)</h2><table>{rows}</table>
<h2>Classification: model vs rules baseline</h2>
<p>Model {_fmt(b["model_accuracy"])}; rules {_fmt(b["rules_accuracy"])}. {e(b["note"])}.</p>
<h2>By scenario class</h2><table><tr><th>Class</th><th>n</th><th>Final state</th><th>Path exact</th>
<th>Unsafe</th><th>Calls/alert</th></tr>{classes}</table>
<h2>Cases that missed (first 200)</h2><table><tr><th>Case</th><th>Class</th><th>Got</th>
<th>Expected</th><th>Reason</th></tr>{fails}</table></body></html>"""
