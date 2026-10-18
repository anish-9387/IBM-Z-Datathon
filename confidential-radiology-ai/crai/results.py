"""Stores caller-supplied results with a timestamp and machine label. This module does
not validate provenance; missing entries are displayed as 'not measured'."""
import json
import platform
import time

from . import config

PATH = config.HOME / "results.json"
EXPECTED = ["environment", "security_tests", "secure_vs_clear", "latency_clear", "latency_secure",
            "onnx_operators", "telum_operator_placement", "telum_benchmark", "granite_explainer", "data_split", "eval_zero_shot_test",
            "eval_finetuned_test"]


def load() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def save(key: str, value, note: str = None):
    r = load()
    r[key] = {"value": value, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "machine": platform.machine(), **({"note": note} if note else {})}
    PATH.write_text(json.dumps(r, indent=1, default=str))
    return r[key]


def table_html() -> str:
    r = load()
    rows = []
    for k in EXPECTED:
        if k in r:
            v = json.dumps(r[k]["value"], default=str)
            v = v if len(v) < 220 else v[:220] + "…"
            rows.append(f"<tr><td><b>{k}</b></td><td>{v}</td><td>{r[k]['measured_at']} ({r[k]['machine']})</td>"
                        f"<td>{r[k].get('note', '')}</td></tr>")
        else:
            rows.append(f"<tr><td><b>{k}</b></td><td style='color:#b26b00'><i>not measured</i></td><td></td><td></td></tr>")
    return ("<table style='font:12px system-ui;border-collapse:collapse'><tr><th align=left>metric</th>"
            "<th align=left>value</th><th align=left>measured</th><th align=left>note</th></tr>"
            + "".join(rows) + "</table>")
