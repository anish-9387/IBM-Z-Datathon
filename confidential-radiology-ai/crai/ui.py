"""Notebook demo UI (ipywidgets). The UI is the 'hospital workstation': it can show the image
the clinician selected BEFORE encryption, then only ever shows ciphertext, statuses, and the
verified, signed result that comes back from the guest."""
import html
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import config, crypto, explain, model, pipeline

_EXPLAINER = None


def explainer():
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = explain.GraniteExplainer()
    return _EXPLAINER

STEP_LABELS = [("encrypt", "1 · Encrypt", "AES-256-GCM, per-study key"),
               ("attestation", "2 · Attest", "guest + model hashes vs policy"),
               ("key_release", "3 · Key release", "broker wraps key to guest"),
               ("inference", "4 · Inference", "decrypt in guest, ResNet-50, wipe"),
               ("output_guard", "5 · Sign output", "fixed schema, Ed25519"),
               ("verification", "6 · Verify", "PACS checks token + signature")]
COLORS = {"pending": ("#8a8f98", "#f1f2f4", "waiting"), "running": ("#b26b00", "#fff4e0", "running…"),
          "ok": ("#1e7d32", "#e7f5ea", "ok"), "refused": ("#b3261e", "#fde8e7", "REFUSED"),
          "rejected": ("#b3261e", "#fde8e7", "REJECTED"), "failed": ("#b3261e", "#fde8e7", "FAILED")}
SCENARIOS = [("Normal request", "normal"),
             ("Attack: tampered model file (1 byte)", "bad_model"),
             ("Attack: modified guest code", "bad_guest"),
             ("Attack: ciphertext altered in vault", "bad_ct")]


def banner_html():
    return (f"<div style='padding:10px 14px;border-radius:8px;background:#fff4e0;color:#5c3b00;"
            f"border:1px solid #e0a040;font:600 13px system-ui'>⚠ {html.escape(config.MODE_LABEL)}. "
            f"Process separation, real AES-256-GCM, real signatures and real attestation checks run "
            f"here; the attestation measurement itself is self-reported by the guest.</div>")


def pipeline_html(states: dict, notes: dict = None):
    notes = notes or {}
    cards = []
    for key, title, sub in STEP_LABELS:
        fg, bg, word = COLORS[states.get(key, "pending")]
        note = html.escape(notes.get(key, sub))
        cards.append(f"<div style='flex:1 1 140px;min-width:130px;border:2px solid {fg};background:{bg};"
                     f"border-radius:8px;padding:8px 10px;color:#1b1d21'>"
                     f"<div style='font:600 13px system-ui'>{title}</div>"
                     f"<div style='font:600 11px system-ui;color:{fg};margin:2px 0'>{word}</div>"
                     f"<div style='font:11px system-ui;color:#444'>{note}</div></div>")
    arrow = "<div style='align-self:center;color:#8a8f98;font:16px system-ui'>→</div>"
    return f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin:8px 0'>{arrow.join(cards)}</div>"


def result_html(res: dict, ct_preview: str):
    if res.get("refusal") or not res.get("verified"):
        why = res.get("refusal") or ("rate limited" if res["stages"].get("rate_limit") == "refused"
                                     else "output/verification failed")
        return (f"<div style='border:2px solid #b3261e;background:#fde8e7;border-radius:8px;padding:12px;"
                f"color:#1b1d21;font:13px system-ui'><b>Refused - no key, no score.</b><br>Reason: "
                f"<code>{html.escape(str(why))}</code>. The study stayed ciphertext; nothing but this "
                f"refusal code left the guest.</div>")
    p = res["result"]["payload"]
    checks = "".join(f"<li>{'✓' if v else '✗'} {html.escape(k)}</li>" for k, v in res["checks"].items())
    t = res.get("guest_timing") or {}
    timing = ", ".join(f"{k[:-3]} {v:.0f} ms" for k, v in t.items())
    return (f"<div style='border:2px solid #1e7d32;background:#e7f5ea;border-radius:8px;padding:12px;"
            f"color:#1b1d21;font:13px system-ui'>"
            f"<div style='font:600 22px system-ui'>Pneumonia probability: {p['pneumonia_probability']}</div>"
            f"<div style='color:#444;margin:4px 0'>Research prototype - not a diagnosis. Model output only.</div>"
            f"<div>model_hash <code>{p['model_hash'][:16]}…</code> · study_hash "
            f"<code>{p['study_hash'][:16]}…</code></div>"
            f"<div style='margin-top:6px'><b>Verification</b><ul style='margin:2px 0 0 18px'>{checks}</ul></div>"
            f"<div style='color:#444'>Host saw only ciphertext: <code>{ct_preview}…</code></div>"
            f"<div style='color:#444'>End-to-end {res['wall_ms']:.0f} ms (guest: {html.escape(timing)})</div>"
            f"</div>")


def audit_html(n=8):
    recs = pipeline.audit.records()[-n:]
    ok, bad = pipeline.audit.verify_chain()
    rows = "".join(f"<tr><td>{r['actor']}</td><td>{r['event']}</td><td>{r.get('study_id', '')}</td>"
                   f"<td>{html.escape(json.dumps(r.get('detail', {}))[:70])}</td></tr>" for r in recs)
    head = (f"<div style='font:12px system-ui'>Hash chain: <b>{'intact' if ok else f'BROKEN at line {bad}'}</b>"
            f" · {len(pipeline.audit.records())} events · no pixels, keys or patient ids are logged</div>")
    return head + ("<table style='font:11px monospace;border-collapse:collapse'><tr><th align=left>actor</th>"
                   "<th align=left>event</th><th align=left>study</th><th align=left>detail</th></tr>"
                   + rows + "</table>")


def explanation_html(ex: dict) -> str:
    g = ex.get("granite")
    timing = f" · {ex['granite_ms']:.0f} ms" if ex.get("granite_ms") else ""
    return (f"<div style='border:1px solid #8a8f98;background:#f6f7f9;border-radius:8px;padding:10px 12px;"
            f"color:#1b1d21;font:13px system-ui;margin-top:6px'>"
            f"<div style='font-weight:600'>Why? (for the security team)</div>"
            f"<div style='margin:4px 0'>{html.escape(ex['canonical'])}</div>"
            f"<div style='margin-top:6px;color:#333'><b>Plain-language version</b> "
            f"<span style='font-size:11px;color:#666'>(AI-generated wording by {html.escape(ex.get('model') or 'Granite')}"
            f"{timing}; runs outside the guest and sees only the decision code, never the image)</span><br>"
            f"{html.escape(g) if g else '<i>withheld by the faithfulness check: ' + html.escape(ex.get('guard') or 'not available') + '. Use the explanation above.</i>'}</div></div>")


def load_any_image(name: str, content: bytes) -> np.ndarray:
    if name.lower().endswith(".dcm"):
        import pydicom
        return model.to_fixed_u8(pydicom.dcmread(io.BytesIO(content)).pixel_array)  # header dropped
    if name.lower().endswith(".npy"):
        return model.to_fixed_u8(np.load(io.BytesIO(content)))
    from PIL import Image
    return model.to_fixed_u8(Image.open(io.BytesIO(content)))


def thumb_png(u8: np.ndarray, size=180) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(u8).resize((size, size)).save(b, format="PNG")
    return b.getvalue()


def run_scenario(u8: np.ndarray, scenario: str, client_id: str, source_tag: str):
    """Encrypt + ingest, optionally apply the chosen attack, then score. Returns (res, ct_preview)."""
    st = pipeline.ingest_u8(u8.copy(), source_tag)
    blob_path = config.VAULT / f"{st['study_id']}.enc"
    ct_preview = blob_path.read_bytes()[4:20].hex()
    if scenario == "bad_ct":
        b = bytearray(blob_path.read_bytes()); b[200] ^= 0x01; blob_path.write_bytes(bytes(b))
        return pipeline.score(client_id, st["study_id"]), ct_preview
    if scenario == "bad_model":
        with tempfile.TemporaryDirectory() as td:
            pol = pipeline.current_policy()
            bad = Path(td) / "model.pt"
            d = bytearray(Path(pol["model_file"]).read_bytes()); d[len(d) // 2] ^= 1; bad.write_bytes(bytes(d))
            return pipeline.score(client_id, st["study_id"], model_path=bad), ct_preview
    if scenario == "bad_guest":
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(pipeline.PKG_DIR, Path(td) / "crai", ignore=shutil.ignore_patterns("__pycache__"))
            g = Path(td) / "crai" / "guest.py"
            g.write_text(g.read_text() + "\n# attacker: exfiltrate decrypted pixels\n")
            return pipeline.score(client_id, st["study_id"], guest_pythonpath=td), ct_preview
    return pipeline.score(client_id, st["study_id"]), ct_preview


def build(client_id="hospital-A"):
    import ipywidgets as W
    from IPython.display import display

    samples = sorted(p for p in config.SAMPLES.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".dcm"})
    pick = W.Dropdown(options=[("— choose a local sample —", None)] + [(p.name, str(p)) for p in samples],
                      description="Sample:", layout=W.Layout(width="420px"))
    up = W.FileUpload(accept=".png,.jpg,.jpeg,.dcm,.npy", multiple=False, description="or upload")
    scen = W.Dropdown(options=SCENARIOS, value="normal", description="Scenario:", layout=W.Layout(width="420px"))
    explain_cb = W.Checkbox(value=True, description="Explain decision with Granite (outside the guest)",
                            indent=False, layout=W.Layout(width="420px"))
    explain_box = W.HTML()
    go = W.Button(description="Encrypt & score securely", button_style="primary", icon="lock",
                  layout=W.Layout(width="240px"))
    preview = W.Image(format="png", width=180, height=180, layout=W.Layout(display="none"))
    preview_lbl = W.HTML("<div style='font:11px system-ui;color:#888'>Hospital workstation view, before encryption</div>")
    pipe = W.HTML(pipeline_html({}))
    result = W.HTML()
    raw = W.Output()
    audit = W.HTML(audit_html())
    acc = W.Accordion(children=[raw, audit])
    acc.set_title(0, "Signed response JSON (exactly what left the guest)")
    acc.set_title(1, "Audit log (hash-chained)")
    state = {"u8": None, "tag": None}

    def _set_image(u8, tag):
        state["u8"], state["tag"] = u8, tag
        preview.value = thumb_png(u8)
        preview.layout.display = ""
        pipe.value, result.value, explain_box.value = pipeline_html({}), "", ""

    def on_pick(ch):
        if ch["new"]:
            p = Path(ch["new"])
            _set_image(load_any_image(p.name, p.read_bytes()), "sample:" + p.stem[:24])

    def on_upload(ch):
        v = up.value
        items = list(v.values()) if isinstance(v, dict) else list(v)
        if items:
            it = items[0]
            name = it.get("name") or it.get("metadata", {}).get("name", "upload.png")
            _set_image(load_any_image(name, bytes(it["content"])), "upload")

    def on_go(_):
        if state["u8"] is None:
            result.value = "<b>Choose or upload an image first.</b>"
            return
        go.disabled = True
        try:
            st = {"encrypt": "running"}
            pipe.value = pipeline_html(st)
            states = {"encrypt": "ok", "attestation": "running", "key_release": "running", "inference": "running"}
            pipe.value = pipeline_html(states, {"attestation": "guest process started…"})
            res, ct = run_scenario(state["u8"], scen.value, client_id, state["tag"])
            states = {"encrypt": "ok", **res["stages"]}
            notes = {"encrypt": f"ciphertext {ct[:12]}…"}
            if res.get("refusal"):
                bad = next((k for k, v in states.items() if v == "refused"), None)
                if bad:
                    notes[bad] = res["refusal"]
            pipe.value = pipeline_html(states, notes)
            result.value = result_html(res, ct)
            raw.clear_output()
            with raw:
                print(json.dumps(res["result"] or {"refusal": res["refusal"]}, indent=1))
            audit.value = audit_html()
            if explain_cb.value:
                explain_box.value = "<i>Granite is writing a plain-language explanation…</i>"
                explain_box.value = explanation_html(explainer().explain(explain.event_from_result(res)))
        except Exception as e:
            result.value = f"<b>Error:</b> {html.escape(type(e).__name__)}: {html.escape(str(e))[:300]}"
        finally:
            go.disabled = False

    pick.observe(on_pick, names="value")
    up.observe(on_upload, names="value")
    go.on_click(on_go)
    left = W.VBox([pick, up, scen, explain_cb, go])
    right = W.VBox([preview, preview_lbl])
    display(W.VBox([W.HTML(banner_html()), W.HBox([left, right]), pipe, result, explain_box, acc]))
