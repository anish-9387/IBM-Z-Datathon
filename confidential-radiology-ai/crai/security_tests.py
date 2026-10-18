"""Security demonstrations using both pipeline requests and isolated component checks.
PASS means the specific expected check held under simulated attestation. These tests do
not prove host isolation. They create runtime studies/logs, including tampered ciphertext."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import attestation, config, crypto, guard, pipeline
from .audit import AuditLog, RateLimiter

PKG_DIR = Path(__file__).resolve().parent


def _row(n, name, attack, expected, observed, ok):
    return {"#": n, "test": name, "attack": attack, "expected": expected, "observed": observed,
            "result": "PASS" if ok else "FAIL"}


def _genuine_token(port=None):
    """Act as an UNTAMPERED guest to obtain a real token (used to isolate later checks)."""
    pol = pipeline.current_policy()
    xk, sk = crypto.ExchangeKey(), crypto.SigningKey.generate()
    nonce = pipeline.broker_call("/nonce", port=port)["nonce"]
    ev = attestation.make_evidence(attestation.measure_guest_image(PKG_DIR), pol["model_hash"], nonce,
                                   xk.public_hex(), sk.public_hex())
    return pipeline.broker_call("/attest", {"evidence": ev}, port=port), xk, sk


def _fresh_study(u8):
    return pipeline.ingest_u8(u8.copy(), "security-test")


def run_all(u8_sample) -> list:
    rows = []
    pol = pipeline.current_policy()

    # 1. wrong model hash: one flipped byte in a copy of the approved model
    st = _fresh_study(u8_sample)
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "tampered_model.pt"
        data = bytearray(Path(pol["model_file"]).read_bytes())
        data[len(data) // 2] ^= 0x01
        bad.write_bytes(bytes(data))
        r = pipeline.score("security-test-1", st["study_id"], model_path=bad)
    rows.append(_row(1, "Wrong model hash", "1 byte flipped in model file",
                     "model_hash_mismatch", r["refusal"], r["refusal"] == "model_hash_mismatch"))

    # 2. wrong guest image: guest code with one added 'leak' line
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(PKG_DIR, Path(td) / "crai", ignore=shutil.ignore_patterns("__pycache__"))
        g = Path(td) / "crai" / "guest.py"
        g.write_text(g.read_text() + "\n# attacker: open('/tmp/leak.bin','wb').write(buf)\n")
        r = pipeline.score("security-test-2", st["study_id"], guest_pythonpath=td)
    rows.append(_row(2, "Wrong guest image hash", "1 line added to guest.py",
                     "guest_image_hash_mismatch", r["refusal"], r["refusal"] == "guest_image_hash_mismatch"))

    # 3. malformed / truncated ciphertext (crypto layer, with the real study key)
    blob, dek = crypto.encrypt_study(u8_sample.tobytes(), "t3")
    n = config.IMG_SIZE ** 2
    try:
        crypto.decrypt_into(blob[:-100], dek, "t3", n)
        obs3 = "decrypted (!)"
    except crypto.CiphertextError as e:
        obs3 = str(e)
    rows.append(_row(3, "Truncated ciphertext", "last 100 bytes removed",
                     "CiphertextError: malformed/truncated", obs3, "truncated" in obs3))

    # 4. invalid AES-GCM tag
    bad = bytearray(blob)
    bad[-1] ^= 0x01
    try:
        buf = crypto.decrypt_into(bytes(bad), dek, "t3", n)
        obs4 = "decrypted (!)"
    except crypto.CiphertextError as e:
        obs4 = str(e)
    rows.append(_row(4, "Invalid AES-GCM tag", "last tag byte flipped",
                     "CiphertextError: invalid tag", obs4, "invalid AES-GCM tag" in obs4))

    # 5. expired attestation token: second broker instance with a 2-second token lifetime
    port = config.BROKER_PORT + 1
    proc = pipeline.start_broker(port=port, ttl_s=2)
    try:
        tok, _, _ = _genuine_token(port)
        time.sleep(3.5)
        r5 = pipeline.broker_call("/release", {"token": tok.get("token", ""), "study_id": st["study_id"],
                                               "study_hash": st["study_hash"]}, port=port)
    finally:
        if proc:
            proc.terminate()
    rows.append(_row(5, "Expired attestation token", "valid token used after its 2 s lifetime",
                     "token_expired", r5.get("refused"), r5.get("refused") == "token_expired"))

    # 6. study-hash mismatch: ciphertext in the vault altered after registration (end to end)
    st6 = _fresh_study(u8_sample)
    p = config.VAULT / f"{st6['study_id']}.enc"
    b = bytearray(p.read_bytes())
    b[100] ^= 0x01
    p.write_bytes(bytes(b))
    r = pipeline.score("security-test-6", st6["study_id"])
    rows.append(_row(6, "Study-hash mismatch", "1 byte of vault ciphertext changed",
                     "study_hash_mismatch", r["refusal"], r["refusal"] == "study_hash_mismatch"))

    # 7. output with an extra field
    ok_r = pipeline.score("security-test-7", st["study_id"])
    env = json.loads(json.dumps(ok_r["result"])) if ok_r["result"] else None
    obs7 = []
    if env:
        for where, mutate in [("envelope", lambda e: e.__setitem__("image_png", "iVBORw0...")),
                              ("payload", lambda e: e["payload"].__setitem__("patient_id", "123"))]:
            e = json.loads(json.dumps(env))
            mutate(e)
            try:
                guard.validate_envelope(e)
                obs7.append(f"{where}: accepted (!)")
            except guard.SchemaError:
                obs7.append(f"{where}: rejected")
    rows.append(_row(7, "Unexpected output field", "extra 'image_png' / 'patient_id' fields",
                     "SchemaError on both", "; ".join(obs7) or "no baseline result",
                     obs7 == ["envelope: rejected", "payload: rejected"]))

    # extra: replayed nonce
    tok, xk, sk = _genuine_token()
    ev_nonce = attestation.verify_token(tok["token"], pipeline.as_public_key(), check_exp=False)["nonce"]
    ev = attestation.make_evidence(attestation.measure_guest_image(PKG_DIR), pol["model_hash"], ev_nonce,
                                   xk.public_hex(), sk.public_hex())
    r = pipeline.broker_call("/attest", {"evidence": ev})
    rows.append(_row("E1", "Replay", "re-send evidence with a used nonce", "nonce_unknown_or_used",
                     r.get("refused"), r.get("refused") == "nonce_unknown_or_used"))

    # extra: rate limit (isolated limiter, burst 5)
    rl = RateLimiter(burst=5, per_min=10)
    allowed = sum(rl.allow("probe", now=0.0 + i * 0.1) for i in range(20))
    rows.append(_row("E2", "Rate limit", "20 requests in 2 s from one client", "5 allowed, 15 refused",
                     f"{allowed} allowed, {20 - allowed} refused", allowed == 5))

    # extra: audit log tampering (on a copy)
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "audit.jsonl"
        shutil.copy(pipeline.audit.path, cp)
        lines = cp.read_text().splitlines()
        k = len(lines) // 2
        rec = json.loads(lines[k]); rec["event"] = "nothing_happened"; lines[k] = json.dumps(rec, sort_keys=True)
        cp.write_text("\n".join(lines) + "\n")
        ok, bad_line = AuditLog(cp).verify_chain()
    rows.append(_row("E3", "Audit log tampering", f"line {k + 1} edited in a copy of the log",
                     f"chain breaks at line {k + 1}", f"ok={ok}, first bad line={bad_line}",
                     (not ok) and bad_line == k + 1))

    # extra: guest egress block (in-process, simulated)
    code = ("from crai import netblock; netblock.install([('127.0.0.1', 8701)]);\n"
            "import socket\ntry:\n socket.create_connection(('pypi.org', 443), timeout=3); print('CONNECTED')\n"
            "except netblock.EgressBlocked: print('BLOCKED')\nexcept Exception as e: print('OTHER', type(e).__name__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=pipeline._env(), cwd=str(PKG_DIR.parent)).stdout.strip()
    rows.append(_row("E4", "Guest egress", "guest code connects to pypi.org:443", "BLOCKED", out, out == "BLOCKED"))
    return rows
