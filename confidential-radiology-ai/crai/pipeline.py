"""Host / PACS side orchestration: reads plaintext at ingestion, encrypts it, stores
ciphertext, launches the guest and verifies the response. Clear-path evaluation also
runs here. Process separation does not protect against a host administrator.
"""
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import attestation, config, crypto, guard
from .audit import AuditLog, RateLimiter

PKG_DIR = Path(__file__).resolve().parent
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
audit = AuditLog()
limiter = RateLimiter()


def _env(port=None):
    env = dict(os.environ, CRAI_HOME=str(config.HOME), PYTHONPATH=str(PKG_DIR.parent),
               TORCHDYNAMO_DISABLE="1")
    if port:
        env["CRAI_BROKER_PORT"] = str(port)
    return env


def broker_call(path, body=None, port=None, timeout=10):
    url = f"http://{config.BROKER_HOST}:{port or config.BROKER_PORT}{path}"
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def broker_alive(port=None) -> bool:
    try:
        return broker_call("/health", port=port, timeout=2).get("ok", False)
    except OSError:
        return False


def start_broker(port=None, ttl_s=None) -> subprocess.Popen:
    """Start the attestation service + key broker as a separate OS process."""
    port = port or config.BROKER_PORT
    if broker_alive(port):
        return None
    env = _env(port)
    if ttl_s:
        env["CRAI_TOKEN_TTL"] = str(ttl_s)
    proc = subprocess.Popen([sys.executable, "-m", "crai.broker", str(port)], env=env,
                            cwd=str(PKG_DIR.parent), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(100):
        if broker_alive(port):
            return proc
        if proc.poll() is not None:
            raise RuntimeError("broker failed to start: " + proc.stderr.read().decode()[-500:])
        time.sleep(0.1)
    raise RuntimeError("broker did not become healthy")


def as_public_key(port=None) -> str:
    """Attestation-service Ed25519 public key (hex), as published by the broker."""
    return broker_call("/as_pubkey", port=port)["ed25519_pub"]


# ------------------------------------------------------------------ policy (admin action)
def approve_policy(model_path: Path, guest_pkg_dir: Path = PKG_DIR) -> dict:
    model_path = Path(model_path)
    pol = {"model_file": str(model_path.resolve()),
           "model_hash": crypto.sha256_hex(model_path.read_bytes()),
           "guest_image_hash": attestation.measure_guest_image(guest_pkg_dir),
           "approved_at": int(time.time())}
    (config.BROKER_STORE / "policy.json").write_text(json.dumps(pol, indent=1))
    audit.append("admin", "policy_approved", model_hash=pol["model_hash"],
                 guest_image_hash=pol["guest_image_hash"])
    return pol


def current_policy() -> dict:
    p = config.BROKER_STORE / "policy.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ ingest (hospital side)
def _index_path():
    return config.VAULT / "index.json"


def vault_index() -> dict:
    p = _index_path()
    return json.loads(p.read_text()) if p.exists() else {}


def ingest_u8(u8, source_tag: str) -> dict:
    """Encrypt one fixed-shape uint8 study, store ciphertext, hand the DEK to the broker.
    `source_tag` must not contain patient identifiers (e.g. 'rsna-demo', 'public-sample')."""
    assert u8.shape == (config.IMG_SIZE, config.IMG_SIZE) and str(u8.dtype) == "uint8"
    study_id = secrets.token_hex(8)
    blob, dek = crypto.encrypt_study(u8.tobytes(), study_id)
    (config.VAULT / f"{study_id}.enc").write_bytes(blob)
    study_hash = crypto.sha256_hex(blob)
    broker_call("/register", {"study_id": study_id, "study_hash": study_hash, "dek": dek.hex()})
    del dek
    idx = vault_index()
    idx[study_id] = {"study_hash": study_hash, "source": source_tag, "created": int(time.time())}
    _index_path().write_text(json.dumps(idx, indent=1))
    audit.append("host", "ingested", study_id, study_hash=study_hash, note=source_tag)
    return {"study_id": study_id, "study_hash": study_hash}


# ------------------------------------------------------------------ scoring
STAGES = ["rate_limit", "attestation", "key_release", "inference", "output_guard", "verification"]


def score(client_id: str, study_id: str, model_path=None, port=None, guest_pythonpath=None) -> dict:
    """Run one study through the secure path. Returns stage statuses + verified result."""
    res = {"study_id": study_id, "stages": {s: "pending" for s in STAGES}, "result": None,
           "refusal": None, "checks": None, "verified": False, "wall_ms": None, "guest_timing": None}
    if not limiter.allow(client_id):
        res["stages"]["rate_limit"] = "refused"
        audit.append("host", "rate_limited", study_id, client_id=client_id)
        return res
    res["stages"]["rate_limit"] = "ok"
    pol = current_policy()
    model_path = model_path or (pol or {}).get("model_file")
    env = _env(port)
    if guest_pythonpath:
        env["PYTHONPATH"] = str(guest_pythonpath)
    t0 = time.perf_counter()
    # cwd decides which crai package `-m` imports, so it must match the guest code under test
    p = subprocess.run([sys.executable, "-m", "crai.guest", study_id, str(model_path)], env=env,
                       cwd=str(guest_pythonpath or PKG_DIR.parent), capture_output=True, text=True,
                       timeout=config.GUEST_TIMEOUT_S)
    res["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    for line in p.stderr.splitlines():
        try:
            res["guest_timing"] = json.loads(line).get("timing", res["guest_timing"])
        except (json.JSONDecodeError, AttributeError):
            pass
    try:
        kind, obj = guard.parse_guest_output(p.stdout)
    except Exception as e:
        res["stages"]["output_guard"] = "rejected"
        audit.append("host", "output_rejected", study_id, reason=type(e).__name__)
        return res
    if kind == "refusal":
        reason = obj["reason"]
        res["refusal"] = reason
        stage = "inference" if reason == "ciphertext_invalid" else (
            "key_release" if reason in {"study_hash_mismatch", "unknown_study", "token_expired"} else "attestation")
        for s in STAGES[1:STAGES.index(stage)]:
            res["stages"][s] = "ok"
        res["stages"][stage] = "refused"
        audit.append("host", "guest_refused", study_id, reason=reason)
        return res
    for s in ["attestation", "key_release", "inference", "output_guard"]:
        res["stages"][s] = "ok"
    expected = vault_index().get(study_id, {}).get("study_hash")
    checks = guard.verify_envelope(obj, as_public_key(port), (pol or {}).get("model_hash"), expected)
    res["checks"], res["verified"] = checks, all(checks.values())
    res["stages"]["verification"] = "ok" if res["verified"] else "failed"
    res["result"] = obj
    audit.append("host", "result_verified" if res["verified"] else "result_verification_failed",
                 study_id, verified=res["verified"], ms=res["wall_ms"])
    return res


def clear_score(net, u8) -> float:
    """CLEAR PATH (comparison only): same model, same preprocessing, no encryption/attestation."""
    from . import model
    return model.predict_u8(net, u8)
