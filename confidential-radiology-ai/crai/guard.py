"""Output guard: the ONLY thing that leaves the guest is a fixed, signed JSON envelope.

envelope = {"payload": {...}, "signature": <128 hex>, "attestation_token": <token>}
payload  = {"schema": "crai.result.v1", "study_hash": <64 hex>, "model_hash": <64 hex>,
            "pneumonia_probability": "0.1234", "issued_at": <int unix s>}

Probability is a fixed-width 6-char string, so every response has the same length whatever
the score (no length side channel). Any extra, missing, or mistyped field is rejected.
"""
import json
import re
import time

from . import attestation, crypto

SCHEMA = "crai.result.v1"
PAYLOAD_KEYS = {"schema", "study_hash", "model_hash", "pneumonia_probability", "issued_at"}
ENVELOPE_KEYS = {"payload", "signature", "attestation_token"}
REFUSAL_KEYS = {"schema", "reason"}
REFUSAL_REASONS = {"model_hash_mismatch", "guest_image_hash_mismatch", "token_expired", "token_invalid",
                   "nonce_unknown_or_used", "report_data_mismatch", "study_hash_mismatch",
                   "unknown_study", "ciphertext_invalid", "no_policy", "broker_unreachable", "internal_error"}
MAX_BYTES = 2048
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX128 = re.compile(r"^[0-9a-f]{128}$")
PROB = re.compile(r"^[01]\.[0-9]{4}$")
TOKEN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


class SchemaError(Exception):
    pass


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def build_payload(study_hash: str, model_hash: str, prob: float, now=None) -> dict:
    p = min(max(float(prob), 0.0), 1.0)
    payload = {"schema": SCHEMA, "study_hash": study_hash, "model_hash": model_hash,
               "pneumonia_probability": f"{p:.4f}", "issued_at": int(time.time() if now is None else now)}
    validate_payload(payload)
    return payload


def sign_envelope(payload: dict, key: "crypto.SigningKey", token: str) -> str:
    validate_payload(payload)
    env = {"payload": payload, "signature": key.sign(canon(payload)).hex(), "attestation_token": token}
    out = canon(env).decode()
    validate_envelope(json.loads(out))
    if len(out) > MAX_BYTES:
        raise SchemaError("oversized response")
    return out


def refusal(reason: str) -> str:
    r = reason if reason in REFUSAL_REASONS else "internal_error"
    return canon({"schema": "crai.refusal.v1", "reason": r}).decode()


# ------------------------------------------------------------------ validation (both sides)
def validate_payload(p) -> dict:
    if not isinstance(p, dict) or set(p) != PAYLOAD_KEYS:
        raise SchemaError(f"payload fields must be exactly {sorted(PAYLOAD_KEYS)}")
    if p["schema"] != SCHEMA:
        raise SchemaError("wrong schema")
    if not (isinstance(p["study_hash"], str) and HEX64.match(p["study_hash"])):
        raise SchemaError("bad study_hash")
    if not (isinstance(p["model_hash"], str) and HEX64.match(p["model_hash"])):
        raise SchemaError("bad model_hash")
    pr = p["pneumonia_probability"]
    if not (isinstance(pr, str) and PROB.match(pr) and float(pr) <= 1.0):
        raise SchemaError("bad pneumonia_probability")
    if not (type(p["issued_at"]) is int and p["issued_at"] > 0):
        raise SchemaError("bad issued_at")
    return p


def validate_envelope(e) -> dict:
    if not isinstance(e, dict) or set(e) != ENVELOPE_KEYS:
        raise SchemaError(f"envelope fields must be exactly {sorted(ENVELOPE_KEYS)}")
    validate_payload(e["payload"])
    if not (isinstance(e["signature"], str) and HEX128.match(e["signature"])):
        raise SchemaError("bad signature field")
    if not (isinstance(e["attestation_token"], str) and TOKEN.match(e["attestation_token"])
            and len(e["attestation_token"]) < 1500):
        raise SchemaError("bad attestation_token field")
    return e


def validate_refusal(r) -> dict:
    if not isinstance(r, dict) or set(r) != REFUSAL_KEYS or r["schema"] != "crai.refusal.v1" \
            or r["reason"] not in REFUSAL_REASONS:
        raise SchemaError("bad refusal")
    return r


def parse_guest_output(text: str):
    """Host-side: strict parse of the guest's single stdout line. Returns ('result'|'refusal', obj)."""
    line = text.strip()
    if len(line.encode()) > MAX_BYTES or "\n" in line:
        raise SchemaError("oversized or multi-line output")
    obj = json.loads(line)
    if isinstance(obj, dict) and obj.get("schema") == "crai.refusal.v1":
        return "refusal", validate_refusal(obj)
    return "result", validate_envelope(obj)


def verify_envelope(env: dict, as_pub_hex: str, approved_model_hash: str,
                    expected_study_hash: str) -> dict:
    """PACS-side verification. Returns a dict of named checks (all must be True)."""
    validate_envelope(env)
    p = env["payload"]
    checks = {"schema": True, "token_signature": False, "issued_after_attestation": False,
              "model_hash_approved": False, "study_hash_matches": False, "payload_signature": False}
    try:
        claims = attestation.verify_token(env["attestation_token"], as_pub_hex, check_exp=False)
        checks["token_signature"] = True
        # result must be produced after attestation, within the guest's max run time
        checks["issued_after_attestation"] = claims["iat"] - 5 <= p["issued_at"] <= claims["iat"] + 600
        checks["model_hash_approved"] = (p["model_hash"] == claims["model_hash"] == approved_model_hash)
        checks["study_hash_matches"] = (p["study_hash"] == expected_study_hash)
        crypto.verify_signature(claims["ed25519_pub"], bytes.fromhex(env["signature"]), canon(p))
        checks["payload_signature"] = True
    except (attestation.TokenError, crypto.InvalidSignature, ValueError):
        pass
    return checks
