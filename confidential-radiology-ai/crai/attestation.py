"""Attestation evidence (guest side) and attestation tokens (attestation-service side).

SIMULATION: here the guest reports its own code and model hashes. A host-level attacker
can forge that evidence. Hardware-backed evidence collection and verification are not
implemented; these tokens do not establish IBM Secure Execution isolation.
"""
import base64
import hashlib
import json
import time
from pathlib import Path

from . import config, crypto

TOKEN_KEYS = {"iss", "iat", "exp", "guest_image_hash", "model_hash", "x25519_pub", "ed25519_pub", "nonce"}


class TokenError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def measure_guest_image(pkg_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(config.GUEST_IMAGE_FILES):
        data = (Path(pkg_dir) / name).read_bytes()
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def report_data(nonce: str, x25519_pub: str, ed25519_pub: str) -> str:
    return hashlib.sha256(f"{nonce}|{x25519_pub}|{ed25519_pub}".encode()).hexdigest()


def make_evidence(guest_image_hash, model_hash, nonce, x25519_pub, ed25519_pub) -> dict:
    return {"guest_image_hash": guest_image_hash, "model_hash": model_hash, "nonce": nonce,
            "x25519_pub": x25519_pub, "ed25519_pub": ed25519_pub,
            "report_data": report_data(nonce, x25519_pub, ed25519_pub)}


# ------------------------------------------------------------ compact signed token (EdDSA)
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def issue_token(as_key: "crypto.SigningKey", claims: dict, ttl_s: int, now=None) -> str:
    now = int(time.time() if now is None else now)
    body = dict(claims, iss="crai-attestation-service (simulated)", iat=now, exp=now + ttl_s)
    payload = _canon(body)
    return _b64(payload) + "." + _b64(as_key.sign(payload))


def verify_token(token: str, as_pub_hex: str, now=None, check_exp=True) -> dict:
    try:
        p, s = token.split(".")
        payload, sig = _unb64(p), _unb64(s)
        crypto.verify_signature(as_pub_hex, sig, payload)
        claims = json.loads(payload)
    except (ValueError, crypto.InvalidSignature, json.JSONDecodeError):
        raise TokenError("token_invalid")
    if set(claims) != TOKEN_KEYS:
        raise TokenError("token_invalid")
    if check_exp and (time.time() if now is None else now) > claims["exp"]:
        raise TokenError("token_expired")
    return claims
