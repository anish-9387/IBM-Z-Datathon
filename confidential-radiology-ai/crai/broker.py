"""Attestation service + key broker, run as a SEPARATE process:  python -m crai.broker

It loads the KEK and attestation-service signing key from broker_store/ (chmod 600).
These files remain accessible to the same OS user and privileged host processes.
There is no HSM or Trustee integration in this prototype.

Endpoints (plain HTTP; loopback bind by default, configurable):
  GET  /health            -> {"ok": true, "mode": ...}
  GET  /nonce             -> {"nonce"}                      one-time, expires in NONCE_TTL_S
  GET  /as_pubkey         -> {"ed25519_pub"}                for PACS-side verification
  POST /register          {study_id, study_hash, dek}     host ingest -> wraps DEK under KEK
  POST /attest            {evidence}                      -> {"token"} or {"refused": code}
  POST /release           {token, study_id, study_hash}   -> {"wrapped"} or {"refused": code}
"""
import json
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import attestation, config, crypto
from .audit import AuditLog


def _write_secret(path, data: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def load_or_create_keys():
    config.ensure_dirs()
    kek_p, as_p = config.BROKER_STORE / "kek.bin", config.BROKER_STORE / "as_ed25519.key"
    if not kek_p.exists():
        _write_secret(kek_p, os.urandom(32))
    if not as_p.exists():
        _write_secret(as_p, crypto.SigningKey.generate().seed())
    return kek_p.read_bytes(), crypto.SigningKey.from_seed(as_p.read_bytes())


def policy_path():
    return config.BROKER_STORE / "policy.json"


def registry_path():
    return config.BROKER_STORE / "registry.json"


class Broker:
    def __init__(self, ttl_s=None):
        self.kek, self.as_key = load_or_create_keys()
        self.ttl = ttl_s or config.TOKEN_TTL_S
        self.nonces = {}                     # nonce -> expiry
        self.audit = AuditLog()

    # -------- helpers
    def policy(self):
        p = policy_path()
        return json.loads(p.read_text()) if p.exists() else None

    def registry(self):
        p = registry_path()
        return json.loads(p.read_text()) if p.exists() else {}

    def refuse(self, code, study_id=None):
        self.audit.append("broker", "refused", study_id, reason=code)
        return {"refused": code}

    # -------- endpoints
    def nonce(self):
        now = time.time()
        self.nonces = {n: e for n, e in self.nonces.items() if e > now}
        n = secrets.token_hex(32)
        self.nonces[n] = now + config.NONCE_TTL_S
        return {"nonce": n}

    def register(self, req):
        sid, sh, dek = req["study_id"], req["study_hash"], bytes.fromhex(req["dek"])
        reg = self.registry()
        reg[sid] = {"study_hash": sh, "wrapped_dek": crypto.wrap_dek(self.kek, dek, sid),
                    "created": int(time.time())}
        _write_secret(registry_path(), json.dumps(reg, indent=1).encode())
        self.audit.append("broker", "study_registered", sid, study_hash=sh)
        return {"ok": True}

    def attest(self, req):
        ev = req["evidence"]
        pol = self.policy()
        if pol is None:
            return self.refuse("no_policy")
        exp = self.nonces.pop(ev.get("nonce"), None)          # single use
        if exp is None or exp < time.time():
            return self.refuse("nonce_unknown_or_used")
        if ev.get("report_data") != attestation.report_data(ev["nonce"], ev["x25519_pub"], ev["ed25519_pub"]):
            return self.refuse("report_data_mismatch")
        if ev.get("guest_image_hash") != pol["guest_image_hash"]:
            return self.refuse("guest_image_hash_mismatch")
        if ev.get("model_hash") != pol["model_hash"]:
            return self.refuse("model_hash_mismatch")
        claims = {k: ev[k] for k in ("guest_image_hash", "model_hash", "x25519_pub", "ed25519_pub", "nonce")}
        token = attestation.issue_token(self.as_key, claims, self.ttl)
        self.audit.append("attestation_service", "attested", guest_image_hash=ev["guest_image_hash"],
                          model_hash=ev["model_hash"])
        return {"token": token}

    def release(self, req):
        sid = req.get("study_id")
        try:
            claims = attestation.verify_token(req.get("token", ""), self.as_key.public_hex())
        except attestation.TokenError as e:
            return self.refuse(e.code, sid)
        pol = self.policy()
        if pol is None:
            return self.refuse("no_policy", sid)
        if claims["guest_image_hash"] != pol["guest_image_hash"]:   # policy may change after attest
            return self.refuse("guest_image_hash_mismatch", sid)
        if claims["model_hash"] != pol["model_hash"]:
            return self.refuse("model_hash_mismatch", sid)
        rec = self.registry().get(sid)
        if rec is None:
            return self.refuse("unknown_study", sid)
        if req.get("study_hash") != rec["study_hash"]:
            return self.refuse("study_hash_mismatch", sid)
        dek = crypto.unwrap_dek(self.kek, rec["wrapped_dek"], sid)
        wrapped = crypto.wrap_to_guest(dek, claims["x25519_pub"], sid)
        self.audit.append("broker", "key_released", sid, model_hash=claims["model_hash"])
        return {"wrapped": wrapped}


def make_handler(broker: Broker):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send({"ok": True, "mode": config.MODE_LABEL, "token_ttl_s": broker.ttl})
            if self.path == "/nonce":
                return self._send(broker.nonce())
            if self.path == "/as_pubkey":
                return self._send({"ed25519_pub": broker.as_key.public_hex()})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > 64 * 1024:
                return self._send({"error": "too large"}, 413)
            try:
                req = json.loads(self.rfile.read(n))
                route = {"/register": broker.register, "/attest": broker.attest, "/release": broker.release}
                if self.path not in route:
                    return self._send({"error": "not found"}, 404)
                return self._send(route[self.path](req))
            except Exception as e:  # never echo secrets; report the class only
                return self._send({"refused": "internal_error", "error": type(e).__name__}, 400)

        def log_message(self, *a):   # keep stderr quiet; decisions go to the audit log
            pass

    return H


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.BROKER_PORT
    b = Broker()
    b.audit.append("broker", "started", port=port)
    HTTPServer((config.BROKER_BIND, port), make_handler(b)).serve_forever()


if __name__ == "__main__":
    main()
