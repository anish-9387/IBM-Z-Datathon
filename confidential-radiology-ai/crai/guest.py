"""Inference subprocess.  python -m crai.guest <study_id> <model_path>

One short-lived process per study: block egress -> measure itself + model -> attest ->
request study key -> decrypt into own buffer -> infer -> wipe -> print ONE guarded line.
stdout carries the guarded envelope/refusal; stderr carries timings and error classes.
Ingestion and clear-path evaluation also see pixels; this process is not hardware-isolated.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

from . import netblock, config

netblock.install(allow=[(config.BROKER_HOST, config.BROKER_PORT)])   # before anything else

import numpy as np  # noqa: E402
from . import attestation, crypto, guard, model  # noqa: E402

BROKER = f"http://{config.BROKER_HOST}:{config.BROKER_PORT}"
PIXELS = config.IMG_SIZE * config.IMG_SIZE
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never use an HTTP proxy


def _call(path, body=None):
    req = urllib.request.Request(BROKER + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=30) as r:
        return json.loads(r.read())


def run(study_id: str, model_path: str):
    t = {"start": time.perf_counter()}
    model_bytes = Path(model_path).read_bytes()             # read ONCE; hash and load these bytes
    model_hash = crypto.sha256_hex(model_bytes)
    image_hash = attestation.measure_guest_image(Path(__file__).resolve().parent)
    xk, sk = crypto.ExchangeKey(), crypto.SigningKey.generate()
    xpub, spub = xk.public_hex(), sk.public_hex()
    t["measure"] = time.perf_counter()

    nonce = _call("/nonce")["nonce"]
    r = _call("/attest", {"evidence": attestation.make_evidence(image_hash, model_hash, nonce, xpub, spub)})
    if "token" not in r:
        return guard.refusal(r.get("refused", "internal_error")), t
    token = r["token"]
    blob = (config.VAULT / f"{study_id}.enc").read_bytes()
    r = _call("/release", {"token": token, "study_id": study_id, "study_hash": crypto.sha256_hex(blob)})
    if "wrapped" not in r:
        return guard.refusal(r.get("refused", "internal_error")), t
    dek = crypto.unwrap_from_broker(r["wrapped"], xk, study_id)
    t["attest_release"] = time.perf_counter()

    net = model.load_frozen_bytes(model_bytes)
    t["model_load"] = time.perf_counter()
    try:
        buf = crypto.decrypt_into(blob, dek, study_id, PIXELS)
    except crypto.CiphertextError:
        return guard.refusal("ciphertext_invalid"), t
    t["decrypt"] = time.perf_counter()
    try:
        u8 = np.frombuffer(buf, dtype=np.uint8, count=PIXELS).reshape(config.IMG_SIZE, config.IMG_SIZE)
        prob = model.predict_u8(net, u8)                    # zeroes its own input tensor
        del u8
    finally:
        crypto.wipe(buf)                                    # overwrite this buffer; other copies may remain
    t["infer"] = time.perf_counter()
    out = guard.sign_envelope(guard.build_payload(crypto.sha256_hex(blob), model_hash, prob), sk, token)
    t["sign"] = time.perf_counter()
    return out, t


def main():
    study_id, model_path = sys.argv[1], sys.argv[2]
    try:
        out, t = run(study_id, model_path)
    except Exception as e:
        out, t = guard.refusal("broker_unreachable" if isinstance(e, OSError) else "internal_error"), {}
        print(json.dumps({"error_class": type(e).__name__}), file=sys.stderr)
    keys = list(t)
    timing = {f"{b}_ms": round((t[b] - t[a]) * 1000, 1) for a, b in zip(keys, keys[1:])}
    print(json.dumps({"timing": timing}), file=sys.stderr)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
