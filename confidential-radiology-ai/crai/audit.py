"""Hash-chained, append-only audit log + token-bucket rate limiter.

Each line: {"ts","actor","event","study_id"?,"detail"?,"prev","h"} where h = sha256(line
without h) and prev = previous line's h. Editing or deleting a line breaks the chain.
Never log pixels, keys, tokens or patient identifiers: `detail` is filtered to an allow-list.
"""
import fcntl
import hashlib
import json
import time
from pathlib import Path

from . import config

ALLOWED_DETAIL = {"reason", "model_hash", "guest_image_hash", "study_hash", "client_id", "ms",
                  "verified", "probability_bucket", "port", "note", "n"}
GENESIS = "0" * 64


def _canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


class AuditLog:
    def __init__(self, path: Path = None):
        self.path = Path(path or config.LOGS / "audit.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, actor: str, event: str, study_id: str = None, **detail) -> dict:
        detail = {k: v for k, v in detail.items() if k in ALLOWED_DETAIL}
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)       # broker and host processes share the log
            lines = f.read().splitlines()
            prev = json.loads(lines[-1])["h"] if lines else GENESIS
            rec = {"ts": round(time.time(), 3), "actor": actor, "event": event, "prev": prev}
            if study_id:
                rec["study_id"] = study_id
            if detail:
                rec["detail"] = detail
            rec["h"] = hashlib.sha256(_canon(rec)).hexdigest()
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        return rec

    def records(self):
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def verify_chain(self):
        """Returns (ok, first_bad_line_number_or_None)."""
        prev = GENESIS
        for i, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                rec = json.loads(line)
                h = rec.pop("h")
            except Exception:
                return False, i
            if rec.get("prev") != prev or hashlib.sha256(_canon(rec)).hexdigest() != h:
                return False, i
            prev = h
        return True, None


class RateLimiter:
    """Per-client token bucket: `burst` requests at once, refilled at `per_min` per minute."""

    def __init__(self, burst=None, per_min=None):
        self.burst = burst or config.RATE_BURST
        self.rate = (per_min or config.RATE_PER_MIN) / 60.0
        self.state = {}

    def allow(self, client_id: str, now=None) -> bool:
        now = time.monotonic() if now is None else now
        tokens, last = self.state.get(client_id, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        ok = tokens >= 1.0
        self.state[client_id] = (tokens - 1.0 if ok else tokens, now)
        return ok
