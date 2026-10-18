"""Plain-language explanations of SECURITY decisions for the hospital's IT/security team.

Two layers:
  1. canonical_explanation(): fixed, hand-written text for every decision code. This is the
     authoritative explanation and is always shown.
  2. GraniteExplainer: IBM Granite 4.0 (350M base, GGUF via llama.cpp on s390x) rewrites those facts
     in plainer words. Its text is labelled as AI-generated wording, and a faithfulness check
     withholds it if it contradicts the facts (a 350M base model sometimes does).

Privacy boundary: Granite runs on the HOST side, outside the trusted guest. It only ever receives
the decision code, the pipeline stage and the names of verification checks - never pixels, keys,
study ids, hashes, patient data or the pneumonia probability. It explains security decisions; it
does not interpret the X-ray or give medical advice.
"""
import os
import re
import time
from pathlib import Path

CANONICAL = {
    "verified": ("The result passed every check: the attestation token is genuine, the answer was signed by the key "
                 "that token approved, the model is the approved one, and the result belongs to the study that was sent.",
                 "The score can be shown to the radiologist as a second opinion."),
    "model_hash_mismatch": ("The guest tried to run a model file whose fingerprint differs from the approved model.",
                            "No study key was released, so the image was never decrypted. Check who changed the model file."),
    "guest_image_hash_mismatch": ("The guest's own code differs from the approved code.",
                                  "No study key was released, so the modified code never saw the image. Treat this as possible tampering."),
    "token_expired": ("The guest's attestation token was older than its 60-second lifetime when it asked for the key.",
                      "No key was released. A new request will attest again."),
    "token_invalid": ("The attestation token was not signed by our attestation service or was malformed.",
                      "No key was released. Treat this as a possible forgery attempt."),
    "nonce_unknown_or_used": ("The attestation evidence reused an old or unknown one-time number.",
                              "This looks like a replay of earlier evidence. No token was issued."),
    "report_data_mismatch": ("The evidence did not bind the guest's public keys to the one-time number.",
                             "Someone may have swapped in their own key. No token was issued."),
    "study_hash_mismatch": ("The encrypted study no longer matches the fingerprint recorded when it was stored.",
                            "No key was released and no analysis ran. The file may have been altered or corrupted."),
    "unknown_study": ("The requested study is not registered with the key broker.", "No key exists to release."),
    "ciphertext_invalid": ("Decryption failed its integrity check (wrong length or authentication tag).",
                           "Any partly decrypted data was wiped immediately. The file is damaged or was tampered with."),
    "rate_limited": ("This client sent more requests than allowed (5 at once, then 10 per minute).",
                     "The request was refused before any processing. Repeated probing is logged."),
    "output_rejected": ("The guest's answer did not match the fixed response format.",
                        "It was discarded. Nothing beyond the fixed fields can leave the guest."),
    "no_policy": ("No approved model and guest code have been registered yet.", "An administrator must approve them first."),
}
ALLOWED_EVENT_KEYS = {"decision", "reason", "stage", "failed_checks"}


def event_from_result(res: dict) -> dict:
    """Reduce a pipeline result to the minimal, non-sensitive event description."""
    st = res.get("stages", {})
    if st.get("rate_limit") == "refused":
        return {"decision": "refused", "reason": "rate_limited", "stage": "rate_limit", "failed_checks": []}
    if res.get("refusal"):
        stage = next((k for k, v in st.items() if v == "refused"), "unknown")
        return {"decision": "refused", "reason": res["refusal"], "stage": stage, "failed_checks": []}
    if st.get("output_guard") == "rejected":
        return {"decision": "refused", "reason": "output_rejected", "stage": "output_guard", "failed_checks": []}
    failed = [k for k, v in (res.get("checks") or {}).items() if not v]
    return {"decision": "accepted" if res.get("verified") else "refused",
            "reason": "verified" if res.get("verified") else "verification_failed",
            "stage": "verification", "failed_checks": failed}


def canonical_explanation(event: dict) -> str:
    what, means = CANONICAL.get(event["reason"], (
        f"The result failed these verification checks: {', '.join(event.get('failed_checks', [])) or 'unknown'}.",
        "The score must not be used."))
    return f"{what} {means}"


def _find_gguf():
    env = os.environ.get("CRAI_GRANITE_GGUF")
    cands = [Path(env)] if env else []
    cands += [Path.cwd() / "granite-4.0-h-350m-base-be.f16.gguf", Path.home() / "granite-4.0-h-350m-base-be.f16.gguf"]
    return next((p for p in cands if p.exists()), None)


FEW_SHOT = """Each technical security note is rewritten in simple words for a hospital manager. The simple version keeps the same meaning and adds nothing.

Technical: The backup server refused the login because the password had expired. No files were opened.
Simple: The backup system turned the login away because its password was too old, so nothing was opened.

Technical: The printer queue rejected the job because the document was larger than the 10 MB limit. Nothing was printed.
Simple: The printer said no because the file was too big, so nothing came out.

Technical: The badge reader denied entry because the badge belonged to a different building. The door stayed locked.
Simple: The door stayed shut because that badge is for another building.

Technical: {facts}
Simple:"""

# Faithfulness guard: a small base model can invert facts ("the key was released"). For a refusal,
# any sentence that says something HAPPENED (released, decrypted, seen, ...) must also contain a
# negation, and the wording must stay close to the facts. Otherwise the text is withheld.
_NEG = re.compile(r"\b(no|not|never|nothing|none|refused|denied|withheld|stayed|without|n't)\b|n't\b", re.I)
_ACTION = re.compile(r"\b(released|decrypted|decrypt|seen|saw|opened|analy[sz]ed|unlocked|readable|ran|shown|used)\b", re.I)
_WORD = re.compile(r"[a-z]{5,}")


def faithfulness(text: str, event: dict, facts: str):
    if not text or len(text) < 15:
        return False, "empty or too short"
    if event["decision"] != "accepted":
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if _ACTION.search(sent) and not _NEG.search(sent):
                return False, "states that an action happened, but the facts say it did not"
    words = set(_WORD.findall(text.lower()))
    overlap = len(words & set(_WORD.findall(facts.lower()))) / max(len(words), 1)
    if overlap < 0.25:
        return False, f"drifted from the facts (word overlap {overlap:.0%})"
    return True, "passed"


class GraniteExplainer:
    def __init__(self, gguf=None, n_threads=None):
        self.path = Path(gguf) if gguf else _find_gguf()
        self.n_threads = n_threads or os.cpu_count()
        self.llm = None
        self.load_ms = None

    @property
    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return self.path is not None

    def _load(self):
        if self.llm is None:
            from llama_cpp import Llama
            t = time.perf_counter()
            self.llm = Llama(model_path=str(self.path), n_ctx=1024, n_threads=self.n_threads, verbose=False)
            self.load_ms = (time.perf_counter() - t) * 1000

    def explain(self, event: dict) -> dict:
        assert set(event) <= ALLOWED_EVENT_KEYS, "only non-sensitive event fields may reach the LLM"
        facts = canonical_explanation(event)
        out = {"canonical": facts, "granite": None, "granite_raw": None, "guard": None, "granite_ms": None, "model": None}
        if not self.available:
            out["granite"] = "(Granite runtime or model file not available)"
            return out
        self._load()
        t = time.perf_counter()
        r = self.llm.create_completion(FEW_SHOT.format(facts=facts), max_tokens=60, temperature=0.0,
                                       stop=["\n", "Technical:"])
        text = re.sub(r"\s+", " ", r["choices"][0]["text"]).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(sentences[:2])[:400]
        ok, why = faithfulness(text, event, facts)
        out["granite_raw"] = text
        out["granite"] = text if ok else None
        out["guard"] = why
        out["granite_ms"] = round((time.perf_counter() - t) * 1000, 1)
        out["model"] = f"IBM Granite 4.0 350M base ({self.path.name}, llama.cpp)"
        return out
