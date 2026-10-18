"""Cryptography for the prototype, on PyCryptodome (builds on s390x without OpenSSL headers;
the 'cryptography' wheel is not available for this LinuxONE image).

  * AES-256-GCM study encryption + decrypt-into-buffer that can be wiped
  * KEK wrap of per-study keys (broker side)
  * X25519 + HKDF-SHA256 transfer of a study key to an attested guest
  * Ed25519 signing keys (attestation tokens, guest responses)

Study file format (vault/<study_id>.enc):  MAGIC(4) | nonce(12) | ciphertext | tag(16)
The study id is bound as AAD, so a ciphertext renamed to another study fails authentication.
Note: PyCryptodome's AES is its own C code; CPACF hardware acceleration is NOT claimed here.
"""
import hashlib
import os

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

MAGIC = b"CRA1"
NONCE_LEN, TAG_LEN = 12, 16
HEADER_LEN = len(MAGIC) + NONCE_LEN


class CiphertextError(Exception):
    """Malformed, truncated or unauthenticated study ciphertext."""


class InvalidSignature(Exception):
    pass


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def random_key() -> bytes:
    return os.urandom(32)


def aad_for(study_id: str) -> bytes:
    return b"crai-study|" + study_id.encode()


def _gcm_encrypt(key: bytes, nonce: bytes, data: bytes, aad: bytes) -> bytes:
    c = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_LEN)
    c.update(aad)
    ct, tag = c.encrypt_and_digest(data)
    return ct + tag


def _gcm_decrypt(key: bytes, nonce: bytes, ct_tag: bytes, aad: bytes) -> bytes:
    c = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_LEN)
    c.update(aad)
    return c.decrypt_and_verify(ct_tag[:-TAG_LEN], ct_tag[-TAG_LEN:])   # ValueError if bad tag


# --------------------------------------------------------------- study encryption
def encrypt_study(pixels: bytes, study_id: str):
    """Returns (blob, dek). Caller hands `dek` straight to the broker for wrapping."""
    dek, nonce = random_key(), os.urandom(NONCE_LEN)
    return MAGIC + nonce + _gcm_encrypt(dek, nonce, pixels, aad_for(study_id)), dek


def wipe(buf: bytearray) -> None:
    """Overwrite a mutable buffer in place (same memory, not a new object)."""
    buf[:] = bytes(len(buf))


def decrypt_into(blob: bytes, dek: bytes, study_id: str, expected_len: int) -> bytearray:
    """Authenticated decryption into a caller-owned buffer that can be wiped afterwards.
    Raises CiphertextError for wrong magic, wrong length, truncation or a bad tag."""
    if len(blob) != HEADER_LEN + expected_len + TAG_LEN or blob[:4] != MAGIC:
        raise CiphertextError("malformed or truncated ciphertext")
    nonce, body, tag = blob[4:HEADER_LEN], blob[HEADER_LEN:-TAG_LEN], blob[-TAG_LEN:]
    buf = bytearray(len(body))
    c = AES.new(dek, AES.MODE_GCM, nonce=nonce, mac_len=TAG_LEN)
    c.update(aad_for(study_id))
    try:
        c.decrypt(body, output=buf)
        c.verify(tag)                     # raises ValueError on a bad tag
    except ValueError as e:
        wipe(buf)                         # overwrite this explicit buffer on authentication failure
        raise CiphertextError("authentication failed (invalid AES-GCM tag)") from e
    return buf


# --------------------------------------------------------------- KEK wrap (broker side)
def wrap_dek(kek: bytes, dek: bytes, study_id: str) -> str:
    n = os.urandom(NONCE_LEN)
    return (n + _gcm_encrypt(kek, n, dek, b"crai-dek|" + study_id.encode())).hex()


def unwrap_dek(kek: bytes, wrapped_hex: str, study_id: str) -> bytes:
    w = bytes.fromhex(wrapped_hex)
    return _gcm_decrypt(kek, w[:NONCE_LEN], w[NONCE_LEN:], b"crai-dek|" + study_id.encode())


# --------------------------------------------------------------- Ed25519 signing keys
class SigningKey:
    def __init__(self, key):
        self._k = key

    @classmethod
    def generate(cls):
        return cls(ECC.generate(curve="ed25519"))

    @classmethod
    def from_seed(cls, seed: bytes):
        return cls(eddsa.import_private_key(seed))

    def seed(self) -> bytes:
        return self._k.seed

    def public_hex(self) -> str:
        return self._k.public_key().export_key(format="raw").hex()

    def sign(self, msg: bytes) -> bytes:
        return eddsa.new(self._k, "rfc8032").sign(msg)


def verify_signature(public_hex: str, sig: bytes, msg: bytes) -> None:
    """Raises InvalidSignature unless `sig` is a valid Ed25519 signature by `public_hex`."""
    try:
        eddsa.new(eddsa.import_public_key(bytes.fromhex(public_hex)), "rfc8032").verify(msg, sig)
    except (ValueError, TypeError) as e:
        raise InvalidSignature(str(e)) from e


# --------------------------------------------------------------- X25519 key transfer
class ExchangeKey:
    """Ephemeral X25519 key. Public part is exchanged as hex(DER SubjectPublicKeyInfo)."""

    def __init__(self, key=None):
        self._k = key or ECC.generate(curve="curve25519")

    def public_hex(self) -> str:
        return self._k.public_key().export_key(format="DER").hex()

    def shared(self, peer_public_hex: str) -> bytes:
        peer = ECC.import_key(bytes.fromhex(peer_public_hex))
        if peer.curve.lower() not in ("curve25519", "x25519"):
            raise ValueError("peer key is not X25519")
        return key_agreement(static_priv=self._k, static_pub=peer, kdf=lambda z: z)


def _kdf(shared: bytes, study_id: str) -> bytes:
    return HKDF(shared, 32, b"", SHA256, context=b"crai-transfer|" + study_id.encode())


def wrap_to_guest(dek: bytes, guest_public_hex: str, study_id: str) -> dict:
    eph = ExchangeKey()
    k = _kdf(eph.shared(guest_public_hex), study_id)
    n = os.urandom(NONCE_LEN)
    return {"epk": eph.public_hex(), "n": n.hex(), "ct": _gcm_encrypt(k, n, dek, study_id.encode()).hex()}


def unwrap_from_broker(wrapped: dict, guest_key: ExchangeKey, study_id: str) -> bytes:
    k = _kdf(guest_key.shared(wrapped["epk"]), study_id)
    return _gcm_decrypt(k, bytes.fromhex(wrapped["n"]), bytes.fromhex(wrapped["ct"]), study_id.encode())
