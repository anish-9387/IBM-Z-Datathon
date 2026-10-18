"""Paths and settings. Nothing secret lives here: keys are generated at runtime into
CRAI_HOME/broker_store (chmod 600) and never printed or logged."""
import os
from pathlib import Path

# All runtime state lives outside the source package.
HOME = Path(os.environ.get("CRAI_HOME", Path.cwd() / "crai_data")).resolve()
PKG_DIR = Path(__file__).resolve().parent

VAULT = HOME / "vault"              # ciphertext only (host-visible)
BROKER_STORE = HOME / "broker_store"  # KEK, broker/AS signing keys, policy, study registry
MODELS = HOME / "models"            # frozen model files + MANIFEST.json
LOGS = HOME / "logs"                # audit.jsonl (no pixels, no patient ids)
DATA = HOME / "data"                # RSNA download location (never leaves this machine)
CACHE = HOME / "cache"              # resized uint8 npy cache for training/eval
SAMPLES = HOME / "samples"          # demo images selectable in the UI

BROKER_HOST = os.environ.get("CRAI_BROKER_HOST", "127.0.0.1")     # container: service name
BROKER_BIND = os.environ.get("CRAI_BROKER_BIND", "127.0.0.1")     # container: 0.0.0.0 on a private network
BROKER_PORT = int(os.environ.get("CRAI_BROKER_PORT", "8701"))

IMG_SIZE = 512                      # fixed input shape for every study
TOKEN_TTL_S = int(os.environ.get("CRAI_TOKEN_TTL", "60"))     # attestation token lifetime
NONCE_TTL_S = 60
PROB_DECIMALS = 4                   # fixed-width probability in responses
RATE_BURST = int(os.environ.get("CRAI_RATE_BURST", "5"))
RATE_PER_MIN = float(os.environ.get("CRAI_RATE_PER_MIN", "10"))
GUEST_TIMEOUT_S = 300

# Files that make up the "guest image" measurement (simulated). On real IBM Secure
# Execution this would be the measurement of the encrypted SE image, produced by firmware.
GUEST_IMAGE_FILES = ["guest.py", "model.py", "crypto.py", "guard.py", "attestation.py", "config.py", "netblock.py"]

MODE_LABEL = "SIMULATION - no IBM Secure Execution hardware; host exclusion is NOT proven"


def ensure_dirs():
    for d in (VAULT, BROKER_STORE, MODELS, LOGS, DATA, CACHE, SAMPLES):
        d.mkdir(parents=True, exist_ok=True)
    os.chmod(BROKER_STORE, 0o700)
