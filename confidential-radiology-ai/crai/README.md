# Confidential Radiology AI — prototype (IBM Z Datathon 2026, Team Revolution)

Secure chest X-ray pneumonia scoring: the study is encrypted before storage, a separate
attestation service + key broker releases the study key only to a guest whose code and model
hashes match policy, the guest decrypts + scores + wipes, and only a fixed, signed JSON leaves.
**Research prototype, not a diagnostic tool.**

## Run
Open `project.ipynb` (in the `confidential-radiology-ai` folder) and run top to bottom. Runtime state goes to `crai_data/`
(ciphertext vault, broker key store, models, audit log, results) — never commit that folder.

## Components (one file each)
| File | Role | Sees pixels? |
|---|---|---|
| `ui.py` | ipywidgets demo (hospital workstation) | only the image the user picked, before encryption |
| `pipeline.py` | host/PACS: ingest+encrypt, launch guest, verify, rate limit | no (ciphertext only) |
| `broker.py` | separate process: attestation service + key broker, holds KEK | no |
| `guest.py` | separate short-lived process: attest, decrypt, infer, wipe, sign | **yes, only here** |
| `crypto.py` | AES-256-GCM envelope encryption, decrypt into wipe-able buffer, X25519 key transfer, Ed25519 (PyCryptodome) | — |
| `attestation.py` | measurement, evidence, signed expiring tokens | — |
| `guard.py` | fixed response schema, signing, strict validation, verification | — |
| `audit.py` | hash-chained audit log + token-bucket rate limiter | — |
| `model.py` | pure-PyTorch ResNet-50 (no torchvision needed on s390x) | — |
| `data.py`, `train.py` | RSNA patient-level split, cache, head fine-tuning, metrics | training data only |
| `security_tests.py`, `bench.py`, `results.py` | tests, measurements, results store | — |

## Real vs simulated
| Real in this prototype | Simulated / not proven |
|---|---|
| AES-256-GCM (PyCryptodome), per-study keys, AAD binding | IBM Secure Execution: no SE guest on this server; guest is an ordinary process |
| KEK held only by a separate broker process; X25519+HKDF key transfer to the guest | Attestation measurement is self-reported by the guest (on SE: Ultravisor-signed, `pvattest`) |
| Ed25519-signed tokens (60 s expiry) and responses; nonce single-use | HSM / Crypto Express: KEK is a chmod-600 file |
| Strict schema validation on both sides; hash-chained audit; rate limiting | Egress block is an in-process socket guard, not a firewall |
| Real model inference on the IBM z17 CPU | Host exclusion, memory encryption: **not demonstrated** |

## Side-channel scope (honest)
Implemented: fixed 512×512 input shape for every study; fixed-length response (probability is a
6-character string); the guard's work does not depend on the score; AES-GCM and Ed25519 from
PyCryptodome (constant-time behaviour not independently verified here); guest outbound network
blocked except the broker (in-process). **Not claimed:** protection from cache-timing,
speculative-execution, power or memory-remanence attacks; Python cannot guarantee every copy of
the image is erased (the decrypted buffer and input tensor are zeroed; allocator copies may remain).

## Why PyCryptodome
The `cryptography` package has no s390x wheel for this image and cannot be built here (no OpenSSL
headers, no root). PyCryptodome compiles with plain gcc. CPACF hardware acceleration is therefore
**not claimed**; a production build would use OpenSSL/libica (CPACF) or protected keys.

## Data
RSNA Pneumonia Detection Challenge (Kaggle). You must have a Kaggle account, accept the
competition rules, and place your own token on the server (`~/.kaggle/kaggle.json`, chmod 600).
The notebook never asks for or prints it. Note: the TorchXRayVision `resnet50-res512-all` weights
were pre-trained on a mix that includes RSNA ("pc-nih-rsna-siim-vin"), so RSNA test metrics are
optimistic for this backbone — reported with that caveat.
