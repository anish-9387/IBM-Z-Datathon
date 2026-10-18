# Implementation map

This package implements a process-based research prototype. See the
[root README](../../README.md) for setup, observations and limitations.

| Module | Responsibility |
| --- | --- |
| `config.py` | Runtime directories, broker settings and measured source-file list |
| `ui.py` | Plaintext input/preview, scenarios, statuses and verified results |
| `pipeline.py` | Plaintext ingestion, encryption, guest launch and verification |
| `broker.py` | File-backed keys, policy checks, tokens and wrapped key release |
| `guest.py` | Self-measurement, key request, decryption, CPU inference and signing |
| `crypto.py` | AES-GCM, X25519/HKDF wrapping, Ed25519 and buffer overwrite |
| `attestation.py` | Self-reported measurement and prototype token format |
| `guard.py` | Exact result/refusal schemas and signature verification |
| `audit.py` | Hash-chained file and in-memory token-bucket rate limiter |
| `model.py` | Pure-PyTorch ResNet-50 and image preprocessing |
| `data.py`, `train.py` | RSNA splits, preprocessing, head training and evaluation |
| `security_tests.py` | Pipeline scenarios and isolated component checks |
| `bench.py`, `telum.py` | Parity, timings, ONNX inventory and optional Telum runs |
| `explain.py` | Fixed explanations and optional Granite rewriting with a heuristic guard |
| `results.py` | Caller-supplied results with timestamps and machine labels |

The UI and ingestion code see plaintext. The broker and guest are ordinary processes
on the same host. Their keys remain accessible to the same OS user and privileged host
processes. Self-reported hashes are not hardware attestation.

Guarded stdout carries a signed result or refusal; stderr carries timings and error
classes. Schema validation does not prevent other I/O. The socket guard is not an OS
firewall, and broker RPC is plain HTTP. Never commit `crai_data/`.

See [security scope](../../docs/SECURITY.md) for further limits.
