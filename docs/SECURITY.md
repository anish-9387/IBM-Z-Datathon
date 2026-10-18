# Security scope

Use this prototype for local research with public or synthetic samples. It is not a
clinical service, a compliance certification, or a hardened deployment.

| Mechanism | Limit |
| --- | --- |
| AES-256-GCM study encryption | Ingestion sees plaintext and the data key |
| Code/model hash policy | Self-reported evidence can be forged by a malicious host |
| Signed tokens and payloads | Keys are local files/process memory, not HSM-protected |
| Exact JSON schema | Constrains the checked response, not arbitrary process I/O |
| Buffer overwrite | Other copies can remain in Python, libraries or the allocator |
| Hash-chained file | Full rewrites/tail truncation need an external checkpoint to detect |
| Token-bucket limiter | In-memory, per process, caller-supplied identity |
| Socket guard | In-process monkey patch, not OS-enforced isolation |
| Plain HTTP broker | No TLS or authenticated client transport; do not expose it to untrusted networks |
| Granite wording heuristic | Can miss incorrect text; fixed explanations remain the reference |

The implementation does not verify IBM Secure Execution evidence, use Crypto Express,
or establish CPACF offload. Containerfiles are deployment sketches, not a tested
production deployment or proof of hardware isolation.

## Data and model scope

Recorded parity and Telum evidence use one public NIH smoke-test image. RSNA evaluation
code is present, but accuracy and calibration results are absent from the snapshot.
The backbone's described pretraining mix includes RSNA. Scores are not diagnoses.
