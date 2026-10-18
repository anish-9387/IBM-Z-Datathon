# Confidential Radiology AI — IBM Z Datathon 2026 (Team Revolution)

Secure chest X-ray pneumonia scoring on IBM LinuxONE. Research prototype, not a diagnostic tool.

## Folder layout
```
confidential-radiology-ai/
  project.ipynb      <- open this and run top to bottom (sections 0-8, plus 4b Granite and 6b Telum)
  README.md          <- this file
  crai/              <- source code (one module per role; see crai/README.md for details)
  deploy/            <- Containerfiles + Podman script (written, NOT run: no Podman on this server)
  crai_data/         <- created at runtime: vault (ciphertext), broker_store (keys, chmod 700),
                        models, logs/audit.jsonl, samples, results.json.  Never commit or share.
```
Outside this folder the notebook only uses `~/granite-4.0-h-350m-base-be.f16.gguf` (Granite model file
that came with the server image) and the Python packages in `/opt/venv`.

## Run
1. Open `confidential-radiology-ai/project.ipynb` in Jupyter.
2. Kernel → Restart and run all cells (about 4–5 minutes on our 3-vCPU z17).
3. Section 7 (RSNA training) needs your own Kaggle token in `~/.kaggle/kaggle.json` (chmod 600);
   it skips cleanly without it.

What is real vs simulated, side-channel scope and data caveats: see `crai/README.md` and the notebook.
