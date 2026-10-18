# Confidential Radiology AI - IBM Z Datathon 2026 (Team Revolution)

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
The original server used Python 3.12 and packages in `/opt/venv`. Public weights and
the smoke-test image are downloaded by the notebook; they are not shipped in Git.
The optional Granite model is supplied separately (see below).

## Run
Use a Linux environment. From the repository root:

```bash
cd confidential-radiology-ai
python -m pip install -r requirements.txt
python -m jupyterlab project.ipynb
```

On LinuxONE/s390x, use the server's architecture-compatible Python/PyTorch environment;
the dependency file is not a guarantee that wheels exist for every architecture.
The recorded environment versions are in `../docs/evidence/results.json`.

1. Run notebook sections 0–6 in order, with the notebook working directory set to this
   folder. Keep outbound internet available for the initial model and sample downloads.
2. **Section 6b is optional and hardware-specific:** run it only with TensorFlow and
   `ibm-zdnn-plugin` configured for Telum/NNPA. Otherwise skip that cell manually.
   The secure guest uses PyTorch CPU inference and does not require TensorFlow.
3. Section 4b uses optional `llama-cpp-python` and an architecture-compatible Granite
   GGUF. Set `CRAI_GRANITE_GGUF` to its absolute path before starting Jupyter, or place
   the model in your home directory. Without these, canonical explanations still work.
4. Section 7 needs your own Kaggle access and accepted RSNA competition rules. Place
   the token in `~/.kaggle/kaggle.json` (chmod 600). It skips without credentials/data;
   with access it downloads several GB and training can take substantially longer
   than the smoke demo. Do not redistribute the dataset or token.
5. Section 8 displays the results computed by your run. The original reported results
   are preserved separately in `../docs/evidence/`; they are not loaded as new results.

Runtime state is regenerated under `crai_data/` (or your `CRAI_HOME` override). Use a
fresh runtime directory for the submission; do not restore exposed archived keys.

What is real vs simulated, side-channel scope and data caveats: see `crai/README.md` and the notebook.
