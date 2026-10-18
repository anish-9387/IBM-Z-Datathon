"""RSNA Pneumonia Detection Challenge data: location checks, patient-level split, pixel cache.

Access: the data is on Kaggle (competition 'rsna-pneumonia-detection-challenge'). You need a
Kaggle account, must accept the competition rules on the website, and must place YOUR OWN API
token on this server yourself (~/.kaggle/kaggle.json chmod 600, or ~/.kaggle/access_token, or
KAGGLE_API_TOKEN env var). This code never asks for, prints or stores the token.

Expected layout after download + unzip into CRAI_HOME/data/rsna/:
  stage_2_train_labels.csv        patientId, x, y, width, height, Target  (one row per box;
                                  Target=1 pneumonia/lung opacity, 0 otherwise)
  stage_2_detailed_class_info.csv patientId, class
  stage_2_train_images/<patientId>.dcm   1024x1024 DICOM, one frontal image per patient
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, model

RSNA = config.DATA / "rsna"
LABELS = RSNA / "stage_2_train_labels.csv"
IMAGES = RSNA / "stage_2_train_images"
SPLITS = config.DATA / "splits.csv"
COMPETITION = "rsna-pneumonia-detection-challenge"


def kaggle_credentials_present() -> bool:
    k = Path.home() / ".kaggle"
    return bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
                or (k / "kaggle.json").exists() or (k / "access_token").exists())


def status() -> dict:
    n_dcm = sum(1 for _ in IMAGES.glob("*.dcm")) if IMAGES.exists() else 0
    return {"kaggle_credentials_present": kaggle_credentials_present(),
            "labels_csv": LABELS.exists(), "train_dicoms": n_dcm, "splits_csv": SPLITS.exists(),
            "cached_npy": sum(1 for _ in config.CACHE.glob("*.npy"))}


def download():
    """Runs the official Kaggle CLI. Fails with 401/403 until you accepted the rules and
    placed your token. ~3.5-4 GB download; needs disk space for zip + extracted files."""
    RSNA.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RSNA)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stdout + r.stderr)[-600:]
        raise RuntimeError("Kaggle download failed. Check: rules accepted on kaggle.com, token placed "
                           "on this server, `pip install kaggle`. Output tail:\n" + msg)
    for z in RSNA.glob("*.zip"):
        subprocess.run([sys.executable, "-m", "zipfile", "-e", str(z), str(RSNA)], check=True)
        z.unlink()
    return status()


def patient_table() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    return lab.groupby("patientId", as_index=False)["Target"].max()   # one row per patient


def make_splits(seed=42, n_demo=20) -> pd.DataFrame:
    """Patient-level stratified split ~70/15/15 + n_demo held-out demo studies (from test).
    Frozen to splits.csv; re-running with the same seed reproduces it exactly."""
    from sklearn.model_selection import StratifiedGroupKFold
    pat = patient_table()
    folds = np.zeros(len(pat), int)
    skf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(skf.split(pat, pat.Target, groups=pat.patientId)):
        folds[idx] = k
    pat["split"] = np.select([folds < 14, folds < 17], ["train", "val"], "test")
    demo = pat[pat.split == "test"].groupby("Target", group_keys=False).apply(
        lambda g: g.sample(n_demo // 2, random_state=seed))
    pat.loc[demo.index, "split"] = "demo"
    assert pat.patientId.is_unique
    pat.to_csv(SPLITS, index=False)
    return pat


def load_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS)


def subsample(df, split, n, seed=0) -> pd.DataFrame:
    """Stratified subsample (keeps the class ratio). n=None -> all rows of that split."""
    d = df[df.split == split]
    if n is None or n >= len(d):
        return d
    return d.groupby("Target", group_keys=False).apply(
        lambda g: g.sample(max(1, round(n * len(g) / len(d))), random_state=seed))


def _cache_one(pid):
    out = config.CACHE / f"{pid}.npy"
    if out.exists():
        return
    import pydicom
    px = pydicom.dcmread(IMAGES / f"{pid}.dcm").pixel_array      # header (incl. identifiers) dropped
    np.save(out, model.to_fixed_u8(px))


def cache_images(patient_ids, workers=3):
    from multiprocessing import Pool
    config.CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as p:
        p.map(_cache_one, list(patient_ids), chunksize=32)


def load_u8(pid) -> np.ndarray:
    return np.load(config.CACHE / f"{pid}.npy")
