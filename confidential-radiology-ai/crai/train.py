"""Fine-tune the pneumonia head on RSNA (backbone frozen), calibrate, evaluate, freeze.

Why head-only: full fine-tuning of a 512x512 ResNet-50 on ~19k images is not practical on a
3-vCPU guest. Features are extracted once (resumable, chunked) and a class-balanced logistic
head is trained on them. Only metrics computed here are reported - nothing is hard-coded.
"""
import time

import numpy as np
import torch

from . import config, data, model

FEAT_DIR = config.CACHE / "features"


def measure_throughput(net, n=8) -> float:
    """Seconds per image for backbone feature extraction on this machine (batch of n)."""
    df = data.load_splits()
    pids = df.patientId.iloc[:n]
    x = torch.empty(n, 1, config.IMG_SIZE, config.IMG_SIZE)
    for i, p in enumerate(pids):
        model.u8_into_tensor(data.load_u8(p), x[i:i + 1])
    with torch.inference_mode():
        net.backbone.features(x[:1])            # warm-up
        t = time.perf_counter()
        net.backbone.features(x)
    return (time.perf_counter() - t) / n


def extract_features(net, pids, name, batch=8, progress=None) -> np.ndarray:
    """Chunked + resumable: re-running continues where it stopped."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pids = list(pids)
    out = []
    for s in range(0, len(pids), 256):
        f = FEAT_DIR / f"{name}_{s:06d}.npy"
        chunk = pids[s:s + 256]
        if not f.exists():
            feats = []
            for b in range(0, len(chunk), batch):
                ids = chunk[b:b + batch]
                x = torch.empty(len(ids), 1, config.IMG_SIZE, config.IMG_SIZE)
                for i, p in enumerate(ids):
                    model.u8_into_tensor(data.load_u8(p), x[i:i + 1])
                with torch.inference_mode():
                    feats.append(net.backbone.features(x).numpy())
            np.save(f, np.concatenate(feats))
        out.append(np.load(f))
        if progress:
            progress(min(s + 256, len(pids)), len(pids))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- metrics
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def bootstrap_auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(auc(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def threshold_for_sensitivity(y, p, target=0.90):
    """Largest threshold whose sensitivity on (validation) data is >= target."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)])


def sens_spec(y, p, thr):
    y, pred = np.asarray(y), np.asarray(p) >= thr
    tp, fn = (pred & (y == 1)).sum(), (~pred & (y == 1)).sum()
    tn, fp = (~pred & (y == 0)).sum(), (pred & (y == 0)).sum()
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def summarize(y, p, thr):
    from sklearn.metrics import brier_score_loss
    lo, hi = bootstrap_auc_ci(y, p)
    se, sp = sens_spec(y, p, thr)
    return {"n": int(len(y)), "positives": int(np.sum(y)), "auc": auc(y, p), "auc_95ci": [lo, hi],
            "threshold": thr, "sensitivity": se, "specificity": sp,
            "brier": float(brier_score_loss(y, p)), "ece_10bin": ece(y, p)}


# ----------------------------------------------------------------------------- head
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_head(Xtr, ytr, Xva, yva):
    """Class-balanced logistic regression (C chosen on val AUC) + Platt calibration on val.
    Returns (w[2048], b) in logit space, already calibrated, plus the chosen C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
        a = auc(yva, lr.decision_function(sc.transform(Xva)))
        if best is None or a > best[0]:
            best = (a, C, lr)
    _, C, lr = best
    # fold the scaler into the linear layer: z = w.(x-mu)/sd + b
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
    # Platt: p = sigmoid(a*z + c) fitted on val (undoes the class-weight prior shift)
    zva = Xva @ w + b
    pl = LogisticRegression(C=1e6, max_iter=5000).fit(zva[:, None], yva)
    a, c = float(pl.coef_[0, 0]), float(pl.intercept_[0])
    return w * a, b * a + c, C


def build_finetuned(xrv_state_path, w, b) -> model.CXRPneumonia:
    m = model.build_zero_shot(xrv_state_path)
    with torch.no_grad():
        m.head.weight.copy_(torch.tensor(w, dtype=torch.float32)[None])
        m.head.bias.copy_(torch.tensor([b], dtype=torch.float32))
    return m.eval()
