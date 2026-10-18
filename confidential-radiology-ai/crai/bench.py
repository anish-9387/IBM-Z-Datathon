"""Measured comparisons: secure path vs clear path, latency, ONNX operator inventory.
Every number shown comes from a run on this machine; nothing is filled in by hand."""
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import config, model, pipeline

TOLERANCE = 1e-4   # secure output is rounded to 4 decimals -> max rounding error 5e-5


def secure_vs_clear(net, u8_list, source_tag, client_id="parity-check"):
    """Score the same studies via the clear path (in-process) and the secure path."""
    rows = []
    lim = pipeline.limiter
    for u8 in u8_list:
        clear = pipeline.clear_score(net, u8)
        st = pipeline.ingest_u8(u8.copy(), source_tag)
        lim.state.pop(client_id, None)          # parity run is not a probing client
        r = pipeline.score(client_id, st["study_id"])
        sec = float(r["result"]["payload"]["pneumonia_probability"]) if r["verified"] else None
        rows.append({"study_id": st["study_id"], "clear": clear, "secure": sec, "verified": r["verified"],
                     "wall_ms": r["wall_ms"], **{k: v for k, v in (r["guest_timing"] or {}).items()}})
    ok = [r for r in rows if r["secure"] is not None]
    d = np.array([abs(r["clear"] - r["secure"]) for r in ok]) if ok else np.array([])
    summary = {"studies": len(rows), "verified": len(ok),
               "mean_abs_diff": float(d.mean()) if len(d) else None,
               "max_abs_diff": float(d.max()) if len(d) else None,
               "tolerance": TOLERANCE, "within_tolerance": bool(len(d) and d.max() <= TOLERANCE)}
    return summary, rows


def clear_latency(net, u8, runs=5):
    t = torch.empty(1, 1, config.IMG_SIZE, config.IMG_SIZE)
    with torch.inference_mode():
        net(model.u8_into_tensor(u8, t))        # warm-up
        ts = []
        for _ in range(runs):
            s = time.perf_counter()
            net(model.u8_into_tensor(u8, t))
            ts.append((time.perf_counter() - s) * 1000)
    return {"runs": runs, "median_ms": float(np.median(ts)), "min_ms": float(np.min(ts)),
            "max_ms": float(np.max(ts)), "torch_threads": torch.get_num_threads(),
            "machine": platform.machine()}


def export_onnx(net, path: Path):
    x = torch.zeros(1, 1, config.IMG_SIZE, config.IMG_SIZE)
    kw = dict(opset_version=17, input_names=["image"], output_names=["pneumonia_probability"])
    try:
        torch.onnx.export(net, x, str(path), dynamo=False, **kw)
    except TypeError:                      # older torch without the dynamo flag
        torch.onnx.export(net, x, str(path), **kw)
    return path


def onnx_operator_inventory(path: Path) -> dict:
    import onnx
    g = onnx.load(str(path)).graph
    return dict(Counter(n.op_type for n in g.node).most_common())
