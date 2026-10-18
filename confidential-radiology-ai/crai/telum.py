"""Telum (on-chip AI accelerator) measurement.

Our frozen PyTorch ResNet-50 is rebuilt in TensorFlow/Keras with the SAME weights and run two ways
on the z17, each in its own fresh process:
  * "nnpa": TensorFlow with IBM's ibm-zdnn-plugin (kernels that call the zDNN library, which uses
            the NNPA instructions of the Telum accelerator)
  * "cpu" : identical TensorFlow, but the plugin's kernel library is hidden from TensorFlow's
            loader for that one process (nothing on disk is changed)
Only measured numbers are reported. This module is NOT part of the guest image: it is a
measurement, the secure guest still runs PyTorch.

Run one mode:  python -m crai.telum --mode nnpa --weights w.npz --inputs x.npy --out r.json
"""
import json
import os
import sys
import time

PLUGIN_SO = "_z_ops.so"


def export_weights(torch_model, path):
    """Save the frozen model's tensors (PyTorch layout) to a .npz the TF side can read."""
    import numpy as np
    sd = {k.replace(".", "/"): v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
          if not k.endswith("num_batches_tracked")}
    np.savez(path, **sd)
    return path


def build_keras(npz_path, size=512):
    """ResNet-50 v1.5 (stride on the 3x3 conv, like torchvision) in Keras, NHWC, with ported weights."""
    import numpy as np
    import tensorflow as tf
    L = tf.keras.layers
    W = np.load(npz_path)
    convs, bns = [], []

    def conv(x, key, filters, k, stride, pad):
        if pad:
            x = L.ZeroPadding2D(pad)(x)
        layer = L.Conv2D(filters, k, strides=stride, padding="valid", use_bias=False)
        convs.append((layer, key))
        return layer(x)

    def bn(x, key):
        layer = L.BatchNormalization(epsilon=1e-5)
        bns.append((layer, key))
        return layer(x)

    inp = L.Input((size, size, 1))
    x = L.ReLU()(bn(conv(inp, "backbone/conv1", 64, 7, 2, 3), "backbone/bn1"))
    x = L.MaxPooling2D(3, 2)(L.ZeroPadding2D(1)(x))        # zero pad is safe after ReLU (x >= 0)
    for li, (planes, blocks, stride) in enumerate([(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)], 1):
        for bi in range(blocks):
            s = stride if bi == 0 else 1
            p = f"backbone/layer{li}/{bi}"
            idt = x
            if bi == 0:
                idt = bn(conv(x, f"{p}/downsample/0", planes * 4, 1, s, 0), f"{p}/downsample/1")
            y = L.ReLU()(bn(conv(x, f"{p}/conv1", planes, 1, 1, 0), f"{p}/bn1"))
            y = L.ReLU()(bn(conv(y, f"{p}/conv2", planes, 3, s, 1), f"{p}/bn2"))
            y = bn(conv(y, f"{p}/conv3", planes * 4, 1, 1, 0), f"{p}/bn3")
            x = L.ReLU()(L.Add()([y, idt]))
    x = L.GlobalAveragePooling2D()(x)
    head = L.Dense(1)
    logit = head(x)
    temp = float(W["temperature"][0])
    out = L.Activation("sigmoid")(L.Rescaling(1.0 / temp)(logit))
    m = tf.keras.Model(inp, out)
    for layer, key in convs:
        layer.set_weights([W[key + "/weight"].transpose(2, 3, 1, 0)])          # OIHW -> HWIO
    for layer, key in bns:
        layer.set_weights([W[key + "/weight"], W[key + "/bias"], W[key + "/running_mean"], W[key + "/running_var"]])
    head.set_weights([W["head/weight"].T, W["head/bias"]])
    return m


def to_input(u8_batch):
    """uint8 (N,512,512) -> float32 NHWC in the TorchXRayVision range [-1024, 1024]."""
    import numpy as np
    return (u8_batch.astype(np.float32) * (2048.0 / 255.0) - 1024.0)[..., None]


def _run(mode, weights, inputs, out_path, runs=7):
    import numpy as np
    import tensorflow as tf
    m = build_keras(weights)
    f = tf.function(lambda t: m(t, training=False))
    xs = to_input(np.load(inputs))
    res = {"mode": mode, "tf": tf.__version__, "n_images": int(len(xs)),
           "plugin_hidden": mode == "cpu", "intra_threads": tf.config.threading.get_intra_op_parallelism_threads()}
    probs = []
    for i in range(0, len(xs), 8):
        probs += f(tf.constant(xs[i:i + 8])).numpy().ravel().tolist()
    res["probs"] = probs
    for bs in (1, 8):
        b = tf.constant(np.repeat(xs[:1], bs, axis=0))
        f(b).numpy()                                        # warm-up / trace
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            f(b).numpy()
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        res[f"batch{bs}"] = {"runs": runs, "median_ms": ts[len(ts) // 2], "min_ms": ts[0], "max_ms": ts[-1],
                             "images_per_s": bs / (ts[len(ts) // 2] / 1000)}
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)


def run_mode(mode, weights, inputs, out_path, timeout=1800):
    """Run one measurement in a fresh process (so the two modes cannot affect each other)."""
    import subprocess
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
    r = subprocess.run([sys.executable, "-m", "crai.telum", "--mode", mode, "--weights", str(weights),
                        "--inputs", str(inputs), "--out", str(out_path)], env=env, capture_output=True,
                       text=True, timeout=timeout, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if r.returncode != 0:
        raise RuntimeError(f"{mode} run failed: {r.stderr[-800:]}")
    with open(out_path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nnpa", "cpu"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "cpu":                       # hide the plugin's kernel library from TF's loader only
        _listdir = os.listdir
        os.listdir = lambda p=".": [n for n in _listdir(p) if n != PLUGIN_SO]
        import tensorflow  # noqa: F401  (TF scans its kernel directory during this import)
        os.listdir = _listdir
    _run(a.mode, a.weights, a.inputs, a.out)
