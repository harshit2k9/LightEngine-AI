"""
Synthetic dataset generator + full training pipeline (pure NumPy).

Four visually distinct classes are procedurally rendered at 64x64 so the
project is fully self-contained:

    circles      - filled discs, random colour/size/position
    rings        - annuli (hollow circles)
    horizontal   - parallel bars (stripe texture)
    vertical     - parallel bars rotated 90 deg

Each sample gets light augmentation (jitter, brightness, contrast, noise)
and is split deterministically into train / validation sets. The trained
model plus feature scaler and metadata are persisted to `models/model.npz`.
"""

import json
import os
import time

import numpy as np

from classifier import features as F
from classifier import model as M


# ---------------------------------------------------------------------------
# procedural rendering (NumPy only)
# ---------------------------------------------------------------------------
def _grid(size: int):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return xx, yy


def _flat_bg(bg, size):
    """Broadcast a length-3 background vector to a (size,size,3) canvas."""
    return np.broadcast_to(bg.reshape(1, 1, 3), (size, size, 3)).copy()


def render_circle(rng, size=64):
    xx, yy = _grid(size)
    cx = rng.uniform(0.30, 0.70) * size
    cy = rng.uniform(0.30, 0.70) * size
    r = rng.uniform(0.16, 0.30) * size
    col = rng.uniform(0.15, 0.95, 3)
    bg = np.full(3, rng.uniform(0.80, 1.0))
    img = _flat_bg(bg, size)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    img[mask] = col
    return img.astype(np.float32)


def render_ring(rng, size=64):
    xx, yy = _grid(size)
    cx, cy = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
    r = rng.uniform(0.20, 0.34) * size
    w = rng.uniform(0.05, 0.10) * size
    col = rng.uniform(0.1, 0.6, 3)
    bg = np.full(3, rng.uniform(0.85, 1.0))
    img = _flat_bg(bg, size)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 <= r * r) & (d2 >= (r - w) ** 2)
    img[mask] = col
    return img.astype(np.float32)


def _bars(rng, size=64, vertical=True):
    col = rng.uniform(0.05, 0.45, 3)
    bg = np.full(3, rng.uniform(0.80, 1.0))
    n = int(rng.integers(3, 6))                   # number of stripes
    thick = rng.uniform(0.25, 0.5)                # duty cycle
    phase = rng.uniform(0, 1)
    idx = np.arange(size, dtype=np.float32)
    band = (((idx * n / size) + phase * n) % 1.0) < thick
    img = _flat_bg(bg, size)
    if vertical:
        img[:, band] = col
    else:
        img[band, :] = col
    # slight misalignment so patterns are not perfectly periodic
    shift = int(rng.integers(-2, 3))
    img = np.roll(img, shift, axis=1 if vertical else 0)
    return img.astype(np.float32)


RENDERERS = {
    "circles": render_circle,
    "rings": render_ring,
    "horizontal_bars": lambda rng, size=64: _bars(rng, size, vertical=False),
    "vertical_bars": lambda rng, size=64: _bars(rng, size, vertical=True),
}
CLASSES = list(RENDERERS.keys())


# ---------------------------------------------------------------------------
# augmentation + dataset assembly
# ---------------------------------------------------------------------------
def augment(img, rng, class_idx=None):
    out = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)      # noise
    out = np.clip(out * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05),
                  0, 1)                                            # brightness/contrast
    if class_idx is not None and class_idx < 2:
        k = int(rng.integers(0, 4))          # discs/rings are rotation invariant
        if k:
            out = np.ascontiguousarray(np.rot90(out, k))
    if rng.random() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1])                   # flip LR
    return out.astype(np.float32)


def build_dataset(per_class=200, size=64, seed=0):
    """Render, augment and featurise the whole dataset. Returns X, y, images."""
    rng = np.random.default_rng(seed)
    Xs, ys, imgs = [], [], []
    for ci, name in enumerate(CLASSES):
        fn = RENDERERS[name]
        for _ in range(per_class):
            img = fn(rng, size=size)
            img = augment(img, rng, ci) if rng.random() < 0.9 else img
            Xs.append(F.extract_features(img))
            ys.append(ci)
            imgs.append(img)
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    perm = np.random.default_rng(seed + 1).permutation(len(y))
    return X[perm], y[perm], [imgs[i] for i in perm]


def stratified_split(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        n_val = max(int(round(val_frac * len(idx))), 5)
        va_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    return (np.array(tr_idx), np.array(va_idx))


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.npz")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


class Scaler:
    """Standardisation fitted on training features (mean/std -> unit scale)."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def save(self, z):
        z["sc_mu"], z["sc_sd"] = self.mu, self.sd

    def load(self, z):
        self.mu, self.sd = z["sc_mu"], z["sc_sd"]
        return self


def train_and_save(per_class=400, hidden=192, depth=2, epochs=120, verbose=True):
    t0 = time.time()
    X, y, _ = build_dataset(per_class=per_class)
    tr, va = stratified_split(X, y)
    sc = Scaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])

    net = M.MLP(X.shape[1], hidden, len(CLASSES), seed=42, dropout=0.3, depth=depth)
    hist = M.train_model(net, Xtr, y[tr], Xva, y[va],
                         epochs=epochs, verbose=verbose)

    probs = net.predict_proba(Xva)
    pred = probs.argmax(axis=1)
    cm = M.confusion_matrix(y[va], pred, len(CLASSES))
    rep = M.classification_report(cm)
    gerr = M.gradient_check(net, Xtr[:16], M.one_hot(y[tr][:16], len(CLASSES)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {f"W{i}": w for i, w in enumerate(net.Ws)}
    data.update({f"b{i}": b for i, b in enumerate(net.bs)})
    data["sc_mu"], data["sc_sd"] = sc.mu, sc.sd
    np.savez_compressed(MODEL_PATH, **data)

    meta = {
        "classes": CLASSES,
        "feature_dim": int(X.shape[1]),
        "hidden": hidden,
        "depth": depth,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "params": net.param_count(),
        "train_seconds": round(time.time() - t0, 2),
        "history": hist,
        "val_report": rep,
        "confusion_matrix": cm.tolist(),
        "gradient_check_max_rel_err": gerr,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_artifacts():
    """Return (model, scaler, meta) ready for inference."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        train_and_save(verbose=False)
    with open(META_PATH) as f:
        meta = json.load(f)
    # backward compatibility: older artefacts stored a single-hidden-layer net
    depth = meta.get("depth", 1)
    z = np.load(MODEL_PATH)
    if "W0" in z.files:
        net = M.MLP(meta["feature_dim"], meta["hidden"], len(meta["classes"]),
                    depth=depth)
        net.Ws = [z[f"W{i}"] for i in range(len(net.Ws))]
        net.bs = [z[f"b{i}"] for i in range(len(net.bs))]
    else:                                    # legacy two-matrix checkpoint
        net = M.MLP(meta["feature_dim"], meta["hidden"], len(meta["classes"]),
                    depth=1)
        net.Ws = [z["l1_W"], z["l2_W"]]
        net.bs = [z["l1_b"], z["l2_b"]]
    sc = Scaler()
    sc.mu, sc.sd = z["sc_mu"], z["sc_sd"]
    del z
    return net, sc, meta


if __name__ == "__main__":
    meta = train_and_save(verbose=True)
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))
