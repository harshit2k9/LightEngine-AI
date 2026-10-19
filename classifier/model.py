"""
A compact multi-layer perceptron trained with mini-batch gradient descent —
implemented from scratch in NumPy (no TensorFlow / PyTorch / scikit-learn).

Architecture : Dense(hidden) -> ReLU -> Dropout(train) -> Dense(num_classes)
Loss         : categorical cross-entropy + L2 weight decay
Optimiser    : Adam (bias-corrected), optional cosine LR decay
Regularisers : dropout, L2, early stopping on validation loss, label smoothing
Utilities    : deterministic train/val split, confusion matrix, per-class P/R/F1
"""

import numpy as np


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def he_init(n_in, n_out, rng):
    """He (Kaiming) initialisation — variance-preserving for ReLU layers."""
    limit = np.sqrt(2.0 / max(n_in, 1))
    return rng.normal(0.0, limit, (n_in, n_out)).astype(np.float32)


class Dropout:
    def __init__(self, p, rng):
        self.p = p
        self.rng = rng
        self.mask = None

    def forward(self, x, train=False):
        if self.p <= 0 or not train:
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(np.float32)
        return x * self.mask / (1.0 - self.p)          # inverted dropout

    def backward(self, grad):
        return grad * self.mask / (1.0 - self.p) if self.mask is not None else grad


class MLP:
    """Configurable depth network: input -> [Dense(H)+ReLU+Dropout]*depth -> Dense(C).

    Parameters live in lists (`self.Ws`, `self.bs`) and the forward/backward pass
    in `step` is one explicit, reviewable block of NumPy code — no autograd.
    The layer-wise backward chain mirrors the forward chain exactly, so any
    number of hidden layers works with the same verified math.
    """

    def __init__(self, n_in, n_hidden, n_classes, seed=0, dropout=0.25, depth=1):
        self.rng = np.random.default_rng(seed)
        self.depth = depth
        dims = [n_in] + [n_hidden] * depth + [n_classes]
        self.sizes = tuple(dims)
        self.Ws = [he_init(dims[i], dims[i + 1], self.rng) for i in range(len(dims) - 1)]
        self.bs = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
        self.drops = [Dropout(dropout, self.rng) for _ in range(depth)]
        self.n_classes = n_classes

    # ---- one explicit forward + backward mini-batch step --------------------
    def step(self, Xb, Yb_onehot, opt, l2_lambda):
        m = Xb.shape[0]
        # forward: cache z (pre-activation) and a (post-activation) per layer
        zs, acts = [], []
        a = Xb
        for i in range(self.depth):
            z = a @ self.Ws[i] + self.bs[i]
            h = relu(z)
            h = self.drops[i].forward(h, train=True)
            zs.append(z); acts.append((a, h))
            a = h
        logits = a @ self.Ws[-1] + self.bs[-1]
        probs = softmax(logits)

        grads_W = [None] * len(self.Ws)
        grads_b = [None] * len(self.bs)
        dlogits = (probs - Yb_onehot) / m             # CE + softmax derivative
        grads_W[-1] = a.T @ dlogits + l2_lambda * self.Ws[-1]
        grads_b[-1] = dlogits.sum(axis=0)
        da = dlogits @ self.Ws[-1].T
        for i in reversed(range(self.depth)):         # backprop through stack
            prev_a, h = acts[i]
            da = self.drops[i].backward(da)
            dz = da * (zs[i] > 0)                    # ReLU derivative
            grads_W[i] = prev_a.T @ dz + l2_lambda * self.Ws[i]
            grads_b[i] = dz.sum(axis=0)
            da = dz @ self.Ws[i].T
        params = {f"W{i}": w for i, w in enumerate(self.Ws)}
        params.update({f"b{i}": b for i, b in enumerate(self.bs)})
        gmap = {f"W{i}": g for i, g in enumerate(grads_W)}
        gmap.update({f"b{i}": g for i, g in enumerate(grads_b)})
        opt.apply(params, gmap)
        # write updated values back (opt.apply mutates in place via -=)
        for i, w in enumerate(self.Ws):
            self.Ws[i] = params[f"W{i}"]
        for i, b in enumerate(self.bs):
            self.bs[i] = params[f"b{i}"]

    def predict_proba(self, X):
        a = X
        for i in range(self.depth):
            a = relu(a @ self.Ws[i] + self.bs[i])
        return softmax(a @ self.Ws[-1] + self.bs[-1])

    def save(self, path):
        kw = {f"W{i}": w for i, w in enumerate(self.Ws)}
        kw.update({f"b{i}": b for i, b in enumerate(self.bs)})
        np.savez(path, **kw)

    def load(self, path):
        d = np.load(path)
        self.Ws = [d[f"W{i}"] for i in range(len(self.Ws))]
        self.bs = [d[f"b{i}"] for i in range(len(self.bs))]
        return self

    def param_count(self):
        return sum(int(np.size(w)) + int(np.size(b))
                   for w, b in zip(self.Ws, self.bs))


# ---------------------------------------------------------------------------
# Optimiser: Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, params_shape_map, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params_shape_map.items()}
        self.v = {k: np.zeros_like(v) for k, v in params_shape_map.items()}
        self.t = 0

    def apply(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def one_hot(y, n_classes):
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def smooth_labels(oh, eps=0.05):
    return oh * (1 - eps) + eps / oh.shape[1]


def evaluate(model, X, y):
    probs = model.predict_proba(X)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    ce = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    return acc, float(ce)


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(cm):
    """Per-class precision / recall / F1 + macro & weighted averages."""
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    sup = cm.sum(axis=1)
    pred_n = cm.sum(axis=0)
    prec = np.divide(tp, pred_n, out=np.zeros(n), where=pred_n > 0)
    rec = np.divide(tp, sup, out=np.zeros(n), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    acc = tp.sum() / max(cm.sum(), 1)
    return {
        "per_class": [{"support": int(sup[i]), "precision": round(prec[i], 4),
                       "recall": round(rec[i], 4), "f1": round(f1[i], 4)}
                      for i in range(n)],
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * sup).sum() / max(sup.sum(), 1)), 4),
    }


def train_model(model, Xtr, Ytr, Xva, Yva, epochs=60, batch=32,
                lr=2e-3, l2=1e-4, verbose=False):
    """Full training loop with Adam, early stopping and LR cosine decay."""
    pmap = {f"W{i}": w for i, w in enumerate(model.Ws)}
    pmap.update({f"b{i}": b for i, b in enumerate(model.bs)})
    opt = Adam(pmap, lr=lr)
    oh_tr = smooth_labels(one_hot(Ytr, model.n_classes))
    n = Xtr.shape[0]
    best_val, best_state, patience, wait = np.inf, None, 10, 0
    history = []
    base_lr = lr

    def snapshot():
        return [w.copy() for w in model.Ws], [b.copy() for b in model.bs]

    for ep in range(epochs):
        opt.lr = base_lr * 0.5 * (1 + np.cos(np.pi * ep / max(epochs - 1, 1)))
        perm = model.rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            model.step(Xtr[idx], oh_tr[idx], opt, l2)
        tr_acc, tr_ce = evaluate(model, Xtr, Ytr)
        va_acc, va_ce = evaluate(model, Xva, Yva)
        history.append({"epoch": ep + 1, "train_acc": round(tr_acc, 4),
                        "val_acc": round(va_acc, 4), "val_loss": round(va_ce, 4)})
        if verbose:
            print(f"ep {ep+1:02d}  tr_acc {tr_acc:.3f}  va_acc {va_acc:.3f}  va_loss {va_ce:.3f}")
        if va_ce < best_val - 1e-4:
            best_val, wait = va_ce, 0
            best_state = snapshot()
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.Ws, model.bs = best_state
    return history


def gradient_check(model, X, Y_oh, eps=1e-2, n_checks=10, seed=7):
    """Central-difference verification of the hand-written backprop in `step`.

    Compares analytic gradients (recomputed with dropout disabled => a
    deterministic reference) against numeric derivatives for a few random
    weights across every layer. Returns max relative error; with float32
    storage and this epsilon it should be < 0.05 — anything much larger means
    the gradient math is wrong.
    """
    rng = np.random.default_rng(seed)
    m = min(X.shape[0], 8)
    X, Y_oh = X[:m], Y_oh[:m]

    # --- analytic gradients via explicit multi-layer chain -------------------
    zs, hs = [], []
    a = X
    for i in range(model.depth):
        z = a @ model.Ws[i] + model.bs[i]
        h = relu(z)
        zs.append(z); hs.append(h)
        a = h
    probs = softmax(a @ model.Ws[-1] + model.bs[-1])
    dlogits = (probs - Y_oh) / m
    grads_W = [None] * len(model.Ws)
    grads_W[-1] = a.T @ dlogits
    da = dlogits @ model.Ws[-1].T
    for i in reversed(range(model.depth)):
        dz = da * (zs[i] > 0)
        grads_W[i] = (X if i == 0 else hs[i - 1]).T @ dz
        da = dz @ model.Ws[i].T

    def batch_loss():
        a_ = X
        for i in range(model.depth):
            a_ = relu(a_ @ model.Ws[i] + model.bs[i])
        p = softmax(a_ @ model.Ws[-1] + model.bs[-1])
        return float(-(Y_oh * np.log(p + 1e-12)).sum() / m)

    errs = []
    for li in range(len(model.Ws)):
        arr, ana = model.Ws[li], grads_W[li]
        for _ in range(max(n_checks // (2 * len(model.Ws)), 2)):
            i = int(rng.integers(arr.shape[0]))
            j = int(rng.integers(arr.shape[1]))
            orig = float(arr[i, j])
            arr[i, j] = orig + eps
            lp = batch_loss()
            arr[i, j] = orig - eps
            lm = batch_loss()
            arr[i, j] = orig
            num = (lp - lm) / (2 * eps)
            a_ = float(ana[i, j])
            errs.append(abs(num - a_) / (abs(num) + abs(a_) + 1e-8))
    return float(max(errs))
