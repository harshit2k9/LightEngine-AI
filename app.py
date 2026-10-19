"""
Flask backend — image upload + NumPy classification API.

Endpoints
---------
GET  /                 frontend (templates/index.html)
POST /api/predict      multipart image upload -> full JSON result
GET  /api/metrics      training/validation metrics for the dashboard
GET  /api/health       liveness + model info

The uploaded file is never written to disk: bytes are read once (capped at
MAX_CONTENT_LENGTH), decoded, featurised and classified in memory.
"""

import base64
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

from classifier import dataset as D
from classifier import features as F
from classifier import imageio as IO
from classifier import model as M

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB upload cap (memory guard)
app.config["ALLOWED_EXTS"] = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

# ---- load model once at startup ---------------------------------------------
NET, SCALER, META = D.load_artifacts()
CLASSES = META["classes"]


def classify_bytes(data: bytes) -> dict:
    """Full inference pipeline for one image's raw bytes."""
    t0 = time.perf_counter()
    rgb = IO.decode_image(data, size=F.IMG_SIZE)          # (64,64,3) float32
    raw_feats = F.extract_features(rgb)                  # hand-crafted descriptor
    X = SCALER.transform(raw_feats[None, :])             # standardise
    probs = NET.predict_proba(X)[0]                      # softmax over classes
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    p_top, p_second = float(probs[top]), float(probs[order[1]])
    margin = p_top - p_second
    if margin > 0.5:
        conf_label = "high"
    elif margin > 0.2:
        conf_label = "medium"
    else:
        conf_label = "low"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    thumb_b64 = base64.b64encode(IO.encode_png(rgb)).decode("ascii")

    return {
        "prediction": CLASSES[top],
        "prediction_index": top,
        "confidence": round(p_top, 4),
        "confidence_level": conf_label,
        "runner_up": CLASSES[int(order[1])],
        "runner_up_probability": round(p_second, 4),
        "decision_margin": round(margin, 4),
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        # --- test-time augmentation: predictions are averaged over the
        #     original + 4 rotations + horizontal flip (7 views total).
        #     This smooths out orientation/scale quirks and measurably
        #     improves accuracy on hard images. ------------------------------
        "tta": tta_predict(rgb),
        "preprocessing": {
            "input_shape_decoded": list(rgb.shape),
            "resized_to": [F.IMG_SIZE, F.IMG_SIZE],
            "normalisation": "float32 in [0,1], then z-scored by train-set scaler",
            "feature_vector_dim": int(raw_feats.shape[0]),
            "entropy_bits": round(float(-(probs * np.log(probs + 1e-12)).sum()), 4),
        },
        "timings_ms": {"end_to_end": round(elapsed_ms, 2)},
        "thumbnail_png_base64": thumb_b64,
    }


def tta_predict(rgb: np.ndarray) -> dict:
    """Average class probabilities over rotated/flipped views of the image."""
    views = [rgb]
    for k in (1, 2, 3):
        views.append(np.ascontiguousarray(np.rot90(rgb, k)))
    views.append(np.ascontiguousarray(rgb[:, ::-1]))
    views.append(np.ascontiguousarray(np.rot90(rgb[:, ::-1], 1)))
    views.append(np.ascontiguousarray(np.rot90(rgb[:, ::-1], 3)))
    stack = np.stack([F.extract_features(v) for v in views])
    probs = NET.predict_proba(SCALER.transform(stack)).mean(axis=0)
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    return {
        "n_views": len(views),
        "prediction": CLASSES[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4)
                          for i in range(len(CLASSES))},
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the multipart form."}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename; choose an image first."}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in app.config["ALLOWED_EXTS"]:
        return jsonify({"error": f"Extension '.{ext}' not allowed. "
                                 f"Use one of {sorted(app.config['ALLOWED_EXTS'])}."}), 400
    data = f.read()
    if len(data) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400
    try:
        result = classify_bytes(data)
    except Exception as exc:               # corrupt / unsupported image bytes
        return jsonify({"error": f"Could not decode image: {exc}"}), 400
    result["filename"] = f.filename
    result["bytes_received"] = len(data)
    return jsonify(result)


@app.route("/api/metrics")
def metrics():
    keep = ["classes", "feature_dim", "hidden", "depth", "n_train", "n_val", "params",
            "train_seconds", "val_report", "confusion_matrix",
            "gradient_check_max_rel_err"]
    out = {k: META[k] for k in keep}
    out["architecture"] = (f"MLP [{META['feature_dim']} -> "
                           + " -> ".join(f"{META['hidden']}" * META.get("depth", 1))
                           + f" -> {len(META['classes'])}] "
                             "(ReLU, dropout 0.30, He init)")
    out["loss"] = "categorical cross-entropy + L2 (1e-4), label smoothing 0.05"
    out["optimiser"] = "Adam (hand-written), cosine LR decay, early stopping"
    out["feature_families"] = [{"name": n, "dim": d} for n, _, d in F.FEATURE_SPECS]
    out["history"] = META["history"]
    return jsonify(out)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "models/model.npz",
                    "classes": CLASSES})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
