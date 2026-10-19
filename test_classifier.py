"""
End-to-end test suite for the NumPy image classifier.

Run with:  python -m pytest test_classifier.py -q      (or)  python test_classifier.py
Covers: feature dimensions, resize correctness, softmax/gradient math,
training convergence on synthetic data, and the Flask API happy/error paths.
"""

import io
import json

import numpy as np

from classifier import dataset as D
from classifier import features as F
from classifier import imageio as IO
from classifier import model as M


def test_feature_dimension():
    rng = np.random.default_rng(0)
    f = F.extract_features(rng.random((64, 64, 3), dtype=np.float32))
    assert f.shape == (F.FEAT_DIM,)
    assert sum(d for _, _, d in F.FEATURE_SPECS) == F.FEAT_DIM
    assert len(F.feature_names()) == F.FEAT_DIM
    assert np.isfinite(f).all()


def test_resize_preserves_constants_and_shape():
    flat = np.full((137, 91, 3), 0.5, dtype=np.float32)
    out = IO.resize_bilinear(flat, (64, 64))
    assert out.shape == (64, 64, 3)
    assert np.allclose(out, 0.5, atol=1e-6)          # constant stays constant
    ramp = np.tile(np.linspace(0, 1, 200, dtype=np.float32), (200, 1))[..., None]
    ramp = np.repeat(ramp, 3, axis=2)
    small = IO.resize_bilinear(ramp, (50, 50))
    assert np.allclose(small[0, :, 0], np.linspace(small[0, 0, 0], small[0, -1, 0], 50),
                       atol=0.05)                    # linear map stays ~linear


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(3).normal(size=(16, 7))
    p = M.softmax(z.astype(np.float32))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_gradient_check_passes():
    net = M.MLP(F.FEAT_DIM, 24, 4, seed=5)
    X = np.random.default_rng(4).random((8, F.FEAT_DIM)).astype(np.float32)
    Y = M.one_hot(np.array([0, 1, 2, 3, 0, 1, 2, 3]), 4)
    err = M.gradient_check(net, X, Y)
    assert err < 0.05, f"backprop mismatch: {err}"    # float32 noise floor


def test_training_converges_and_persists(tmp_path=None):
    meta = D.train_and_save(per_class=60, hidden=64, epochs=40, verbose=False)
    rep = meta["val_report"]
    assert rep["accuracy"] > 0.9, rep
    assert len(meta["history"]) >= 5                  # early stopping not too eager
    net, sc, m2 = D.load_artifacts()
    assert m2["classes"] == D.CLASSES
    img = D.RENDERERS["rings"](np.random.default_rng(11), size=64)
    x = sc.transform(F.extract_features(img)[None, :])
    assert int(net.predict_proba(x).argmax()) == D.CLASSES.index("rings")


def test_flask_api_end_to_end():
    import app as A
    client = A.app.test_client()

    # health + metrics
    assert client.get("/api/health").get_json()["status"] == "ok"
    mets = client.get("/api/metrics").get_json()
    assert mets["val_report"]["accuracy"] > 0.9

    # frontend renders
    html = client.get("/").get_data(as_text=True)
    assert "NumPy Image Classifier" in html and "/api/predict" not in html.split("<form")[0]

    # a real prediction from rendered PNG bytes
    from PIL import Image
    img = D.RENDERERS["circles"](np.random.default_rng(21), size=64)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    resp = client.post("/api/predict",
                       content_type="multipart/form-data",
                       data={"file": (io.BytesIO(buf.getvalue()), "c.png")})
    assert resp.status_code == 200
    j = resp.get_json()
    assert j["prediction"] == "circles"
    assert abs(sum(j["probabilities"].values()) - 1.0) < 1e-3
    assert j["preprocessing"]["feature_vector_dim"] == F.FEAT_DIM

    # error paths
    bad = client.post("/api/predict", content_type="multipart/form-data",
                      data={"file": (io.BytesIO(b"junk"), "x.png")})
    assert bad.status_code == 400 and "decode" in bad.get_json()["error"].lower()
    ext = client.post("/api/predict", content_type="multipart/form-data",
                      data={"file": (io.BytesIO(b"MZ"), "evil.exe")})
    assert ext.status_code == 400
    empty = client.post("/api/predict", data={})
    assert empty.status_code == 400


if __name__ == "__main__":
    for fn in [test_feature_dimension, test_resize_preserves_constants_and_shape,
               test_softmax_rows_sum_to_one, test_gradient_check_passes,
               test_training_converges_and_persists, test_flask_api_end_to_end]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("All tests passed.")
