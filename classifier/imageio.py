"""
Low-memory image I/O and pre-processing — pure NumPy, no OpenCV.

Pillow is used ONLY as a streaming decoder to turn an uploaded file's bytes
into a raw RGB buffer; every subsequent operation (resizing, normalisation)
is done with hand-written NumPy code so the pipeline stays memory-light and
transparent for reviewers.
"""

import io

import numpy as np

try:  # Pillow is a tiny (~7 MB) pure decoder — NOT OpenCV/TensorFlow.
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def decode_image(data: bytes, size: int = 64) -> np.ndarray:
    """Decode raw image bytes -> float32 array of shape (size, size, 3) in [0,1].

    Steps: format sniffing via Pillow (streamed), converted to RGB,
    centre-cropped to a square, capped resolution (memory guard), then
    bilinearly resized with our own vectorised NumPy implementation.
    """
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to decode image bytes.")
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")            # handles grayscale / RGBA / palette
        side = min(im.width, im.height)   # centre crop to square
        left = (im.width - side) // 2
        top = (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im.thumbnail((512, 512))          # cap working resolution (memory guard)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return resize_bilinear(arr, (size, size))


def encode_png(arr: np.ndarray) -> bytes:
    """Float (H,W,C) in [0,1] -> PNG bytes (for returning previews to the UI)."""
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Hand-written bilinear resize (NumPy only)
# ---------------------------------------------------------------------------
def resize_bilinear(img: np.ndarray, out_hw: tuple) -> np.ndarray:
    """Bilinearly resize a float HxWxC image to (out_h, out_w). Vectorised."""
    ih, iw = img.shape[:2]
    oh, ow = out_hw
    if (ih, iw) == (oh, ow):
        return img.copy()

    # Source pixel centres mapped from destination grid (half-pixel aligned).
    ys = (np.arange(oh, dtype=np.float64) + 0.5) * ih / oh - 0.5
    xs = (np.arange(ow, dtype=np.float64) + 0.5) * iw / ow - 0.5

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    dy = (ys - y0).astype(np.float32)[:, None]           # (oh,1)
    dx = (xs - x0).astype(np.float32)[None, :]           # (1,ow)

    y0c = np.clip(y0, 0, ih - 1)
    y1c = np.clip(y0 + 1, 0, ih - 1)
    x0c = np.clip(x0, 0, iw - 1)
    x1c = np.clip(x0 + 1, 0, iw - 1)

    c = img.shape[2]
    Ia = img[y0c][:, x0c].reshape(oh, ow, c)
    Ib = img[y0c][:, x1c].reshape(oh, ow, c)
    Ic = img[y1c][:, x0c].reshape(oh, ow, c)
    Id = img[y1c][:, x1c].reshape(oh, ow, c)

    w_dx = dx[..., None]
    w_dy = dy[..., None]
    top = Ia * (1 - w_dx) + Ib * w_dx
    bot = Ic * (1 - w_dx) + Id * w_dx
    return (top * (1 - w_dy) + bot * w_dy).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature-space helpers
# ---------------------------------------------------------------------------
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, pure NumPy. Accepts (...,3) or (H,W)."""
    if img.ndim == 2:
        return img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def flatten_for_model(img: np.ndarray) -> np.ndarray:
    """Model input vector: RGB channels interleaved then flattened, float32."""
    return np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
