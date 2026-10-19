"""
Hand-crafted image feature extractors — 100% NumPy (no OpenCV, no scikit-image).

Every function takes a float32 RGB image of shape (H, W, 3) with values in
[0, 1] and returns a fixed-length 1-D float32 descriptor. The concatenation
of all descriptors forms the model input vector.

Feature families
----------------
1.  colour            : mean/std per channel + coarse HSV histogram     (19 dims)
2.  colour moments    : Hue circular mean/var, saturation percentiles   (6 dims)
3.  hue co-occurrence : neighbouring-pixel hue+saturation joint hist    (16 dims)
4.  edge direction    : full-circle histogram of strong-gradient angles (16 dims)
5.  texture LBP       : uniform local binary patterns                   (59 dims)
6.  Gabor filters     : 4 scales x 4 orientations energy/mean/std       (48 dims)
7.  LoG blobs         : Laplacian-of-Gaussian peak count + scale        (4 dims)
8.  edges (Sobel)     : gradient magnitude/density statistics           (6 dims)
9.  HOG               : oriented-gradients on a 4x4 cell grid           (144 dims)
10. Canny-style edges : non-max suppression + hysteresis counts         (4 dims)
11. Hu moments        : 7 log-compressed region moments                 (7 dims)
12. shape/region      : Otsu mask stats, moments, radial profile        (20 dims)
13. Fourier spectrum  : ring-wise spectral energy (periodicity)         (8 dims)
14. autocorrelation   : row/column periodicity peaks                    (4 dims)
15. Haar wavelets     : multi-scale detail energy (3 levels)            (9 dims)
16. GLCM              : co-occurrence contrast/corr/energy/entropy      (4 dims)
17. intensity stats   : neighbour-difference contrast stats             (5 dims)
18. layout blocks     : 4x4 block means/stds                            (32 dims)
19. thumbnail         : 16x16 RGB raw pixels                            (768 dims)
                                                          total = 1180 dims
"""

import math

import numpy as np

IMG_SIZE = 64          # canonical working resolution
FEAT_DIM = 1180        # keep in sync with FEATURE_SPECS below


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _luma(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised HSV. Returns (H,S,V) with H in [0,1), S,V in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn + 1e-8
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    rm = mask & (mx == r)
    gm = mask & (mx == g) & ~rm
    bm = mask & ~rm & ~gm
    h[rm] = ((g - b)[rm] / diff[rm]) % 6.0
    h[gm] = ((b - r)[gm] / diff[gm]) + 2.0
    h[bm] = ((r - g)[bm] / diff[bm]) + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-6, diff / (mx + 1e-8), 0.0).astype(np.float32)
    return np.stack([h, s, mx], axis=-1).astype(np.float32)


def _hist(x: np.ndarray, bins: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Normalised histogram, pure NumPy."""
    idx = np.clip(((x - lo) / (hi - lo + 1e-12) * bins).astype(np.int64), 0, bins - 1)
    h = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
    return h / (h.sum() + 1e-8)


def _conv_same(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """2-D correlation with 'same' padding via edge replication (NumPy only)."""
    kh, kw = kern.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            k = kern[i, j]
            if k != 0:
                out += k * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Colour features
# ---------------------------------------------------------------------------
def color_features(rgb: np.ndarray) -> np.ndarray:
    mean = rgb.reshape(-1, 3).mean(axis=0)                    # 3
    std = rgb.reshape(-1, 3).std(axis=0)                     # 3
    hsv = _rgb_to_hsv(rgb)
    h_hist = _hist(hsv[..., 0], 8)                            # 8 hue bins
    s_mean = np.array([hsv[..., 1].mean()], dtype=np.float32)  # 1
    v_hist = _hist(hsv[..., 2], 2)                            # 2 dark/bright
    sat_hist = _hist(hsv[..., 1], 2)                          # 2 dull/vivid
    return np.concatenate([mean, std, h_hist, s_mean, v_hist, sat_hist])


# ---------------------------------------------------------------------------
# 2. Colour moments: hue circular statistics + saturation distribution
# ---------------------------------------------------------------------------
def color_moments(rgb: np.ndarray) -> np.ndarray:
    """Hue is circular -> use vector (cos,sin) averaging; add saturation spread."""
    hsv = _rgb_to_hsv(rgb)
    h, s = hsv[..., 0], hsv[..., 1]
    wgt = s.ravel()                                    # weight by saturation
    ang = 2 * np.pi * h.ravel()
    cs, sn = np.cos(ang), np.sin(ang)
    W = wgt.sum() + 1e-8
    mean_vec = np.array([(wgt * cs).sum(), (wgt * sn).sum()]) / W
    r_len = float(np.hypot(*mean_vec))                 # 1 = concentrated hue
    mean_hue = (np.arctan2(mean_vec[1], mean_vec[0]) / (2 * np.pi)) % 1.0
    var_hue = 1.0 - r_len                              # circular variance
    return np.asarray([mean_hue, var_hue, r_len,
                       s.mean(), np.percentile(s, 25), np.percentile(s, 75)],
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Hue co-occurrence: joint histogram of neighbouring hue x saturation
#    (captures smooth colour regions vs speckled multi-colour textures)
# ---------------------------------------------------------------------------
def hue_cooc(rgb: np.ndarray, hb: int = 4, sb: int = 4) -> np.ndarray:
    hsv = _rgb_to_hsv(rgb)
    hi = np.clip((hsv[..., 0] * hb).astype(np.int64), 0, hb - 1)
    si = np.clip((hsv[..., 1] * sb).astype(np.int64), 0, sb - 1)
    a = (hi[:, :-1] * sb + si[:, :-1]).ravel()          # horizontal pairs
    b = (hi[:-1, :] * sb + si[:-1, :]).ravel()         # vertical pairs
    code = np.concatenate([a, b])
    h = np.bincount(code, minlength=hb * sb).astype(np.float32)
    return h / (h.sum() + 1e-8)


# ---------------------------------------------------------------------------
# 4. Edge-direction histogram over the FULL circle of strong gradients
#    (bars => energy at 0/180 deg; discs/rings => energy spread evenly)
# ---------------------------------------------------------------------------
def edge_direction_hist(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    g = _luma(rgb)
    gx, gy = sobel(g)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 85) + 1e-8
    strong = mag >= thr
    if not strong.any():
        return np.zeros(bins, dtype=np.float32)
    ang = np.arctan2(gy[strong], gx[strong]) + np.pi   # [0, 2pi)
    idx = np.clip((ang / (2 * np.pi) * bins).astype(np.int64), 0, bins - 1)
    h = np.bincount(idx, weights=mag[strong], minlength=bins).astype(np.float32)
    return h / (h.sum() + 1e-8)


# ---------------------------------------------------------------------------
# 6. Gabor filter bank — 4 scales x 4 orientations (kernels built by hand)
#    Strong frequency/orientation response tells stripes from blobs apart.
# ---------------------------------------------------------------------------
_GABOR_CACHE = {}


def _gabor_kernel(lam: float, theta: float, sigma: float, size: int = 15) -> np.ndarray:
    key = (lam, theta, sigma, size)
    if key in _GABOR_CACHE:
        return _GABOR_CACHE[key]
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    xp = xx * math.cos(theta) + yy * math.sin(theta)
    yp = -xx * math.sin(theta) + yy * math.cos(theta)
    k = np.exp(-(xp ** 2 + (_GABOR_GAMMA ** 2) * yp ** 2) / (2 * sigma ** 2)) \
        * np.cos(2 * np.pi * xp / lam)
    k = k - k.mean()                                   # zero DC
    n = np.linalg.norm(k)
    k /= (n + 1e-8)
    _GABOR_CACHE[key] = k
    return k


_GABOR_GAMMA = 0.5   # ellipticity of the Gabor envelope


def gabor_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    feats = []
    for scale in (3.0, 5.0, 8.0, 13.0):                # stripe period in px
        for ori in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
            k = _gabor_kernel(scale, ori, 0.5 * scale)
            resp = _conv_same(g, k)
            feats.extend([float(np.abs(resp).mean()),
                          float(resp.std()),
                          float((resp ** 2).mean())])
    return np.asarray(feats, dtype=np.float32)         # 4*4*3 = 48 dims


# ---------------------------------------------------------------------------
# 7. Laplacian-of-Gaussian blob detector summary
#    (ring interiors produce a characteristic LoG peak count)
# ---------------------------------------------------------------------------
def log_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    best = None
    for sigma in (1.0, 1.6, 2.4):
        # separable Gaussian blur: shifted-window correlation, vectorised
        t = int(round(3 * sigma)) or 1
        x = np.arange(-t, t + 1, dtype=np.float32)
        ker = np.exp(-x ** 2 / (2 * sigma ** 2))
        ker /= ker.sum()
        h_, w_ = g.shape
        pad_h = np.pad(g, ((0, 0), (t, t)), mode="edge")
        tmp = np.zeros_like(g)
        for i, wv in enumerate(ker):                   # horizontal pass
            tmp += wv * pad_h[:, i:i + w_]
        pad_v = np.pad(tmp, ((t, t), (0, 0)), mode="edge")
        blur = np.zeros_like(g)
        for i, wv in enumerate(ker):                   # vertical pass
            blur += wv * pad_v[i:i + h_, :]
        lap = _conv_same(blur, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                        dtype=np.float32)) * sigma ** 2
        if best is None or lap.std() > best.std():
            best = lap
    lap = best
    mx = np.maximum(lap, 0.0)
    peaks = int((mx > 0.55 * mx.max()).sum())          # local blob responses
    return np.asarray([lap.std(), peaks / 100.0,
                       float(mx.mean()), float(np.percentile(lap, 5))],
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# 10. Canny-style edge counts (non-max suppression + double threshold)
# ---------------------------------------------------------------------------
def canny_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    gx, gy = sobel(g)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    # quantise gradient direction to 4 stripes and suppress non-maxima
    q = (np.round(((ang + np.pi) / (math.pi / 4))) % 4).astype(np.int64)
    m = mag.copy()
    shifts = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)),
              2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    for val, (d1, d2) in shifts.items():
        sel = q == val
        for dy, dx in (d1, d2):
            nb = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)
            m[sel & (mag < nb)] = 0.0
    hi = np.percentile(m, 90)
    lo = 0.4 * hi
    strong = m >= hi
    weak = (m >= lo) & ~strong
    return np.asarray([strong.mean(), weak.mean(),
                       m.mean(), float(hi)], dtype=np.float32)


# ---------------------------------------------------------------------------
# 13. Fourier radial spectrum — periodicity signature (bars -> ring peaks)
# ---------------------------------------------------------------------------
_FFT_R = None


def fourier_features(rgb: np.ndarray) -> np.ndarray:
    """Ring-wise power spectrum energy (DC removed). Periodic stripes concentrate
    energy in narrow mid-frequency rings; blobs spread it radially."""
    global _FFT_R
    g = _luma(rgb)
    P = np.abs(np.fft.fft2(g)) ** 2
    P = np.fft.fftshift(P, axes=(0, 1))
    h, w = P.shape
    if _FFT_R is None or _FFT_R.shape != (h, w):
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _FFT_R = np.hypot(yy - cy, xx - cx)
    P = P.copy()
    P[h // 2, w // 2] = 0.0                            # drop DC
    edges = np.linspace(0, min(h, w) // 2, 9).astype(np.int64)
    out = np.zeros(8, dtype=np.float32)
    tot = P.sum() + 1e-8
    for i in range(8):
        band = (_FFT_R >= edges[i]) & (_FFT_R < edges[i + 1] + 1)
        out[i] = P[band].sum() / tot
    return out


# ---------------------------------------------------------------------------
# 14. Spatial autocorrelation — how periodic are rows/columns?
# ---------------------------------------------------------------------------
def autocorr_features(rgb: np.ndarray, max_lag: int = 24) -> np.ndarray:
    g = _luma(rgb)
    out = []
    for axis in (0, 1):
        prof = g.mean(axis=axis)                       # 1-D signal per line
        prof = prof - prof.mean()
        e = (prof ** 2).sum() + 1e-8
        ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:] / e
        seg = ac[2:max_lag]
        out.extend([float(seg.max()), float(np.argmax(seg) + 2) / max_lag])
    return np.asarray(out, dtype=np.float32)           # 4 dims


# ---------------------------------------------------------------------------
# 16. GLCM texture statistics (horizontal offsets, 16 grey levels)
# ---------------------------------------------------------------------------
def glcm_features(rgb: np.ndarray, levels: int = 16) -> np.ndarray:
    g = _luma(rgb)
    q = np.clip((g * levels).astype(np.int64), 0, levels - 1)
    a, b = q[:, :-1].ravel(), q[:, 1:].ravel()
    c = np.bincount(a * levels + b, minlength=levels * levels).astype(np.float32)
    c = c.reshape(levels, levels)
    c /= (c.sum() + 1e-8)
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    diff = (ii - jj).astype(np.float32)
    contrast = float((c * diff ** 2).sum())
    energy = float((c ** 2).sum())                     # angular second moment
    entropy = float(-(c[c > 0] * np.log2(c[c > 0])).sum())
    mu_i = (c.sum(axis=1) * ii[:, 0]).sum()
    mu_j = (c.sum(axis=0) * jj[0, :]).sum()
    corr = float(((ii - mu_i) * (jj - mu_j) * c).sum()
                 / (levels * levels * c.std() ** 2 + 1e-8))
    return np.asarray([contrast, energy, entropy / 8.0, corr], dtype=np.float32)


# ---------------------------------------------------------------------------
# 5. Texture: Local Binary Pattern (uniform-ish encoding, 59 bins)
# ---------------------------------------------------------------------------
def lbp_histogram(rgb: np.ndarray, n_bins: int = 59) -> np.ndarray:
    g = _luma(rgb)
    c = g[1:-1, 1:-1]
    codes = np.zeros(c.shape, dtype=np.int32)
    shifts = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, :-2],                 g[1:-1, 2:],
        g[2:, :-2],  g[2:, 1:-1],     g[2:, 2:],
    ]
    for bit, nb in enumerate(shifts):
        codes |= (nb >= c).astype(np.int32) << bit            # 8-bit LBP code
    # Map 256 raw codes -> 59 bins: "uniform" patterns get own bin, rest share.
    table = np.zeros(256, dtype=np.int64)
    u = 0
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        trans = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if trans <= 2:
            table[code] = u
            u += 1
        else:
            table[code] = 58                                   # non-uniform bin
    idx = table[codes]
    h = np.bincount(idx.ravel(), minlength=n_bins).astype(np.float32)
    return h / (h.sum() + 1e-8)


# ---------------------------------------------------------------------------
# 3./4. Edges + oriented gradients (Sobel implemented by hand)
# ---------------------------------------------------------------------------
_SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def sobel(g: np.ndarray):
    gx = _conv_same(g, _SX)
    gy = _conv_same(g, _SY)
    return gx, gy


def edge_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    gx, gy = sobel(g)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)                                  # (-pi, pi]
    strong = mag > (mag.mean() + mag.std())
    feats = [
        mag.mean(), mag.std(), np.percentile(mag, 95),
        strong.mean(),                                                # edge density
        ang[strong].mean() if strong.any() else 0.0,
        ang[strong].std() if strong.any() else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def hog_features(rgb: np.ndarray, cells: int = 4, orient_bins: int = 9) -> np.ndarray:
    """Compact histogram-of-oriented-gradients on a cells x cells grid."""
    g = _luma(rgb)
    gx, gy = sobel(g)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi)                        # [0, 2pi)
    h, w = g.shape
    cell_h, cell_w = h // cells, w // cells
    out = np.zeros(cells * cells * orient_bins, dtype=np.float32)
    bi = 0
    for cy in range(cells):
        for cx in range(cells):
            sl = (slice(cy * cell_h, (cy + 1) * cell_h),
                  slice(cx * cell_w, (cx + 1) * cell_w))
            a = ang[sl].ravel()
            m = mag[sl].ravel()
            idx = np.clip((a / (2 * np.pi) * orient_bins).astype(np.int64),
                          0, orient_bins - 1)
            hist = np.bincount(idx, weights=m, minlength=orient_bins).astype(np.float32)
            out[bi * orient_bins:(bi + 1) * orient_bins] = hist / (hist.sum() + 1e-8)
            bi += 1
    return out


# ---------------------------------------------------------------------------
# 5. Shape / region features (Otsu threshold implemented from scratch)
# ---------------------------------------------------------------------------
def _otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 1))
    hist = hist.astype(np.float64) / hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * (np.arange(256) + 0.5) / 256.0)
    mu_t, mu_g = mu[-1], mu
    denom = omega * (1 - omega) + 1e-12
    sigma_b2 = (mu_t - mu_g) ** 2 / denom - (mu_t - mu_g) ** 2  # between-class var
    best = int(np.argmax((mu_t * (1 - omega) - (mu_t - mu)) ** 2 / denom))
    del sigma_b2
    return (best + 0.5) / 256.0


def shape_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    t = _otsu_threshold(g)
    fg = g < t                                                 # dark = object
    if not fg.any():
        fg = g > t                                             # flip if empty
    ys, xs = np.nonzero(fg)
    n = max(int(fg.sum()), 1)
    h, w = g.shape
    cy, cx = ys.mean(), xs.mean()
    area_ratio = fg.mean()
    spread = np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean()) / np.hypot(h, w)
    ext = n / max((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1), 1)
    q = [fg[:h // 2, :w // 2].sum(), fg[:h // 2, w // 2:].sum(),
         fg[h // 2:, :w // 2].sum(), fg[h // 2:, w // 2:].sum()]
    q = np.asarray(q, dtype=np.float32) / (sum(q) + 1e-8)

    # --- central moments -> skewness / kurtosis of the region ---------------
    sp = float(spread) + 1e-8
    dy = (ys - cy) / sp / n
    dx = (xs - cx) / sp / n
    mu20, mu02 = (dx ** 2).sum(), (dy ** 2).sum()
    mu30, mu03 = (dx ** 3).sum(), (dy ** 3).sum()
    mu21, mu12 = (dx ** 2 * dy).sum(), (dx * dy ** 2).sum()
    mu40, mu04 = (dx ** 4).sum(), (dy ** 4).sum()
    skew = np.sqrt(mu20) * mu03 - 3 * np.sqrt(mu02) * mu12      # y-skew proxy
    kurt = (mu04 - 3 * mu02 ** 2) / (mu02 + 1e-8)

    # --- radial profile of the mask (disc vs ring discriminator) ------------
    rad = np.hypot(ys - cy, xs - cx) / (np.hypot(h, w) / 2 + 1e-8)
    prof, _ = np.histogram(rad, bins=9, range=(0, 1))
    prof = prof.astype(np.float32) / (prof.sum() + 1e-8)

    return np.asarray([t, area_ratio, cx / w, cy / h, spread, ext,
                       *q, float(skew), float(kurt),
                       *prof], dtype=np.float32)               # 20 dims


# ---------------------------------------------------------------------------
# 11. Hu moment invariants (log-compressed for stable scaling)
# ---------------------------------------------------------------------------
def hu_moments(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    t = _otsu_threshold(g)
    m_img = (g < t).astype(np.float64)
    if m_img.sum() < 4:
        m_img = (g > t).astype(np.float64)
    h, w = m_img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    m00 = m_img.sum() + 1e-12
    cx = (m_img * xx).sum() / m00
    cy = (m_img * yy).sum() / m00
    eta = lambda p, q_: (m_img * ((xx - cx) ** p) * ((yy - cy) ** q_)).sum() / (m00 ** (1 + (p + q_) / 2.0))
    e20, e02, e11 = eta(2, 0), eta(0, 2), eta(1, 1)
    e30, e03, e12, e21 = eta(3, 0), eta(0, 3), eta(1, 2), eta(2, 1)
    h1 = e20 + e02
    h2 = (e20 - e02) ** 2 + 4 * e11 ** 2
    h3 = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h4 = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h5 = ((e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
          + (3 * e21 - e03) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2))
    h6 = ((e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2)
          + 4 * e11 * (e30 + e12) * (e21 + e03))
    h7 = ((3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
          - (e30 - 3 * e12) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2))
    hs = [h1, h2, h3, h4, h5, h6, h7]
    out = []
    for v in hs:
        sgn = -1.0 if v < 0 else 1.0
        out.append(sgn * math.log10(abs(v) + 1e-12) / 10.0)
    return np.asarray(out, dtype=np.float32)                   # 7 dims


# ---------------------------------------------------------------------------
# 6. Multi-scale Haar-like wavelet energy (Daubechies-1, hand implemented)
# ---------------------------------------------------------------------------
def haar_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    feats = []
    cur = g
    for _level in range(3):
        if min(cur.shape) < 2:
            break
        h_, w_ = cur.shape
        a = cur[:h_ // 2 * 2:2][:, :w_ // 2 * 2][:, ::1]       # even rows
        # average / detail via simple pooling (Haar approximation)
        approx = 0.5 * (cur[0::2, 0::2] + cur[1::2, 0::2] +
                        cur[0::2, 1::2] + cur[1::2, 1::2])
        detail = cur[0::2, 0::2] - cur[1::2, 1::2]
        feats.extend([approx.std(), detail.std(), np.abs(detail).mean()])
        cur = approx
    while len(feats) < 9:
        feats.append(0.0)
    return np.asarray(feats[:9], dtype=np.float32)


# ---------------------------------------------------------------------------
# 7. Contrast / GLCM-style statistics
# ---------------------------------------------------------------------------
def contrast_features(rgb: np.ndarray) -> np.ndarray:
    g = _luma(rgb)
    d_h = np.abs(g[:, :-1] - g[:, 1:])            # horizontal neighbours
    d_v = np.abs(g[:-1, :] - g[1:, :])            # vertical neighbours
    d_d = np.abs(g[:-1, :-1] - g[1:, 1:])         # diagonal neighbours
    p = np.concatenate([d_h.ravel(), d_v.ravel(), d_d.ravel()])
    return np.asarray([p.mean(), p.std(), np.percentile(p, 90),
                      (p > 0.15).mean(), g.max() - g.min()], dtype=np.float32)


# ---------------------------------------------------------------------------
# 8. Spatial layout (block means/variances, 4x4 grid)
# ---------------------------------------------------------------------------
def layout_features(rgb: np.ndarray, grid: int = 4) -> np.ndarray:
    g = _luma(rgb)
    h, w = g.shape
    bh, bw = h // grid, w // grid
    means, vars_ = [], []
    for i in range(grid):
        for j in range(grid):
            blk = g[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            means.append(blk.mean())
            vars_.append(blk.std())
    return np.asarray(means + vars_, dtype=np.float32)


# ---------------------------------------------------------------------------
# 9. Downsampled raw thumbnail (preserves global structure)
# ---------------------------------------------------------------------------
def thumbnail_features(rgb: np.ndarray, size: int = 16) -> np.ndarray:
    h, w, _ = rgb.shape
    step_y, step_x = max(h // size, 1), max(w // size, 1)
    small = rgb[::step_y, ::step_x][:size, :size]
    pad_h = size - small.shape[0]
    pad_w = size - small.shape[1]
    if pad_h or pad_w:
        small = np.pad(small, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return small.reshape(-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Master extractor
# ---------------------------------------------------------------------------
FEATURE_SPECS = [
    ("colour",             color_features,       19),
    ("colour_moments",     color_moments,         6),
    ("hue_cooc",           hue_cooc,             16),
    ("edge_direction",     edge_direction_hist,  16),
    ("texture_lbp",        lbp_histogram,        59),
    ("gabor_bank",         gabor_features,       48),
    ("blob_log",           log_features,          4),
    ("edges_sobel",        edge_features,         6),
    ("orientation_hog",    hog_features,        144),
    ("edges_canny",        canny_features,        4),
    ("shape_region",       shape_features,       21),
    ("hu_moments",         hu_moments,            7),
    ("fourier_rings",      fourier_features,      8),
    ("autocorr",           autocorr_features,     4),
    ("wavelet_haar",       haar_features,         9),
    ("glcm_texture",       glcm_features,         4),
    ("contrast_glcm",      contrast_features,     5),
    ("layout_blocks",      layout_features,      32),
    ("thumbnail_rgb",      thumbnail_features,  768),
]


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Return the concatenated FEAT_DIM-length descriptor for one image."""
    parts = []
    for _, fn, dim in FEATURE_SPECS:
        v = np.asarray(fn(rgb), dtype=np.float32).ravel()
        assert v.shape[0] == dim, f"{fn.__name__}: got {v.shape[0]}, want {dim}"
        parts.append(v)
    f = np.concatenate(parts)
    assert f.shape[0] == FEAT_DIM, f"feature dim {f.shape[0]} != {FEAT_DIM}"
    return f


def feature_names() -> list:
    names = []
    for spec_name, _, dim in FEATURE_SPECS:
        names.extend(f"{spec_name}_{i:03d}" for i in range(dim))
    return names
