# Light Engine AI: Low-Memory Image Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Pure NumPy](https://img.shields.io/badge/Framework-Pure%20NumPy-orange.svg)](https://numpy.org/)

An end-to-end, framework-free machine learning pipeline constructed completely from scratch using pure **NumPy**. By eliminating heavy ML runtimes like TensorFlow and OpenCV, **Light Engine AI** achieves a **~91–95% memory footprint reduction** (~52 MB peak RSS vs. ~600 MB standard baseline) while delivering high-accuracy classification and full model telemetry through an interactive web dashboard.

Developed as a B.Tech project at **Delhi Technological University (DTU)** under the supervision of **Prof. Sangita Kansal** and **Ms. Anju Kansal**.

---

## 🔑 Key Features

- **Zero ML Runtimes:** No TensorFlow, PyTorch, or OpenCV dependencies. Pure Python and NumPy array manipulation.
- **1,180-Dimensional Feature Engine:** Extracts 19 signal processing feature families (Gabor filters, HOG, LBP, Hu moments, Fourier spectra, GLCM, Haar wavelets, etc.).
- **From-Scratch Deep Learning:** Custom Multi-Layer Perceptron (MLP) featuring hand-written backpropagation, Adam optimizer, cosine learning rate decay, inverted dropout, and label smoothing.
- **Gradient Verification:** Analytic backpropagation verified via central finite-difference numerical gradient checking (`max rel error < 0.01`).
- **Test-Time Augmentation (TTA):** Robustness evaluation through multi-view ensemble consensus over 7 rotated and flipped variations.
- **Low-Memory Serving:** Lightweight Flask REST API with a peak process RSS of ~52 MB and a steady-state per-image working set of ~53 KB.
- **Interactive UI Dashboard:** Built-in web dashboard displaying real-time predictions, probability distribution charts, model parameters, precision/recall metrics, and confusion matrices.

---

## 🛠️ System Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Browser / Client                     │
└────────────────────────────┬────────────────────────────┘
                             │ POST /api/predict (Multipart)
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    Flask REST API                       │
│  ├── 1. imageio.decode_image (Bilinear resize to 64x64) │
│  ├── 2. features.extract_features (1180-dim vector)     │
│  ├── 3. Scaler.transform (Z-score normalization)        │
│  ├── 4. MLP.predict_proba (Softmax output)              │
│  └── 5. TTA consensus averaging (7 geometric views)     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               Interactive Web Dashboard                 │
│  ├── Predictions & Confidence Tiers                     │
│  ├── Softmax Probabilities & Margin Analysis            │
│  └── Model Metrics, Loss History & Confusion Matrix     │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Extraction Pipeline (1,180 Dimensions)

The feature engine transforms a $64\times64\times3$ image into a fixed 1,180-dimensional descriptor spanning 19 feature family categories:

| Feature Family | Dimension | Description |
| :--- | :---: | :--- |
| `colour` | 19 | Channel statistics, 8-bin hue histogram, saturation/value splits |
| `colour_moments` | 6 | Circular hue mean and variance weighted by saturation |
| `hue_cooc` | 16 | Joint 4x4 co-occurrence histogram of neighboring hue/saturation pairs |
| `edge_direction` | 16 | Magnitude-weighted histogram of edge orientations over $[0, 2\pi)$ |
| `texture_lbp` | 59 | Local Binary Patterns (8-neighbor binary codes) mapped to uniform bins |
| `gabor_bank` | 48 | Filter responses across 4 scales $\times$ 4 orientations Gabor kernels |
| `blob_log` | 4 | Scale-normalized Laplacian-of-Gaussian ($\sigma^2 \nabla^2 G$) peak counts |
| `edges_sobel` | 6 | Sobel $G_x, G_y$ gradient magnitude statistics and angle moments |
| `orientation_hog` | 144 | Histogram of Oriented Gradients ($4\times4$ cell grid $\times$ 9 orientation bins) |
| `edges_canny` | 4 | Non-max suppression along gradient direction with double-thresholding |
| `shape_region` | 21 | Otsu mask properties: centroid, spread, extent, quadrant distribution |
| `hu_moments` | 7 | Seven log-compressed scale, rotation, and translation invariant moments |
| `fourier_rings` | 8 | Radial band energy distribution of 2D Fourier spectra ($|\mathcal{F}(u,v)|^2$) |
| `autocorr` | 4 | Profile autocorrelation $r(\tau)$ along row/column spatial projections |
| `wavelet_haar` | 9 | 3-level Haar wavelet decomposition statistics |
| `glcm_texture` | 4 | Gray-Level Co-occurrence Matrix stats: contrast, energy, entropy, correlation |
| `contrast_glcm` | 5 | Neighborhood absolute-difference statistics across H/V/D directions |
| `layout_blocks` | 32 | $4\times4$ spatial grid block means and standard deviations of luma |
| `thumbnail_rgb` | 768 | Raw downsampled $16\times16\times3$ RGB representation for global layout |

---

## 📈 Performance Benchmarks

Measured on a synthetic 4-class shape benchmark (`circles`, `rings`, `horizontal_bars`, `vertical_bars`) against standard baseline deployments:

| Metric | Light Engine AI | TensorFlow Baseline | OpenCV Baseline | Reduction vs. TF |
| :--- | :---: | :---: | :---: | :---: |
| **Process Peak RSS** | **~52 MB** | ~550–600 MB | ~90–120 MB | **~91%** |
| **Retained Per-Image Set** | **53 KB** | MB-scale tensors | 3–25 MB | **>99%** |
| **Model Checkpoint Size** | **299 KB (.npz)** | 10–100 MB | N/A | **>99%** |
| **Dependency Install Size** | **~51 MB** | ~600 MB+ | ~150 MB | **~92%** |
| **Validation Accuracy** | **100.0%** | — | — | — |

---

## 🚀 Getting Started

### Prerequisites

Python 3.8 or higher installed on your machine.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harshit2k9/LightEngine-AI.git
   cd LightEngine-AI
   ```

2. Install the lightweight dependencies (~51 MB total):
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train Model & Generate Artifacts (Optional):**
   ```bash
   python classifier/dataset.py
   ```

2. **Run Test Suite:**
   ```bash
   python test_classifier.py
   ```

3. **Launch Web Server:**
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`.

---

## 🧪 Testing & Validation

The project includes an automated test suite (`test_classifier.py`) covering:
- Feature vector dimensionality and finite checks
- Bilinear interpolation matrix alignment
- Softmax probability normalisation
- Numerical gradient checking against finite differences
- End-to-end API request/response cycles and error handling

Run all tests via:
```bash
python -m pytest test_classifier.py -q
```

---

## 📜 Repository Layout

```text
LightEngine-AI/
├── app.py                   # Flask server, API endpoints, TTA orchestration
├── classifier/
│   ├── __init__.py          # Package initializer
│   ├── imageio.py           # Stream decoding & bilinear resize (pure NumPy)
│   ├── features.py          # 19 signal processing feature extraction families
│   ├── model.py             # Custom MLP, backpropagation, Adam, gradient check
│   └── dataset.py           # Procedural rendering, scaling & persistence
├── models/
│   ├── model.npz            # Compressed model weights and scaler
│   └── meta.json            # Architecture parameters, training history & metrics
├── templates/
│   └── index.html           # Web application dashboard UI
├── static/
│   ├── app.js               # Dashboard interaction, metrics renderer & sparkline
│   └── style.css            # Dark mode UI styles
├── test_classifier.py       # End-to-end test suite
├── requirements.txt         # Minimal project dependencies
└── README.md                # Project documentation
```

---

## 🎓 Academic Credit & Citation

This project was developed at **Delhi Technological University (DTU)** for the subject *MC103: Python Programming* in partial fulfillment for the degree of **B.Tech in Mathematics & Computing**.

- **Author:** Harshit Kumar (26/B05/011)
- **Department:** Department of Applied Mathematics, Delhi Technological University
- **Supervisors:** Prof. Sangita Kansal & Ms. Anju Kansal

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.