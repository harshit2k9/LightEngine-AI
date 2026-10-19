/* Frontend logic: upload -> /api/predict -> render results; metrics dashboard. */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- file selection ---------------- */
const dz = $("dropzone"), fileInput = $("file"), preview = $("preview"),
      btn = $("predict-btn"), errBox = $("error"), resetBtn = $("reset-btn");

dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter") fileInput.click(); });
["dragover", "dragenter"].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(f) {
  errBox.hidden = true;
  const okExt = /\.(png|jpe?g|bmp|gif|webp)$/i.test(f.name);
  if (!okExt) return showError(`Unsupported file type: ${f.name}`);
  if (f.size > 8 * 1024 * 1024) return showError("File exceeds the 8 MB limit.");
  // move into the input so FormData picks it up
  const dt = new DataTransfer();
  dt.items.add(f);
  fileInput.files = dt.files;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  $("dz-text").hidden = true;
  btn.disabled = false;
  resetBtn.hidden = false;
}

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  preview.hidden = true; preview.src = "";
  $("dz-text").hidden = false;
  btn.disabled = true;
  resetBtn.hidden = true;
  $("result-card").hidden = true;
  errBox.hidden = true;
});

function showError(msg) {
  errBox.textContent = msg;
  errBox.hidden = false;
  $("result-card").hidden = true;
}

/* ---------------- prediction ---------------- */
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;
  btn.disabled = true; btn.textContent = "Classifying…";
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const json = await res.json();
    if (!res.ok) return showError(json.error || `HTTP ${res.status}`);
    renderResult(json);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    btn.disabled = false; btn.textContent = "Classify →";
  }
});

function renderResult(r) {
  errBox.hidden = true;
  $("pred-name").textContent = r.prediction.replace(/_/g, " ");
  $("pred-conf").textContent = (r.confidence * 100).toFixed(1) + "%";
  const badge = $("conf-badge");
  badge.textContent = r.confidence_level.toUpperCase();
  badge.className = "badge " + r.confidence_level;
  $("runner-up").textContent =
    `${r.runner_up.replace(/_/g, " ")} (${(r.runner_up_probability * 100).toFixed(1)}%) · margin ${(r.decision_margin * 100).toFixed(1)}%`;
  $("thumb").src = "data:image/png;base64," + r.thumbnail_png_base64;

  /* test-time augmentation summary — averaged over rotated/flipped views */
  const tta = r.tta;
  const note = $("tta-note");
  if (tta) {
    const agrees = tta.prediction === r.prediction;
    note.innerHTML =
      `<span class="tta-chip ${agrees ? "agree" : "disagree"}">` +
      `${agrees ? "✓" : "⚠"} Augmented vote (${tta.n_views} views): ` +
      `${tta.prediction.replace(/_/g, " ")} @ ${(tta.confidence * 100).toFixed(1)}%</span>` +
      `<small>predictions are averaged over rotations &amp; flips for a more robust result</small>`;
  } else { note.innerHTML = ""; }

  const bars = $("bars");
  bars.innerHTML = "";
  Object.entries(r.probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cls, p], i) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (i === 0 ? " top" : "");
      row.innerHTML =
        `<span>${cls.replace(/_/g, " ")}</span>
         <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
         <span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
      bars.appendChild(row);
      requestAnimationFrame(() =>
        setTimeout(() => row.querySelector(".bar-fill").style.width = (p * 100) + "%", 30 * i));
    });

  const tb = $("param-table").querySelector("tbody");
  tb.innerHTML = "";
  const add = (k, v) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    tb.appendChild(tr);
  };
  add("Uploaded filename", r.filename);
  add("Bytes received", r.bytes_received.toLocaleString() + " B");
  add("Decoded array shape (H,W,C)", JSON.stringify(r.preprocessing.input_shape_decoded));
  add("Resized to (bilinear, NumPy)", r.preprocessing.resized_to.join("×"));
  add("Normalisation", r.preprocessing.normalisation);
  add("Feature vector dimension", r.preprocessing.feature_vector_dim);
  add("Top-class probability (softmax)", r.confidence);
  add("Entropy of prediction (bits)", r.preprocessing.entropy_bits);
  add("Augmented-view prediction", tta ? `${tta.prediction} @ ${(tta.confidence*100).toFixed(1)}%` : "-");
  add("End-to-end inference time", r.timings_ms.end_to_end + " ms");
  $("result-card").hidden = false;
  $("result-card").scrollIntoView({ behavior: "smooth" });
}

/* ---------------- metrics dashboard ---------------- */
fetch("/api/metrics").then(r => r.json()).then(m => {
  const el = $("metrics");

  let html = `<div class="grid2"><div>
    <h3>Architecture &amp; training</h3>
    <table><tbody>
      <tr><td>Model</td><td>${esc(m.architecture)}</td></tr>
      <tr><td>Loss</td><td>${esc(m.loss)}</td></tr>
      <tr><td>Optimiser</td><td>${esc(m.optimiser)}</td></tr>
      <tr><td>Trainable parameters</td><td>${m.params.toLocaleString()}</td></tr>
      <tr><td>Training samples</td><td>${m.n_train}</td></tr>
      <tr><td>Validation samples</td><td>${m.n_val}</td></tr>
      <tr><td>Epochs run</td><td>${m.history.length} (early-stopped)</td></tr>
      <tr><td>Train time</td><td>${m.train_seconds} s (CPU, NumPy)</td></tr>
      <tr><td>Gradient check (max rel. err.)</td><td>${m.gradient_check_max_rel_err.toExponential(2)}</td></tr>
    </tbody></table>
    <h3>Feature families</h3>
    ${m.feature_families.map(f => `<span class="tag">${esc(f.name)} · ${f.dim}d</span>`).join("")}
  </div><div>
    <h3>Validation report</h3>
    <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>
    ${m.classes.map((c, i) => {
      const r = m.val_report.per_class[i];
      return `<tr><td>${esc(c)}</td><td>${r.precision.toFixed(3)}</td>
              <td>${r.recall.toFixed(3)}</td><td>${r.f1.toFixed(3)}</td><td>${r.support}</td></tr>`;
    }).join("")}
    <tr><td><b>Overall accuracy</b></td><td colspan="3"></td><td><b>${(m.val_report.accuracy * 100).toFixed(2)}%</b></td></tr>
    <tr><td><b>Macro-F1</b></td><td colspan="3"></td><td><b>${m.val_report.macro_f1.toFixed(3)}</b></td></tr>
    </tbody></table>
    ${sparkline(m.history)}
  </div></div>`;

  html += "<h3>Confusion matrix (rows = true, cols = predicted)</h3>";
  html += '<table class="cm"><tr><th></th>' +
          m.classes.map(c => `<th>${esc(c)}</th>`).join("") + "</tr>";
  m.confusion_matrix.forEach((row, i) => {
    html += `<tr><th>${esc(m.classes[i])}</th>` + row.map((v, j) =>
      `<td class="${i === j ? "diag" : (v ? "off" : "")}">${v}</td>`).join("") + "</tr>";
  });
  html += "</table>";
  el.innerHTML = html;
}).catch(() => { $("metrics").innerHTML = "<em>Metrics unavailable.</em>"; });

function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }

/* tiny SVG validation-accuracy curve, pure DOM (no chart library) */
function sparkline(history) {
  if (!history || history.length < 2) return "";
  const w = 300, h = 70, pad = 4;
  const accs = history.map(p => p.val_acc);
  const lo = Math.min(...accs), hi = Math.max(...accs);
  const span = (hi - lo) || 1;
  const pts = accs.map((a, i) =>
    `${pad + i * (w - 2 * pad) / (accs.length - 1)},${h - pad - (a - lo) * (h - 2 * pad) / span}`
  ).join(" ");
  const best = (hi * 100).toFixed(1);
  return `<h3>Validation accuracy per epoch</h3>
    <svg viewBox="0 0 ${w} ${h}" class="spark" role="img" aria-label="validation accuracy curve">
      <polyline points="${pts}" fill="none" stroke="#38bdf8" stroke-width="2"/>
    </svg>
    <small class="mut">best val accuracy: ${best}% after ${accs.length} epochs</small>`;
}
