// NutriVision static site — used on GitHub Pages.
// The live demo POSTs to API_URL.  If empty/null, the demo gracefully degrades.

// ★★★  PASTE YOUR CLOUDFLARE TUNNEL URL HERE  ★★★
// e.g. const API_URL = "https://orange-pancake-22.trycloudflare.com";
const API_URL = "";

// ────────────────── theme cycle (academic → forest → dark) ──────────────────
const THEMES = ["academic", "forest", "dark"];
const themeBtn = document.getElementById("theme-btn");
if (themeBtn) {
  const saved = localStorage.getItem("nv-theme");
  if (saved && THEMES.includes(saved)) document.body.dataset.theme = saved;
  themeBtn.addEventListener("click", () => {
    const cur = document.body.dataset.theme || "academic";
    const next = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
    document.body.dataset.theme = next;
    localStorage.setItem("nv-theme", next);
  });
}

// ────────────────── DOM ──────────────────
const fileInput   = document.getElementById("image-input");
const dropLabel   = document.getElementById("dropzone-label");
const preview     = document.getElementById("preview");
const ingredients = document.getElementById("ingredients");
const caption     = document.getElementById("caption");
const form        = document.getElementById("predict-form");
const predictBtn  = document.getElementById("predict-btn");
const suggestBtn  = document.getElementById("suggest-btn");
const resultsCard = document.getElementById("results-card");
const resultsBody = document.querySelector("#results-table tbody");
const metaP       = document.getElementById("meta");
const demoBlurb   = document.getElementById("demo-blurb");

const DV = { calories_kcal: 2000, fat_g: 78, carbs_g: 275, protein_g: 50 };
const PMAE = parseFloat(document.body.dataset.pmae) || 13.52;

// ────────────────── graceful degrade if no backend ──────────────────
function disableDemo(reason) {
  if (demoBlurb) {
    demoBlurb.innerHTML =
      `<b>Live demo offline:</b> ${reason} ` +
      `Set <code>API_URL</code> at the top of <code>app.js</code> to your ` +
      `backend (e.g. a Cloudflare Tunnel URL) to enable.`;
  }
  [fileInput, ingredients, predictBtn, suggestBtn].forEach(el => {
    if (el) el.disabled = true;
  });
  predictBtn?.classList.remove("primary");
}
if (!API_URL) {
  disableDemo("backend URL not configured.");
}

// ────────────────── file preview ──────────────────
fileInput?.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.style.display = "block";
  dropLabel.style.display = "none";
});

// ────────────────── auto-suggest ──────────────────
suggestBtn?.addEventListener("click", async () => {
  if (!API_URL) return;
  if (!fileInput.files[0]) { alert("Choose an image first."); return; }
  suggestBtn.disabled = true; suggestBtn.textContent = "Suggesting…";
  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  try {
    const r = await fetch(`${API_URL}/api/suggest`, { method: "POST", body: fd });
    const j = await r.json();
    if (j.ok) {
      ingredients.value = j.ingredients || "";
      caption.textContent = j.caption ? "BLIP caption: " + j.caption : "";
    } else { caption.textContent = "Error: " + (j.error || "unknown"); }
  } catch (e) { caption.textContent = "Network error: " + e.message; }
  suggestBtn.disabled = false; suggestBtn.textContent = "Auto-suggest";
});

// ────────────────── predict ──────────────────
form?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!API_URL) return;
  if (!fileInput.files[0]) { alert("Choose an image first."); return; }
  predictBtn.disabled = true; predictBtn.textContent = "Predicting…";
  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  fd.append("ingredients", ingredients.value || "");
  try {
    const r = await fetch(`${API_URL}/api/predict`, { method: "POST", body: fd });
    const j = await r.json();
    if (j.ok) { renderResults(j); }
    else { alert("Error: " + (j.error || "unknown")); }
  } catch (e) { alert("Network error: " + e.message); }
  predictBtn.disabled = false; predictBtn.textContent = "Predict nutrition";
});

function renderResults(j) {
  const p = j.predictions;
  resultsBody.innerHTML = "";
  const rows = [
    ["Calories", p.calories_kcal, "kcal", "calories_kcal"],
    ["Mass",     p.mass_g,        "g",    null],
    ["Fat",      p.fat_g,         "g",    "fat_g"],
    ["Carbs",    p.carbs_g,       "g",    "carbs_g"],
    ["Protein",  p.protein_g,     "g",    "protein_g"],
  ];
  for (const [name, val, unit, dvKey] of rows) {
    const band = (PMAE / 100) * val;
    const dv = dvKey ? ((val / DV[dvKey]) * 100).toFixed(0) + "%" : "—";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td>
                    <td><b>${val.toFixed(1)} ${unit}</b></td>
                    <td>±${band.toFixed(1)}</td>
                    <td>${dv}</td>`;
    resultsBody.appendChild(tr);
  }
  metaP.textContent =
    `Latency: ${j.latency_ms} ms · depth range: ${j.depth_min.toFixed(2)}–` +
    `${j.depth_max.toFixed(2)} · checkpoint: ${j.checkpoint}`;
  resultsCard.classList.remove("hidden");
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}
