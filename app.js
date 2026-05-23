// NutriVision static site — used on GitHub Pages.
// The live demo POSTs to API_URL.  If empty/null, the demo gracefully degrades.

// ★★★  PERMANENT API URL  ★★★
// Modal serverless backend — fully independent of the institute server,
// scales to zero, free-tier hosted. Serves /api/predict and /api/suggest.
// First request after idle is a ~90s cold start; warm requests are quick.
const API_URL = "https://kianraj--nutrivision-nutrivision-web.modal.run";

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

// ────────────────── dish carousel ──────────────────
// Scroll-snap track + prev/next arrows + page-dot indicator.
// Supports keyboard ← → when the carousel is focused, plus swipe (native).
function initCarousel(root) {
  const track = root.querySelector("[data-carousel-track]");
  const dotsWrap = root.querySelector("[data-carousel-dots]");
  const prevBtn = root.querySelector(".car-prev");
  const nextBtn = root.querySelector(".car-next");
  if (!track) return;

  const cards = Array.from(track.children);
  if (cards.length === 0) return;

  // step = card width + gap
  const styles = window.getComputedStyle(track);
  const gap = parseFloat(styles.columnGap || styles.gap || "0") || 0;
  const step = () => cards[0].getBoundingClientRect().width + gap;

  // build dots — one per card (kept simple; visible cards depend on viewport)
  if (dotsWrap) {
    dotsWrap.innerHTML = "";
    cards.forEach((_, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-label", `Go to dish ${i + 1}`);
      b.addEventListener("click", () => {
        track.scrollTo({ left: i * step(), behavior: "smooth" });
      });
      dotsWrap.appendChild(b);
    });
  }

  function update() {
    const max = track.scrollWidth - track.clientWidth - 1;
    const left = track.scrollLeft;
    if (prevBtn) prevBtn.disabled = left <= 0;
    if (nextBtn) nextBtn.disabled = left >= max;
    if (dotsWrap) {
      const idx = Math.round(left / step());
      Array.from(dotsWrap.children).forEach((d, i) =>
        d.setAttribute("aria-current", i === idx ? "true" : "false")
      );
    }
  }

  function scrollBy(dir) {
    // scroll by one card; on wide viewports the user still sees several
    track.scrollBy({ left: dir * step(), behavior: "smooth" });
  }

  prevBtn?.addEventListener("click", () => scrollBy(-1));
  nextBtn?.addEventListener("click", () => scrollBy(1));
  track.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);

  // keyboard support — only when the carousel area has focus
  root.tabIndex = -1;
  root.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft")  { e.preventDefault(); scrollBy(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); scrollBy(1);  }
  });

  update();
}

document.querySelectorAll("[data-carousel]").forEach(initCarousel);

// ────────────────── paper-overview horizontal auto-slider ──────────────────
function initPaperSlider(root) {
  const track  = root.querySelector("[data-paper-track]");
  const slides = track ? Array.from(track.children) : [];
  const dots   = Array.from(root.querySelectorAll("[data-paper-dots] button"));
  const interval = parseInt(root.dataset.interval || "6000", 10);
  if (!track || slides.length < 2) return;

  let i = 0, timer = null, paused = false;
  const N = slides.length;

  function show(next) {
    i = ((next % N) + N) % N;
    track.style.transform = `translateX(-${(100 / N) * i}%)`;
    dots.forEach((d, k) => d.classList.toggle("is-active", k === i));
  }
  function start() { stop(); timer = setInterval(() => { if (!paused) show(i + 1); }, interval); }
  function stop()  { if (timer) { clearInterval(timer); timer = null; } }

  dots.forEach((d, idx) => d.addEventListener("click", () => { show(idx); start(); }));
  root.addEventListener("mouseenter", () => { paused = true;  });
  root.addEventListener("mouseleave", () => { paused = false; });
  root.addEventListener("focusin",    () => { paused = true;  });
  root.addEventListener("focusout",   () => { paused = false; });

  show(0); start();
}
document.querySelectorAll("[data-paper-slider]").forEach(initPaperSlider);
