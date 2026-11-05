
const API_BASE = "https://truthlens-ai-api-j4r5.onrender.com";
const HISTORY_KEY = "truthlens_prediction_history_v1";
const $ = (id) => document.getElementById(id);
let latestResult = null;

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
}

function showToast(message) {
  const t = $("toast");
  t.textContent = message;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2300);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

function navigate(page) {
  document.querySelectorAll(".page").forEach(p =>
    p.classList.toggle("active", p.id === `page-${page}`)
  );

  document.querySelectorAll(".nav-link").forEach(b =>
    b.classList.toggle("active", b.dataset.page === page)
  );

  const names = {
    dashboard: "Dashboard",
    analysis: "News Analysis",
    history: "Prediction History",
    analytics: "Analytics",
    about: "About Project"
  };

  $("pageName").textContent = names[page] || "Dashboard";
  $("sidebar").classList.remove("open");

  if (page === "dashboard") renderDashboard();
  if (page === "history") renderHistory();

  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll("[data-page]").forEach(b =>
  b.addEventListener("click", e => {
    e.preventDefault();
    navigate(b.dataset.page);
  })
);

document.querySelectorAll("[data-go]").forEach(b =>
  b.addEventListener("click", () => navigate(b.dataset.go))
);

$("menuToggle").addEventListener("click", () =>
  $("sidebar").classList.toggle("open")
);

$("newsText").addEventListener("input", () => {
  const value = $("newsText").value.trim();
  $("wordCount").textContent =
    `${value ? value.split(/\s+/).length : 0} words`;
});

$("clearBtn").addEventListener("click", () => {
  $("newsTitle").value = "";
  $("newsText").value = "";
  $("wordCount").textContent = "0 words";
  $("formError").textContent = "";
  $("resultContent").classList.add("hidden");
  $("resultPlaceholder").classList.remove("hidden");
  latestResult = null;
});

function normalizePrediction(data) {
  const raw = String(
    data.label ?? data.prediction ?? data.result ?? ""
  ).toLowerCase();

  let label;

  if (raw.includes("fake") || raw === "0") {
    label = "Fake";
  } else if (raw.includes("real") || raw === "1") {
    label = "Real";
  } else {
    throw new Error("Unrecognised prediction label.");
  }

  return {
    label,
    model: data.model ?? "Unknown",
    score: null,
    note: data.disclaimer ??
      data.note ??
      "This is a model prediction, not independent fact verification."
  };
}

$("analyzeBtn").addEventListener("click", async () => {
  const title = $("newsTitle").value.trim();
  const text = $("newsText").value.trim();

  $("formError").textContent = "";

  if (!title && !text) {
    $("formError").textContent =
      "Please enter a headline or article text.";
    return;
  }

  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.textContent = "Analysing…";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text })
    });

    if (!response.ok) {
      let detail = `API returned ${response.status}`;

      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch {}

      throw new Error(detail);
    }

    const data = await response.json();
    const result = normalizePrediction(data);

    latestResult = {
      ...result,
      title,
      text,
      createdAt: new Date().toISOString(),
      raw: data
    };

    renderResult(latestResult);

    const history = getHistory();
    history.unshift(latestResult);
    saveHistory(history.slice(0, 100));

    showToast("Prediction completed and saved.");

  } catch (err) {
    $("formError").textContent =
      `Could not get a prediction: ${err.message}. Check that the FastAPI backend is running and its /predict endpoint is available.`;

  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Analyse article <span>↗</span>';
  }
});

function renderResult(r) {
  $("resultPlaceholder").classList.add("hidden");
  $("resultContent").classList.remove("hidden");

  $("resultLabel").textContent =
    `PREDICTED ${r.label.toUpperCase()}`;

  $("resultLabel").classList.toggle(
    "fake",
    r.label === "Fake"
  );

  $("resultHeadline").textContent =
    r.label === "Fake"
      ? "Model detected fake-news patterns"
      : "Model detected real-news patterns";

  $("resultDescription").textContent =
    `The model classified the submitted article as ${r.label}. This label reflects the model's learned patterns, not independent verification of the story.`;

  $("decisionValue").textContent = r.model;
  $("resultNote").textContent = r.note;
}

function renderDashboard() {
  const h = getHistory();
  const fake = h.filter(x => x.label === "Fake").length;
  const real = h.filter(x => x.label === "Real").length;

  $("totalCount").textContent = h.length;
  $("fakeCount").textContent = fake;
  $("realCount").textContent = real;

  $("latestDate").textContent = h.length
    ? new Date(h[0].createdAt).toLocaleDateString()
    : "—";

  $("recentList").innerHTML = h.length
    ? h.slice(0, 4).map(x => `
        <div class="recent-row">
          <b>${escapeHTML(x.title || x.text.slice(0, 65) || "Untitled article")}</b>
          <span class="badge ${x.label.toLowerCase()}">${escapeHTML(x.label)}</span>
        </div>
      `).join("")
    : '<div class="empty-state compact"><span>◷</span><b>No activity yet</b><p>Your predictions will appear here.</p></div>';
}

function renderHistory() {
  const q = $("historySearch").value.toLowerCase();
  const filter = $("historyFilter").value;

  const rows = getHistory().filter(x =>
    (filter === "all" || x.label === filter) &&
    (`${x.title} ${x.text}`.toLowerCase().includes(q))
  );

  $("historyRows").innerHTML = rows.length
    ? rows.map(x => `
        <tr>
          <td title="${escapeHTML(x.title || "")}">
            ${escapeHTML(x.title || x.text.slice(0, 90) || "Untitled article")}
          </td>
          <td>
            <span class="badge ${x.label.toLowerCase()}">${escapeHTML(x.label)}</span>
          </td>
          <td>${escapeHTML(new Date(x.createdAt).toLocaleString())}</td>
        </tr>
      `).join("")
    : '<tr><td colspan="3" class="table-empty">No matching predictions found.</td></tr>';
}

$("historySearch").addEventListener("input", renderHistory);
$("historyFilter").addEventListener("change", renderHistory);

$("clearHistoryBtn").addEventListener("click", () => {
  if (confirm("Delete all prediction history saved in this browser?")) {
    saveHistory([]);
    renderHistory();
    renderDashboard();
    showToast("History cleared.");
  }
});

// Copy complete prediction report
$("copyBtn").addEventListener("click", async () => {
  if (!latestResult) return;

  const r = latestResult;

  const report = `TruthLens AI - News Analysis Report
Project: PCS26/84

Prediction: ${r.label}
Headline: ${r.title || "Untitled article"}
Model: ${r.model}

Article Content:
${r.text}

Disclaimer:
${r.note}

This is a model prediction, not independent fact verification.`;

  try {
    await navigator.clipboard.writeText(report);
    showToast("Complete report copied.");
  } catch {
    showToast("Clipboard access unavailable.");
  }
});

// Generate complete PDF / print report
$("pdfBtn").addEventListener("click", () => {
  if (!latestResult) return;

  const r = latestResult;
  const w = window.open("", "_blank");

  if (!w) {
    showToast("Allow pop-ups to print the report.");
    return;
  }

  w.document.write(`<!doctype html>
  <html>
  <head>
    <meta charset="UTF-8">
    <title>TruthLens AI Report</title>
    <style>
      body {
        font: 15px Arial, sans-serif;
        max-width: 760px;
        margin: 50px auto;
        padding: 0 20px;
        color: #17243a;
        line-height: 1.7;
      }

      h1 { color: #1769aa; }

      .label {
        font-size: 22px;
        font-weight: bold;
        color: ${r.label === "Fake" ? "#b54737" : "#16845d"};
      }

      .article {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        background: #f5f7fa;
        padding: 15px;
        border-radius: 8px;
      }

      .muted {
        color: #68788b;
        border-top: 1px solid #ddd;
        padding-top: 18px;
        margin-top: 25px;
      }

      @media print {
        body { margin: 20px auto; }
      }
    </style>
  </head>

  <body>
    <h1>TruthLens AI</h1>
    <p>News analysis report · PCS26/84</p>

    <h2>${escapeHTML(r.title || "Untitled article")}</h2>

    <p class="label">Predicted: ${escapeHTML(r.label)}</p>

    <p><b>Model:</b> ${escapeHTML(r.model)}</p>

    <h3>Article Content</h3>
    <div class="article">${escapeHTML(r.text)}</div>

    <h3>Model Disclaimer</h3>
    <p>${escapeHTML(r.note)}</p>

    <p class="muted">
      This report contains a machine-learning prediction,
      not independent verification of factual claims.
    </p>

    <script>
      window.onload = () => window.print();
    <\/script>
  </body>
  </html>`);

  w.document.close();
});

renderDashboard();
