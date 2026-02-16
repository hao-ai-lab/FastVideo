/* ────────────────────────────────────────────────────────────
   FastVideo Job Runner – Frontend Application
   ──────────────────────────────────────────────────────────── */

const API = "";  // same origin

// ── Helpers ──────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

function relTime(ts) {
  if (!ts) return "—";
  const diff = (Date.now() / 1000 - ts);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function duration(start, end) {
  if (!start) return "";
  const elapsed = (end || Date.now() / 1000) - start;
  if (elapsed < 60) return `${elapsed.toFixed(1)}s`;
  return `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s`;
}

// ── Toast notifications ──────────────────────────────────────

let toastContainer;
function initToast() {
  toastContainer = document.createElement("div");
  toastContainer.className = "toast-container";
  document.body.appendChild(toastContainer);
}

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = msg;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ── State ────────────────────────────────────────────────────

let models = [];
let pollTimer = null;

// Track which consoles are open and their scroll-offset for incremental fetch
const openConsoles = new Map(); // jobId -> { after: number, autoScroll: bool }

// ── Bootstrap ────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initToast();

  // Load models
  try {
    models = await api("GET", "/api/models");
    populateModelSelect(models);
  } catch (e) {
    toast("Failed to load models: " + e.message, true);
  }

  // Wire up form
  document.getElementById("job-form").addEventListener("submit", onCreateJob);
  document.getElementById("refresh-btn").addEventListener("click", refreshJobs);

  // Modal close handlers
  document.querySelector(".modal-backdrop").addEventListener("click", closeModal);
  document.querySelector(".modal-close").addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Initial job list load & start polling
  refreshJobs();
  startPolling();
});

function populateModelSelect(models) {
  const sel = document.getElementById("model-select");
  sel.innerHTML = '<option value="" disabled selected>Select a model…</option>';
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label}  (${m.id})`;
    sel.appendChild(opt);
  }
}

// ── Polling ──────────────────────────────────────────────────

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(() => {
    refreshJobs();
    pollOpenConsoles();
  }, 2000);
}

// ── Create Job ───────────────────────────────────────────────

async function onCreateJob(e) {
  e.preventDefault();
  const btn = document.getElementById("create-btn");
  btn.disabled = true;

  const payload = {
    model_id:            document.getElementById("model-select").value,
    prompt:              document.getElementById("prompt-input").value.trim(),
    num_inference_steps: parseInt(document.getElementById("num-steps").value, 10),
    num_frames:          parseInt(document.getElementById("num-frames").value, 10),
    height:              parseInt(document.getElementById("height").value, 10),
    width:               parseInt(document.getElementById("width").value, 10),
    guidance_scale:      parseFloat(document.getElementById("guidance").value),
    seed:                parseInt(document.getElementById("seed").value, 10),
    num_gpus:            parseInt(document.getElementById("num-gpus").value, 10),
  };

  try {
    await api("POST", "/api/jobs", payload);
    toast("Job created");
    document.getElementById("prompt-input").value = "";
    refreshJobs();
  } catch (err) {
    toast("Create failed: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ── Job Actions ──────────────────────────────────────────────

async function startJob(id) {
  try {
    await api("POST", `/api/jobs/${id}/start`);
    toast("Job started");
    refreshJobs();
  } catch (err) {
    toast("Start failed: " + err.message, true);
  }
}

async function stopJob(id) {
  try {
    await api("POST", `/api/jobs/${id}/stop`);
    toast("Stop requested");
    refreshJobs();
  } catch (err) {
    toast("Stop failed: " + err.message, true);
  }
}

async function deleteJob(id) {
  if (!confirm("Delete this job?")) return;
  try {
    await api("DELETE", `/api/jobs/${id}`);
    toast("Job deleted");
    openConsoles.delete(id);
    refreshJobs();
  } catch (err) {
    toast("Delete failed: " + err.message, true);
  }
}

function viewVideo(id) {
  const modal = document.getElementById("video-modal");
  const video = document.getElementById("modal-video");
  video.src = `/api/jobs/${id}/video`;
  modal.classList.remove("hidden");
}

function closeModal() {
  const modal = document.getElementById("video-modal");
  const video = document.getElementById("modal-video");
  modal.classList.add("hidden");
  video.pause();
  video.src = "";
}

// ── Console toggling ─────────────────────────────────────────

function toggleConsole(id) {
  const panel = document.getElementById(`console-${id}`);
  if (!panel) return;

  if (panel.classList.contains("hidden")) {
    panel.classList.remove("hidden");
    if (!openConsoles.has(id)) {
      openConsoles.set(id, { after: 0, autoScroll: true });
    }
    // Fetch immediately
    fetchLogs(id);
  } else {
    panel.classList.add("hidden");
    openConsoles.delete(id);
  }
}

async function fetchLogs(id) {
  const state = openConsoles.get(id);
  if (!state) return;
  try {
    const data = await api("GET", `/api/jobs/${id}/logs?after=${state.after}`);
    const pre = document.getElementById(`console-output-${id}`);
    if (!pre) return;

    if (data.lines.length > 0) {
      pre.textContent += data.lines.join("\n") + "\n";
      state.after = data.total;
      // Auto-scroll to bottom
      if (state.autoScroll) {
        pre.scrollTop = pre.scrollHeight;
      }
    }
  } catch {
    // silently ignore fetch errors for logs
  }
}

function pollOpenConsoles() {
  for (const id of openConsoles.keys()) {
    fetchLogs(id);
  }
}

// ── Render Jobs ──────────────────────────────────────────────

async function refreshJobs() {
  try {
    const jobs = await api("GET", "/api/jobs");
    renderJobs(jobs);
  } catch (err) {
    // Silently swallow poll errors to avoid toast spam
    console.error("Refresh failed:", err);
  }
}

function renderJobs(jobs) {
  const container = document.getElementById("jobs-container");

  if (!jobs.length) {
    container.innerHTML = '<p class="placeholder">No jobs yet. Create one above.</p>';
    return;
  }

  // Preserve open console contents across re-renders
  const savedConsoles = {};
  for (const [id] of openConsoles) {
    const pre = document.getElementById(`console-output-${id}`);
    if (pre) savedConsoles[id] = pre.textContent;
  }

  container.innerHTML = jobs.map(renderJobCard).join("");

  // Restore console contents
  for (const [id, text] of Object.entries(savedConsoles)) {
    const pre = document.getElementById(`console-output-${id}`);
    if (pre) {
      pre.textContent = text;
      pre.scrollTop = pre.scrollHeight;
    }
    // Make sure the panel is visible
    const panel = document.getElementById(`console-${id}`);
    if (panel) panel.classList.remove("hidden");
  }

  // Bind action buttons
  container.querySelectorAll("[data-action]").forEach((btn) => {
    const action = btn.dataset.action;
    const id     = btn.dataset.id;
    btn.addEventListener("click", () => {
      if (action === "start")   startJob(id);
      if (action === "stop")    stopJob(id);
      if (action === "delete")  deleteJob(id);
      if (action === "view")    viewVideo(id);
      if (action === "console") toggleConsole(id);
    });
  });
}

function renderJobCard(job) {
  const badgeCls = `badge badge-${job.status}`;
  const modelLabel = models.find(m => m.id === job.model_id)?.label || job.model_id;

  // ── Action buttons ──
  let actions = "";
  if (job.status === "pending" || job.status === "stopped" || job.status === "failed") {
    actions += `<button class="btn btn-start" data-action="start" data-id="${job.id}">&#9654; Start</button>`;
  }
  if (job.status === "running") {
    actions += `<button class="btn btn-stop" data-action="stop" data-id="${job.id}">&#9632; Stop</button>`;
  }
  if (job.status === "completed") {
    actions += `<button class="btn btn-view" data-action="view" data-id="${job.id}">&#9654; View</button>`;
  }
  actions += `<button class="btn btn-console" data-action="console" data-id="${job.id}">&#9000; Console</button>`;
  actions += `<button class="btn btn-delete" data-action="delete" data-id="${job.id}">&#10005; Delete</button>`;

  // ── Error ──
  let errorHtml = "";
  if (job.error) {
    errorHtml = `<div class="job-error">${escapeHtml(job.error)}</div>`;
  }

  // ── Timing ──
  const elapsed = job.status === "running"
    ? `⏱ ${duration(job.started_at)}`
    : job.started_at
      ? `⏱ ${duration(job.started_at, job.finished_at)}`
      : "";

  // ── Progress bar ──
  const pct = Math.min(Math.max(job.progress || 0, 0), 100);
  const progressLabel = job.progress_msg || job.phase || "";
  const showBar = job.status === "running" || job.status === "completed";
  const progressHtml = showBar ? `
    <div class="progress-container">
      <div class="progress-bar-bg">
        <div class="progress-bar-fill ${job.status === "completed" ? "completed" : ""}" style="width:${pct}%"></div>
      </div>
      <span class="progress-label">${escapeHtml(progressLabel)}${pct > 0 ? ` — ${pct.toFixed(0)}%` : ""}</span>
    </div>` : "";

  // ── Console panel (hidden by default) ──
  const consoleOpen = openConsoles.has(job.id);
  const consoleHtml = `
    <div id="console-${job.id}" class="console-panel ${consoleOpen ? "" : "hidden"}">
      <pre id="console-output-${job.id}" class="console-output"></pre>
    </div>`;

  return `
    <div class="job-card">
      <div class="job-header">
        <span class="job-model">${escapeHtml(modelLabel)}</span>
        <span class="${badgeCls}">${job.status}</span>
      </div>
      <div class="job-prompt" title="${escapeHtml(job.prompt)}">${escapeHtml(job.prompt)}</div>
      <div class="job-meta">
        <span>Created ${relTime(job.created_at)}</span>
        ${elapsed ? `<span>${elapsed}</span>` : ""}
        <span>${job.num_frames} frames · ${job.width}×${job.height}</span>
        <span>Steps: ${job.num_inference_steps}</span>
        <span>Seed: ${job.seed}</span>
      </div>
      ${progressHtml}
      ${errorHtml}
      <div class="job-actions">${actions}</div>
      ${consoleHtml}
    </div>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
