/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const labels = { queued: "Queued", in_progress: "Generating", completed: "Completed", failed: "Failed" };
  const active = (job) => ["queued", "in_progress"].includes(job.status);
  let model = "";
  let readyLabel = "Server ready";
  let busy = false;
  let currentJob = null;
  let pollVersion = 0;

  const api = async (path, options = {}) => {
    const response = await fetch(path, { ...options, signal: AbortSignal.timeout(15000) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || `Request failed (HTTP ${response.status}).`);
    return body;
  };
  const setBusy = (value) => {
    busy = value;
    $("generate").disabled = value || !model;
    $("generate").textContent = value ? "Waiting for this job…" : "Generate video ↗";
  };
  const showError = (message = "") => {
    $("error").textContent = message;
    $("error").hidden = !message;
  };
  const payload = () => {
    const body = { model, prompt: $("prompt").value.trim() };
    if ($("seed").value !== "") body.seed = Number($("seed").value);
    return body;
  };
  const updateCurl = () => {
    const quote = (text) => "'" + text.replaceAll("'", "'\\''") + "'";
    $("curl-command").textContent = `curl --fail-with-body ${quote(location.origin + "/v1/videos")} \\\n  -H 'Content-Type: application/json' \\\n  --data-raw ${quote(JSON.stringify(payload(), null, 2))}`;
    $("copy-curl").disabled = !model || !$("prompt").value.trim() || !$("seed").validity.valid;
    $("copy-status").textContent = "";
  };
  const refreshJobs = async () => {
    $("refresh-jobs").disabled = true;
    try {
      const result = await api("/v1/videos?limit=8&order=desc");
      $("jobs").replaceChildren();
      result.data.forEach((job) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("aria-current", String(job.id === currentJob?.id));
        const prompt = document.createElement("span");
        prompt.className = "job-prompt";
        prompt.textContent = job.prompt || job.id;
        const state = document.createElement("span");
        state.className = "job-state";
        state.textContent = labels[job.status] || job.status;
        button.append(prompt, state);
        button.addEventListener("click", () => {
          if (busy) {
            showError("Wait for this job, or open another playground tab to inspect a different job.");
            return;
          }
          $("prompt").value = job.prompt || "";
          // Jobs do not report the seed; do not pair an old seed with this prompt.
          $("seed").value = "";
          updateCurl();
          followJob(job.id);
        });
        item.append(button);
        $("jobs").append(item);
      });
      $("history-status").textContent = result.data.length ? "Showing the latest jobs from this server." : "No jobs yet. Submit a prompt to start.";
    } catch (error) {
      $("history-status").textContent = `Could not load jobs. Check the server and select Refresh jobs. ${error.message}`;
    } finally {
      $("refresh-jobs").disabled = false;
    }
  };
  const renderJob = (job) => {
    currentJob = job;
    $("job-state").textContent = labels[job.status] || job.status;
    $("job-state").dataset.status = job.status;
    $("job-id").textContent = `Job ${job.id}`;
    const path = `/v1/videos/${encodeURIComponent(job.id)}`;
    $("job-link").href = path;
    $("job-link").hidden = false;
    const complete = job.status === "completed";
    $("video").hidden = !complete;
    $("empty-preview").hidden = complete;
    $("download").hidden = !complete;
    if (complete) {
      $("video").src = `${path}/content`;
      $("download").href = `${path}/content`;
      $("download").download = `${job.id}.mp4`;
      $("job-status").textContent = "Video ready. Change the prompt and generate again without restarting the server.";
    } else {
      $("video").pause();
      $("video").removeAttribute("src");
      $("empty-preview").querySelector("h3").textContent = job.status === "failed" ? "Generation failed" : "Your job is on the server";
      $("empty-preview").querySelector("p").textContent = job.status === "failed" ? "Read the error below before trying again." : "You can keep editing the next prompt while you wait.";
      $("job-status").textContent = job.status === "queued" ? "Queued. The server runs one generation at a time." : job.status === "failed" ? "The server could not complete this job." : "Generating video and audio. This page checks the job status automatically.";
    }
    if (job.status === "failed") showError(job.error?.message || "Check the server logs before submitting another job.");
  };
  const followJob = async (id) => {
    const version = ++pollVersion;
    const url = new URL(location.href);
    url.searchParams.set("job", id);
    history.replaceState({}, "", url);
    showError();
    setBusy(true);
    $("check-status").hidden = true;
    const deadline = Date.now() + 30 * 60 * 1000;
    let restorePrompt = !$("prompt").value.trim();
    try {
      while (version === pollVersion) {
        const job = await api(`/v1/videos/${encodeURIComponent(id)}`);
        if (version !== pollVersion) return;
        $("connection").textContent = readyLabel;
        if (restorePrompt) {
          $("prompt").value = job.prompt || "";
          updateCurl();
          restorePrompt = false;
        }
        renderJob(job);
        if (!active(job)) break;
        if (Date.now() >= deadline) throw new Error("Stopped checking after 30 minutes. The job may still be running.");
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch (error) {
      if (version !== pollVersion) return;
      $("connection").textContent = "Job status unavailable · check the server";
      showError(`${error.message} Select Check status to reconnect. A connection error does not cancel generation. If the job is missing, the server may have restarted or the job was deleted.`);
      $("check-status").hidden = false;
    } finally {
      if (version === pollVersion) {
        setBusy(false);
        await refreshJobs();
      }
    }
  };
  $("generate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy || !model) return;
    if (!$("prompt").value.trim()) {
      showError("Write a prompt before generating a video.");
      $("prompt").focus();
      return;
    }
    showError();
    setBusy(true);
    $("job-status").textContent = "Submitting the prompt…";
    try {
      const job = await api("/v1/videos", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()),
      });
      renderJob(job);
      await followJob(job.id);
    } catch (error) {
      showError(`${error.message} Check Recent jobs before submitting again; the server may have received the prompt. Submissions are never retried automatically.`);
      $("job-status").textContent = "Could not confirm the submission.";
      setBusy(false);
      await refreshJobs();
    }
  });
  $("prompt").addEventListener("input", updateCurl);
  $("seed").addEventListener("input", updateCurl);
  $("refresh-jobs").addEventListener("click", refreshJobs);
  $("check-status").addEventListener("click", () => {
    const id = new URL(location.href).searchParams.get("job");
    if (id) followJob(id);
  });
  $("video").addEventListener("error", () => {
    if (!$("video").hasAttribute("src")) return;
    showError("The browser could not play this video. Download the MP4 to check it, or inspect the server logs.");
  });
  $("copy-curl").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("curl-command").textContent);
      $("copy-status").textContent = "Copied. Run it in a terminal to create a job.";
    } catch {
      $("copy-status").textContent = "Clipboard access is unavailable. Select and copy the command above.";
    }
  });
  const connect = async () => {
    try {
      await api("/health");
      const config = await api("/playground/config");
      model = config.model;
      readyLabel = config.runtime === "mlx" ? "Server ready · MLX" : "Server ready · model loaded";
      $("connection").textContent = readyLabel;
      $("lifetime").textContent = config.runtime === "mlx"
        ? "MLX keeps the server and prompt cache available, but releases model components between phases to limit unified-memory use. Closing this page does not cancel a job."
        : "The model stays loaded until you stop the server. Closing this page does not cancel a job.";
      $("model").textContent = model;
      const d = config.defaults;
      const facts = [];
      if (d.width && d.height) facts.push(`${d.width} × ${d.height}`);
      if (d.num_frames) facts.push(`${d.num_frames} frames`);
      if (d.fps) facts.push(`${d.fps} fps`);
      if (d.seed != null) $("seed").placeholder = String(d.seed);
      $("settings").textContent = facts.length ? `Server defaults · ${facts.join(" · ")}` : "Resolution and sampling come from the server configuration.";
      setBusy(false);
      updateCurl();
      const id = new URL(location.href).searchParams.get("job");
      if (id) await followJob(id);
      else await refreshJobs();
    } catch (error) {
      $("connection").textContent = "Server unavailable";
      $("history-status").textContent = "Start the H3 server, then reload this page.";
      showError(`${error.message} Check the server terminal and reload this page when model loading completes.`);
    }
  };
  connect();
})();
