/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const labels = { queued: "Queued", in_progress: "Generating", completed: "Completed", failed: "Failed" };
  const active = (job) => ["queued", "in_progress"].includes(job.status);
  let model = "";
  let readyLabel = "Server ready";
  let watching = null;
  let jobs = [];
  let pollTimer = 0;
  let pollVersion = 0;
  let submitting = false;

  const api = async (path, options = {}) => {
    const response = await fetch(path, { ...options, signal: AbortSignal.timeout(15000) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || `Request failed (HTTP ${response.status}).`);
    return body;
  };
  const jobPath = (id) => `/v1/videos/${encodeURIComponent(id)}`;
  const syncGenerateButton = () => {
    $("generate").disabled = !model || submitting;
    $("generate").textContent = jobs.some(active) ? "Queue next clip ↗" : "Generate video ↗";
  };
  const watchingIndex = () => jobs.findIndex((job) => job.id === watching?.id);
  const syncClipNav = () => {
    const index = watchingIndex();
    const show = jobs.length > 1 && index >= 0;
    $("clip-nav").hidden = !show;
    if (!show) return;
    $("older-clip").disabled = index >= jobs.length - 1;
    $("newer-clip").disabled = index <= 0;
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
  const setVideoSource = (job) => {
    const video = $("video");
    const complete = job.status === "completed";
    video.hidden = !complete;
    $("empty-preview").hidden = complete;
    $("download").hidden = !complete;
    if (!complete) {
      video.pause();
      video.removeAttribute("src");
      delete video.dataset.job;
      $("empty-preview").querySelector("h3").textContent = job.status === "failed" ? "Generation failed" : "Your job is on the server";
      $("empty-preview").querySelector("p").textContent = job.status === "failed"
        ? "Read the error below before trying again."
        : "Queue another prompt or open a finished clip from Recent jobs. This one keeps generating.";
      return;
    }
    $("download").href = `${jobPath(job.id)}/content`;
    $("download").download = `${job.id}.mp4`;
    if (video.dataset.job !== job.id) {
      video.src = `${jobPath(job.id)}/content`;
      video.dataset.job = job.id;
    }
  };
  const renderWatching = (job) => {
    watching = job;
    const url = new URL(location.href);
    url.searchParams.set("job", job.id);
    history.replaceState({}, "", url);
    $("job-state").textContent = labels[job.status] || job.status;
    $("job-state").dataset.status = job.status;
    $("job-id").textContent = `Job ${job.id}`;
    $("job-link").href = jobPath(job.id);
    $("job-link").hidden = false;
    setVideoSource(job);
    if (job.status === "completed") {
      $("job-status").textContent = "Playing this clip. Queue the next prompt without leaving it, or pick another job from Recent jobs.";
    } else if (job.status === "queued") {
      $("job-status").textContent = "Queued. The server runs one generation at a time. Finished clips stay playable while you wait.";
    } else if (job.status === "failed") {
      $("job-status").textContent = "The server could not complete this job.";
    } else {
      $("job-status").textContent = "Generating. Open any finished clip from Recent jobs; this job keeps running.";
    }
    if (job.status === "failed") showError(job.error?.message || "Check the server logs before submitting another job.");
    syncClipNav();
    for (const button of $("jobs").querySelectorAll("button[data-job-id]")) {
      if (button.dataset.jobId === job.id) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
  };
  const selectJob = (job) => {
    showError();
    renderWatching(job);
  };
  const renderHistory = () => {
    $("jobs").replaceChildren();
    jobs.forEach((job) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.jobId = job.id;
      if (job.id === watching?.id) button.setAttribute("aria-current", "true");
      const prompt = document.createElement("span");
      prompt.className = "job-prompt";
      prompt.textContent = job.prompt || job.id;
      const state = document.createElement("span");
      state.className = "job-state";
      state.textContent = labels[job.status] || job.status;
      button.append(prompt, state);
      button.addEventListener("click", () => selectJob(job));
      item.append(button);
      $("jobs").append(item);
    });
    const queued = jobs.filter((job) => job.status === "queued").length;
    const running = jobs.some((job) => job.status === "in_progress");
    if (!jobs.length) {
      $("history-status").textContent = "No jobs yet. Submit a prompt to start.";
    } else if (queued || running) {
      $("history-status").textContent = running
        ? `One clip generating${queued ? `, ${queued} queued` : ""}. Select any row to play or inspect it.`
        : `${queued} clip${queued === 1 ? "" : "s"} queued. Select any row to play a finished clip.`;
    } else {
      $("history-status").textContent = "Select any clip to play it. The prompt box is left alone so you can queue the next one.";
    }
  };
  const schedulePoll = () => {
    clearTimeout(pollTimer);
    if (!jobs.some(active)) return;
    const version = ++pollVersion;
    pollTimer = setTimeout(async () => {
      if (version !== pollVersion) return;
      try {
        await refreshJobs();
        $("connection").textContent = readyLabel;
        $("check-status").hidden = true;
      } catch (error) {
        $("connection").textContent = "Job status unavailable · check the server";
        showError(`${error.message} Select Check status to reconnect. A connection error does not cancel generation.`);
        $("check-status").hidden = false;
      }
    }, 2000);
  };
  const refreshJobs = async () => {
    $("refresh-jobs").disabled = true;
    try {
      const result = await api("/v1/videos?limit=32&order=desc");
      jobs = result.data;
      const latestWatching = jobs.find((job) => job.id === watching?.id);
      if (latestWatching) watching = latestWatching;
      renderHistory();
      if (latestWatching) renderWatching(latestWatching);
      syncGenerateButton();
      syncClipNav();
      schedulePoll();
    } catch (error) {
      $("history-status").textContent = `Could not load jobs. Check the server and select Refresh jobs. ${error.message}`;
      throw error;
    } finally {
      $("refresh-jobs").disabled = false;
    }
  };
  $("generate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!model || submitting) return;
    if (!$("prompt").value.trim()) {
      showError("Write a prompt before generating a video.");
      $("prompt").focus();
      return;
    }
    showError();
    submitting = true;
    syncGenerateButton();
    const keepClip = watching?.status === "completed";
    $("job-status").textContent = keepClip ? "Queuing the next clip…" : "Submitting the prompt…";
    try {
      const job = await api("/v1/videos", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()),
      });
      await refreshJobs();
      if (keepClip) {
        $("job-status").textContent = "Queued. This clip keeps playing while the server generates the next one.";
      } else {
        selectJob(job);
      }
    } catch (error) {
      showError(`${error.message} Check Recent jobs before submitting again; the server may have received the prompt. Submissions are never retried automatically.`);
      $("job-status").textContent = "Could not confirm the submission.";
      await refreshJobs().catch(() => undefined);
    } finally {
      submitting = false;
      syncGenerateButton();
    }
  });
  $("prompt").addEventListener("input", updateCurl);
  $("seed").addEventListener("input", updateCurl);
  $("refresh-jobs").addEventListener("click", () => refreshJobs().catch(() => undefined));
  $("older-clip").addEventListener("click", () => {
    const index = watchingIndex();
    if (index >= 0 && index < jobs.length - 1) selectJob(jobs[index + 1]);
  });
  $("newer-clip").addEventListener("click", () => {
    const index = watchingIndex();
    if (index > 0) selectJob(jobs[index - 1]);
  });
  $("check-status").addEventListener("click", () => {
    refreshJobs().catch(() => undefined);
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
        : "The model stays loaded until you stop the server. Queue the next prompt while a clip plays; closing this page does not cancel a job.";
      $("model").textContent = model;
      const d = config.defaults;
      const facts = [];
      if (d.width && d.height) facts.push(`${d.width} × ${d.height}`);
      if (d.num_frames) facts.push(`${d.num_frames} frames`);
      if (d.fps) facts.push(`${d.fps} fps`);
      if (d.seed != null) $("seed").placeholder = String(d.seed);
      $("settings").textContent = facts.length ? `Server defaults · ${facts.join(" · ")}` : "Resolution and sampling come from the server configuration.";
      syncGenerateButton();
      updateCurl();
      await refreshJobs().catch(() => undefined);
      const id = new URL(location.href).searchParams.get("job");
      const linked = jobs.find((job) => job.id === id);
      if (linked) {
        if (!$("prompt").value.trim() && linked.prompt) {
          $("prompt").value = linked.prompt;
          updateCurl();
        }
        selectJob(linked);
      }
    } catch (error) {
      $("connection").textContent = "Server unavailable";
      $("history-status").textContent = "Start the H3 server, then reload this page.";
      showError(`${error.message} Check the server terminal and reload this page when model loading completes.`);
    }
  };
  connect();
})();
