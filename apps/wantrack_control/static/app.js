(() => {
  const $ = (id) => document.getElementById(id);
  const imageInput = $("image");
  const promptInput = $("prompt");
  const prepareButton = $("prepare");
  const canvas = $("canvas");
  const context = canvas.getContext("2d");
  const addButton = $("add");
  const removeButton = $("remove");
  const gridInput = $("grid");
  const radiusInput = $("radius");
  const radiusOutput = document.querySelector(".radius output");
  const startButton = $("start");
  const stopButton = $("stop");
  const video = $("video");
  const status = $("status");
  const download = $("download");

  let socket;
  let preparedImage;
  let handles = [];
  let selectedId = null;
  let addMode = false;
  let dragging = false;
  let generating = false;
  let revision = 0;
  let sessionStart = 0;
  let mediaSource;
  let sourceBuffer;
  let mediaQueue = [];
  let streamComplete = false;

  function setStatus(value) { status.textContent = value; }
  function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => setStatus("Ready");
    socket.onclose = () => setStatus("Disconnected");
    socket.onerror = () => setStatus("Connection error");
    socket.onmessage = async (event) => {
      if (typeof event.data !== "string") {
        mediaQueue.push(event.data);
        flushMedia();
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === "prepared") {
        preparedImage = new Image();
        preparedImage.onload = () => {
          canvas.width = message.width;
          canvas.height = message.height;
          draw();
        };
        preparedImage.src = message.image;
        handles = [];
        selectedId = null;
        addButton.disabled = false;
        startButton.disabled = true;
        setStatus(`Prepared · ${message.fps} FPS`);
      } else if (message.type === "session_started") {
        generating = true;
        sessionStart = performance.now();
        startButton.disabled = true;
        stopButton.disabled = false;
        setStatus("Generating");
      } else if (message.type === "block_started") {
        setStatus(`Generating block ${message.block_index}`);
      } else if (message.type === "control_applied") {
        if (message.status === "ignored_stale") {
          setStatus(`Stale revision ${message.revision} ignored`);
        } else {
          setStatus(`Control ${message.revision} applied`);
        }
      } else if (message.type === "media_init") {
        setupMediaSource(message.mime);
      } else if (message.type === "media_segment_complete") {
        setStatus(`Playing through block ${message.block_index}`);
      } else if (message.type === "stream_complete") {
        generating = false;
        streamComplete = true;
        stopButton.disabled = true;
        startButton.disabled = false;
        if (message.download_url) {
          download.href = message.download_url;
          download.hidden = false;
        }
        flushMedia();
        setStatus(`Complete · ${message.blocks} blocks`);
      } else if (message.type === "error") {
        generating = false;
        stopButton.disabled = true;
        if (message.download_url) {
          download.href = message.download_url;
          download.hidden = false;
        }
        setStatus(message.message || "Error");
      }
    };
  }

  function setupMediaSource(mime) {
    mediaQueue = [];
    streamComplete = false;
    mediaSource = new MediaSource();
    video.src = URL.createObjectURL(mediaSource);
    mediaSource.addEventListener("sourceopen", () => {
      sourceBuffer = mediaSource.addSourceBuffer(mime);
      sourceBuffer.mode = "sequence";
      sourceBuffer.addEventListener("updateend", flushMedia);
      flushMedia();
    }, { once: true });
  }

  function flushMedia() {
    if (!sourceBuffer || sourceBuffer.updating) return;
    if (mediaQueue.length) {
      sourceBuffer.appendBuffer(mediaQueue.shift());
      return;
    }
    if (streamComplete && mediaSource && mediaSource.readyState === "open") {
      mediaSource.endOfStream();
    }
    video.play().catch(() => {});
  }

  function draw() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (preparedImage) context.drawImage(preparedImage, 0, 0, canvas.width, canvas.height);
    if (gridInput.checked) {
      context.strokeStyle = "rgba(255,255,255,.12)";
      context.lineWidth = 1;
      for (let index = 0; index < 50; index += 1) {
        const x = index * canvas.width / 49;
        const y = index * canvas.height / 49;
        context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
        context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
      }
    }
    for (const handle of handles) {
      const x = handle.x * canvas.width;
      const y = handle.y * canvas.height;
      context.beginPath();
      context.arc(x, y, handle.id === selectedId ? 9 : 7, 0, Math.PI * 2);
      context.fillStyle = handle.id === selectedId ? "#ffcf4a" : "#58c7ff";
      context.fill();
      context.strokeStyle = "#111";
      context.lineWidth = 2;
      context.stroke();
    }
    removeButton.disabled = !selectedId;
    startButton.disabled = !preparedImage || handles.length === 0 || generating;
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    };
  }
  function nearest(point) {
    let match = null;
    let distance = Infinity;
    for (const handle of handles) {
      const value = Math.hypot(handle.x - point.x, handle.y - point.y);
      if (value < distance && value < 18 / canvas.clientWidth) {
        match = handle;
        distance = value;
      }
    }
    return match;
  }
  function sendControl(extra = {}) {
    if (!generating) return;
    revision += 1;
    socket.send(JSON.stringify({ type: "control_update", revision, ...extra }));
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (!preparedImage) return;
    const point = canvasPoint(event);
    if (addMode) {
      const handle = { id: crypto.randomUUID(), ...point };
      handles.push(handle);
      selectedId = handle.id;
      addMode = false;
      addButton.textContent = "Add handle";
      sendControl({ add: [handle] });
      draw();
      return;
    }
    const handle = nearest(point);
    selectedId = handle ? handle.id : null;
    dragging = Boolean(handle);
    if (dragging) canvas.setPointerCapture(event.pointerId);
    draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging || !selectedId) return;
    const point = canvasPoint(event);
    const handle = handles.find((item) => item.id === selectedId);
    if (!handle) return;
    handle.x = point.x; handle.y = point.y;
    sendControl({ samples: [{ id: handle.id, ...point, timestamp_ms: performance.now() - sessionStart }] });
    draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  addButton.addEventListener("click", () => {
    addMode = !addMode;
    addButton.textContent = addMode ? "Click canvas" : "Add handle";
  });
  removeButton.addEventListener("click", () => {
    if (!selectedId) return;
    const removed = selectedId;
    handles = handles.filter((item) => item.id !== removed);
    selectedId = null;
    sendControl({ remove: [removed] });
    draw();
  });
  gridInput.addEventListener("change", draw);
  radiusInput.addEventListener("input", () => {
    radiusOutput.textContent = Number(radiusInput.value).toFixed(2);
    sendControl({ radius: Number(radiusInput.value) });
  });

  prepareButton.addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) { setStatus("Choose an image"); return; }
    setStatus("Preparing");
    const data = await file.arrayBuffer();
    let binary = "";
    for (const value of new Uint8Array(data)) binary += String.fromCharCode(value);
    socket.send(JSON.stringify({
      type: "prepare",
      image: btoa(binary),
      prompt: promptInput.value,
    }));
  });
  startButton.addEventListener("click", () => {
    revision = 0;
    download.hidden = true;
    socket.send(JSON.stringify({
      type: "start",
      handles,
      radius: Number(radiusInput.value),
      seed: Number($("seed").value),
      steps: Number($("steps").value),
      text_guidance: Number($("text-guidance").value),
      motion_guidance: Number($("motion-guidance").value),
    }));
  });
  stopButton.addEventListener("click", () => {
    socket.send(JSON.stringify({ type: "stop" }));
    stopButton.disabled = true;
    setStatus("Finishing current block");
  });
  connect();
})();
