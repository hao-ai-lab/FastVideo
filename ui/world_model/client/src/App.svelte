<script>
  import { onMount, onDestroy } from 'svelte';

  let ws = null;
  let connected = false;
  let connecting = false;
  let sessionStarted = false;  // User has clicked "Join Session"
  let frameSrc = '';
  let frameCount = 0;
  let blockCount = 0;
  let maxBlocks = 50;
  let resetting = false;
  let sessionTimeout = null;
  let timeLeft = null;
  let countdownInterval = null;
  let queuePosition = 0;
  let gpuAssigned = false;
  let pressedKeys = {
    up: false,
    down: false,
    left: false,
    right: false
  };
  let pressedArrows = {
    up: false,
    down: false,
    left: false,
    right: false
  };

  let availableModels = [];
  let selectedModelId = '';

  let frameBuffer = [];
  let isPlayingBuffer = false;
  let animationFrameId = null;
  const TARGET_FPS = 24;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;
  const MIN_BUFFER_SIZE = 4;
  let lastFrameTime = 0;
  let canvas;
  let ctx;
  let loadingAnimation = false;

  let benchmarkStats = {
    decodeTime: 0,
    renderTime: 0,
    e2eLatency: 0,
    frameSize: 0
  };

  $: if (canvas) {
    ctx = canvas.getContext('2d', { alpha: false });
  }

  function playBufferedFrames() {
    if (isPlayingBuffer) return;
    if (frameBuffer.length < MIN_BUFFER_SIZE) return;

    isPlayingBuffer = true;
    lastFrameTime = performance.now();
    animationFrameId = requestAnimationFrame(playNextFrame);
  }

  let canvasInitialized = false;

  function playNextFrame(currentTime) {
    // If buffer is empty, stop but keep last frame visible
    if (frameBuffer.length === 0) {
      isPlayingBuffer = false;
      animationFrameId = null;
      return;
    }

    const elapsed = currentTime - lastFrameTime;

    if (elapsed >= FRAME_INTERVAL) {
      const bitmap = frameBuffer.shift();
      frameBuffer = frameBuffer;

      // Draw pre-decoded bitmap to canvas
      if (ctx && bitmap) {
        if (!canvasInitialized) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvasInitialized = true;
          loadingAnimation = false;
        }

        const renderStart = performance.now();
        ctx.drawImage(bitmap, 0, 0);
        benchmarkStats.renderTime = Math.round(performance.now() - renderStart);

        bitmap.close();
      }

      lastFrameTime = currentTime;
    }

    // Continue animation loop
    if (frameBuffer.length > 0) {
      animationFrameId = requestAnimationFrame(playNextFrame);
    } else {
      // Buffer empty - stop and wait for more frames
      isPlayingBuffer = false;
      animationFrameId = null;
    }
  }

  function joinSession() {
    sessionStarted = true;
    connectWebSocket();
  }

  function leaveSession() {
    sessionStarted = false;
    if (ws) {
      ws.close();
      ws = null;
    }
    connected = false;
    connecting = false;
    gpuAssigned = false;
    timeLeft = null;
    queuePosition = 0;
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
    // Clear canvas
    frameBuffer = [];
    canvasInitialized = false;
    loadingAnimation = false;
  }

  function connectWebSocket() {
    // Close any existing connection before creating a new one
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close();
      ws = null;
    }

    connecting = true;
    connected = false;

    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

      ws.onopen = () => {
        connected = true;
        connecting = false;
        // Send selected model to server
        ws.send(JSON.stringify({ type: 'select_model', model_id: selectedModelId }));
      };

      let pendingFrames = new Map(); // Track frames in decode order
      let nextFrameIndex = 0;
      let nextFrameToAdd = 0; // Track which frame should be added next

      ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'frame_batch') {
          const receiveTime = performance.now();
          const frames = data.frames;

          console.log(`Received batch of ${frames.length} frames`);

          // Measure first frame size
          benchmarkStats.frameSize = Math.round(frames[0].length / 1024);

          // Decode all frames in parallel
          const decodeStart = performance.now();
          const decodePromises = frames.map((frameData, i) => {
            const base64Data = `data:image/jpeg;base64,${frameData}`;
            const frameIndex = nextFrameIndex++;

            return fetch(base64Data)
              .then(res => res.blob())
              .then(blob => createImageBitmap(blob))
              .then(bitmap => ({ frameIndex, bitmap }));
          });

          Promise.all(decodePromises).then(decodedFrames => {
            const decodeEnd = performance.now();
            benchmarkStats.decodeTime = Math.round((decodeEnd - decodeStart) / frames.length);

            decodedFrames.forEach(({ frameIndex, bitmap }) => {
              pendingFrames.set(frameIndex, bitmap);
            });

            // Add all consecutive decoded frames to buffer in order
            let addedFrames = false;
            while (pendingFrames.has(nextFrameToAdd)) {
              const orderedBitmap = pendingFrames.get(nextFrameToAdd);
              frameBuffer.push(orderedBitmap);
              pendingFrames.delete(nextFrameToAdd);
              nextFrameToAdd++;
              addedFrames = true;
            }

            if (addedFrames) {
              frameBuffer = frameBuffer;
              playBufferedFrames();
            }

            console.log(`Decoded and buffered ${decodedFrames.length} frames in ${decodeEnd - decodeStart}ms`);
          });

          frameCount += frames.length;
        } else if (data.type === 'frame') {
          const receiveTime = performance.now();

          const base64Data = `data:image/jpeg;base64,${data.data}`;
          const frameIndex = nextFrameIndex++;

          benchmarkStats.frameSize = Math.round(data.data.length / 1024);

          const decodeStart = performance.now();
          fetch(base64Data)
            .then(res => res.blob())
            .then(blob => createImageBitmap(blob))
            .then(bitmap => {
              const decodeEnd = performance.now();
              benchmarkStats.decodeTime = Math.round(decodeEnd - decodeStart);

              if (data.timestamp) {
                benchmarkStats.e2eLatency = Math.round(receiveTime - (data.timestamp * 1000));
              }

              pendingFrames.set(frameIndex, bitmap);

              let addedFrames = false;
              while (pendingFrames.has(nextFrameToAdd)) {
                const orderedBitmap = pendingFrames.get(nextFrameToAdd);
                frameBuffer.push(orderedBitmap);
                pendingFrames.delete(nextFrameToAdd);
                nextFrameToAdd++;
                addedFrames = true;
              }

              if (addedFrames) {
                frameBuffer = frameBuffer;
                playBufferedFrames();
              }
            });

          frameCount++;
        } else if (data.type === 'block_count') {
          blockCount = data.count;
          maxBlocks = data.max;
        } else if (data.type === 'reset_started') {
          console.log('Reset started on server');
          // Clear frame buffer on reset
          frameBuffer = [];
          pendingFrames.clear();
          nextFrameIndex = 0;
          nextFrameToAdd = 0;
          isPlayingBuffer = false;
          canvasInitialized = false;
          loadingAnimation = false;
        } else if (data.type === 'reset_complete') {
          console.log('Reset complete');
          resetting = false;
          frameCount = 0;
        } else if (data.type === 'queue_status') {
          queuePosition = data.position;
          console.log(`Queue position: ${queuePosition}`);
        } else if (data.type === 'gpu_assigned') {
          gpuAssigned = true;
          resetting = false;
          queuePosition = 0;
          sessionTimeout = data.session_timeout;
          timeLeft = data.session_timeout;
          console.log(`GPU ${data.gpu_id} assigned, session timeout: ${sessionTimeout}s`);

          // Load initial image onto canvas with animation
          if (data.image_url && canvas) {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
              if (!canvasInitialized) {
                canvas.width = img.width;
                canvas.height = img.height;
                if (ctx) {
                  ctx.drawImage(img, 0, 0);
                }
                loadingAnimation = true;
              }
            };
            img.src = data.image_url;
          }

          // Start countdown timer
          if (countdownInterval) clearInterval(countdownInterval);
          countdownInterval = setInterval(() => {
            if (timeLeft > 0) {
              timeLeft--;
            } else {
              clearInterval(countdownInterval);
            }
          }, 1000);
        } else if (data.type === 'session_timeout') {
          console.log('Session timed out');
          timeLeft = 0;
          gpuAssigned = false;
          if (countdownInterval) clearInterval(countdownInterval);
        }
      };

      ws.onerror = () => {
        // Let onclose handle retry
      };

      ws.onclose = () => {
        // Always reset to lobby state on close
        sessionStarted = false;
        connected = false;
        connecting = false;
        gpuAssigned = false;
        timeLeft = null;
        queuePosition = 0;
        if (countdownInterval) {
          clearInterval(countdownInterval);
          countdownInterval = null;
        }
      };
    } catch (error) {
      // Connection failed - reset to lobby
      sessionStarted = false;
      connected = false;
      connecting = false;
    }
  }

  function sendInput(key) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ key }));
    }
  }

  function handleReset() {
    resetting = true;

    // Close existing connection without triggering onclose handler
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close();
      ws = null;
    }

    // Reset session state but stay in session view
    connected = false;
    connecting = false;
    gpuAssigned = false;
    timeLeft = null;
    queuePosition = 0;
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
    frameBuffer = [];
    canvasInitialized = false;
    loadingAnimation = false;

    // Reconnect
    setTimeout(connectWebSocket, 100);
  }

  function handleKeyDown(event) {
    const validKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd'];
    if (validKeys.includes(event.key)) {
      event.preventDefault();

      if (event.key === 'w') pressedKeys.up = true;
      if (event.key === 's') pressedKeys.down = true;
      if (event.key === 'a') pressedKeys.left = true;
      if (event.key === 'd') pressedKeys.right = true;

      if (event.key === 'ArrowUp') pressedArrows.up = true;
      if (event.key === 'ArrowDown') pressedArrows.down = true;
      if (event.key === 'ArrowLeft') pressedArrows.left = true;
      if (event.key === 'ArrowRight') pressedArrows.right = true;

      sendInput(event.key);
    }
  }

  function handleKeyUp(event) {
    if (event.key === 'w') pressedKeys.up = false;
    if (event.key === 's') pressedKeys.down = false;
    if (event.key === 'a') pressedKeys.left = false;
    if (event.key === 'd') pressedKeys.right = false;

    if (event.key === 'ArrowUp') pressedArrows.up = false;
    if (event.key === 'ArrowDown') pressedArrows.down = false;
    if (event.key === 'ArrowLeft') pressedArrows.left = false;
    if (event.key === 'ArrowRight') pressedArrows.right = false;
  }

  onMount(async () => {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Fetch available models
    try {
      const res = await fetch('/models');
      const data = await res.json();
      availableModels = data.models;
      selectedModelId = data.default_model_id;
    } catch (e) {
      console.error('Failed to fetch models:', e);
    }
  });

  onDestroy(() => {
    if (countdownInterval) clearInterval(countdownInterval);
    if (ws) ws.close();
    window.removeEventListener('keydown', handleKeyDown);
    window.removeEventListener('keyup', handleKeyUp);
  });

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
</script>

<main>
  <div class="page-header">
    <img src="/logo.svg" alt="FastVideo" class="logo" />
    <div class="page-subtitle">
      Real-time World Models powered by FastVideo
    </div>
    <div class="page-links">
      <a href="https://github.com/hao-ai-lab/FastVideo" target="_blank">Code</a>
      <span class="separator">|</span>
      <a href="https://hao-ai-lab.github.io/blogs/" target="_blank">Blog</a>
      <span class="separator">|</span>
      <a href="https://hao-ai-lab.github.io/FastVideo/" target="_blank">Docs</a>
    </div>
  </div>

    <details class="about-section">
    <summary>About FastVideo</summary>
    <div class="about-content">
      <p>
        FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end
        unified pipeline for accelerating diffusion models, starting from data preprocessing to model training,
        finetuning, distillation, and inference.
      </p>
      <p>
        FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations
        and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo
        has you covered.
      </p>
    </div>
  </details>

  <details class="about-section">
    <summary>What is Matrix-Game?</summary>
    <div class="about-content">
      <p>
        Matrix-Game 2.0 is an interactive world model that generates video frames in real-time
        based on your keyboard inputs.
      </p>
      <p>
        Powered by <strong>FastVideo</strong>'s streaming inference pipeline, this demo showcases
        low-latency video generation with diffusion models optimized for interactive applications.
      </p>
    </div>
  </details>

  <div class="header-section">
    <div class="model-selector">
      {#if availableModels.length > 0}
        <select bind:value={selectedModelId} disabled={sessionStarted}>
          {#each availableModels as model}
            <option value={model.id}>{model.name}</option>
          {/each}
        </select>
      {:else}
        <h2>Loading...</h2>
      {/if}
    </div>
    <div class="status">
      {#if connected && gpuAssigned && timeLeft !== null}
        <span class="time-left" class:warning={timeLeft <= 30}>
          Time left: {formatTime(timeLeft)}
        </span>
      {:else if sessionStarted && queuePosition > 0}
        <span class="status-queue">Queue position: {queuePosition}</span>
      {:else if sessionStarted && (connecting || resetting)}
        <span class="status-connecting">Connecting...</span>
      {/if}
    </div>
    <div class="header-buttons">
      {#if !sessionStarted}
        <button class="join-btn" on:click={joinSession}>
          Join Session
        </button>
      {:else}
        <button class="reset-btn" on:click={handleReset} disabled={!connected || !gpuAssigned || resetting}>
          {#if resetting}
            <span class="spinner">⟳</span> Resetting...
          {:else}
            Reset
          {/if}
        </button>
        <button class="leave-btn" on:click={leaveSession} disabled={resetting}>
          Leave
        </button>
      {/if}
    </div>
  </div>

  <div class="game-container">
    <!-- <div class="benchmark">
      Decode: {benchmarkStats.decodeTime}ms
      | Render: {benchmarkStats.renderTime}ms
      | Frame: {benchmarkStats.frameSize}KB
    </div> -->

    <div class="canvas">
    <canvas bind:this={canvas} class:blurred={resetting} class:loading-animation={loadingAnimation} style="display: block; max-width: 100%; height: auto;"></canvas>
    {#if !sessionStarted}
      <div class="placeholder">Click "Join Session" to start</div>
    {:else if !loadingAnimation && (!canvas || (frameBuffer.length === 0 && !isPlayingBuffer && !canvasInitialized))}
      <div class="placeholder">Waiting for frame...</div>
    {/if}

    {#if blockCount >= maxBlocks}
      <div class="max-blocks-overlay">
        <div class="overlay-content">
          <h2>50 Blocks Reached!</h2>
          <p>The generator has completed 50 blocks.</p>
          <p>Click "Reset Model" to start a new session.</p>
        </div>
      </div>
    {/if}

    {#if timeLeft === 0 && gpuAssigned}
      <div class="max-blocks-overlay">
        <div class="overlay-content timeout-overlay">
          <h2>Session Expired</h2>
          <p>Your 90-second session has ended.</p>
          <p>Reconnecting to get a new session...</p>
        </div>
      </div>
    {/if}

    {#if sessionStarted && connected && queuePosition > 0 && !gpuAssigned}
      <div class="max-blocks-overlay">
        <div class="overlay-content queue-overlay">
          <h2>In Queue</h2>
          <p>All GPUs are currently busy.</p>
          <p>Your position: <strong>{queuePosition}</strong></p>
          <p>Please wait for a GPU to become available.</p>
        </div>
      </div>
    {/if}

    <div class="key-overlay key-overlay-left">
      <div class="key-grid">
        <div class="key-spacer"></div>
        <div class="key {pressedKeys.up ? 'pressed' : ''}">W</div>
        <div class="key-spacer"></div>
        <div class="key {pressedKeys.left ? 'pressed' : ''}">A</div>
        <div class="key {pressedKeys.down ? 'pressed' : ''}">S</div>
        <div class="key {pressedKeys.right ? 'pressed' : ''}">D</div>
      </div>
    </div>

    <div class="key-overlay key-overlay-right">
      <div class="key-grid">
        <div class="key-spacer"></div>
        <div class="key {pressedArrows.up ? 'pressed' : ''}">▲</div>
        <div class="key-spacer"></div>
        <div class="key {pressedArrows.left ? 'pressed' : ''}">◀</div>
        <div class="key {pressedArrows.down ? 'pressed' : ''}">▼</div>
        <div class="key {pressedArrows.right ? 'pressed' : ''}">▶</div>
      </div>
    </div>
  </div>

    <div class="instructions">
      WASD: Movement | Arrow Keys: Camera
    </div>
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    overflow-y: auto;
    background: #1a1a1a;
  }

  main {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 2rem;
    background: #1a1a1a;
    color: white;
    min-height: 100vh;
  }

  .page-header {
    text-align: center;
    margin-bottom: 1rem;
  }

  .logo {
    height: 80px;
    width: auto;
    margin-bottom: 0.5rem;
  }

  .page-subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    margin-bottom: 1rem;
  }

  .page-links {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    align-items: center;
    font-size: 1rem;
  }

  .page-links a {
    color: #60a5fa;
    text-decoration: none;
    transition: color 0.2s;
  }

  .page-links a:hover {
    color: #93c5fd;
  }

  .page-links .separator {
    color: #4b5563;
  }

  .about-section {
    width: 100%;
    max-width: 640px;
    margin-bottom: 0.5rem;
    background: rgba(31, 41, 55, 0.5);
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 1rem;
  }

  .about-section summary {
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    color: #d1d5db;
    user-select: none;
    list-style: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .about-section summary::-webkit-details-marker {
    display: none;
  }

  .about-section summary::after {
    content: "▶";
    font-size: 0.8rem;
    color: #9ca3af;
    transition: transform 0.2s ease;
  }

  .about-section[open] summary::after {
    transform: rotate(90deg);
  }

  .about-section[open] summary {
    margin-bottom: 1rem;
  }

  .about-content {
    padding: 0.5rem 0;
    line-height: 1.6;
    color: #9ca3af;
  }

  .about-content p {
    margin: 0 0 1rem 0;
  }

  .about-content p:last-child {
    margin-bottom: 0;
  }

  .about-content strong {
    color: #d1d5db;
  }

  .header-section {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    width: 100%;
    max-width: 672px;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
  }

  .model-selector {
    justify-self: start;
  }

  .model-selector select {
    background: #1f2937;
    color: white;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    appearance: auto;
  }

  .model-selector select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .model-selector select:hover:not(:disabled) {
    border-color: #60a5fa;
  }

  .header-section h2 {
    justify-self: start;
  }

  .header-section .status {
    justify-self: center;
  }

  .header-buttons {
    justify-self: end;
    display: flex;
    gap: 0.5rem;
  }

  .game-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 1rem;
    background: rgba(31, 41, 55, 0.5);
  }

  h1 {
    font-size: 2rem;
    margin: 0;
  }

  h2 {
    font-size: 1.5rem;
    margin: 0;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .reset-btn {
    padding: 0.5rem 1rem;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
    white-space: nowrap;
  }

  .reset-btn:hover:not(:disabled) {
    background: #1d4ed8;
  }

  .reset-btn:disabled {
    background: #374151;
    cursor: not-allowed;
    opacity: 0.6;
  }

  .join-btn {
    padding: 0.5rem 1.5rem;
    background: #10b981;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .join-btn:hover {
    background: #059669;
  }

  .leave-btn {
    padding: 0.5rem 1rem;
    background: #6b7280;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .leave-btn:hover:not(:disabled) {
    background: #4b5563;
  }

  .leave-btn:disabled {
    background: #374151;
    cursor: not-allowed;
    opacity: 0.6;
  }

  .spinner {
    display: inline-block;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .status {
    font-size: 1rem;
    color: #aaa;
  }

  .benchmark {
    font-size: 0.9rem;
    color: #888;
    font-family: monospace;
  }

  .status-connected {
    color: #4ade80;
  }

  .status-connecting {
    color: #facc15;
  }

  .status-disconnected {
    color: #f87171;
  }

  .status-idle {
    color: #9ca3af;
  }

  .status-queue {
    color: #facc15;
  }

  .time-left {
    color: #4ade80;
    font-family: monospace;
    font-weight: 600;
  }

  .time-left.warning {
    color: #f87171;
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.6;
    }
  }

  .canvas {
    position: relative;
    width: 640px;
    height: 352px;
    background: #000;
    border: 2px solid #333;
    border-radius: 8px;
    overflow: hidden;
    flex-shrink: 0;
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  canvas.blurred {
    filter: blur(8px);
    transition: filter 0.3s ease;
  }

  @keyframes warp-in {
    0%   { transform: scale(1.05); filter: blur(6px) brightness(0.5); }
    50%  { transform: scale(1.02); filter: blur(3px) brightness(0.8); }
    100% { transform: scale(1.0);  filter: blur(0px) brightness(1.0); }
  }

  canvas.loading-animation {
    animation: warp-in 1.5s ease-out forwards;
  }

  .placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #666;
  }

  .instructions {
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.5rem;
  }

  .key-overlay {
    position: absolute;
    bottom: 16px;
    pointer-events: none;
  }

  .key-overlay-left {
    left: 16px;
  }

  .key-overlay-right {
    right: 16px;
  }

  .key-grid {
    display: grid;
    grid-template-columns: repeat(3, 40px);
    grid-template-rows: repeat(2, 40px);
    gap: 4px;
  }

  .key-spacer {
    background: transparent;
  }

  .key {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.6);
    font-size: 20px;
    transition: all 0.1s ease;
    backdrop-filter: blur(4px);
  }

  .key.pressed {
    background: rgba(255, 255, 255, 0.4);
    border-color: rgba(255, 255, 255, 0.8);
    color: rgba(255, 255, 255, 1);
    transform: scale(0.95);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  .max-blocks-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    backdrop-filter: blur(4px);
  }

  .overlay-content {
    text-align: center;
    padding: 2rem;
    background: rgba(30, 30, 30, 0.95);
    border-radius: 12px;
    border: 2px solid #fbbf24;
    max-width: 400px;
  }

  .overlay-content h2 {
    font-size: 1.5rem;
    margin: 0 0 1rem 0;
    color: #fbbf24;
  }

  .overlay-content p {
    font-size: 1rem;
    margin: 0.5rem 0;
    color: #d1d5db;
    line-height: 1.5;
  }

  .timeout-overlay {
    border-color: #f87171;
  }

  .timeout-overlay h2 {
    color: #f87171;
  }

  .queue-overlay {
    border-color: #60a5fa;
  }

  .queue-overlay h2 {
    color: #60a5fa;
  }

  .queue-overlay strong {
    color: #facc15;
    font-size: 1.5rem;
  }
</style>
