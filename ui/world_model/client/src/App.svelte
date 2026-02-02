<script>
  import { onMount, onDestroy } from 'svelte';

  let ws = null;
  let connected = false;
  let connecting = false;
  let frameSrc = '';
  let frameCount = 0;
  let blockCount = 0;
  let maxBlocks = 50;
  let reconnectTimer = null;
  let connectionTimeout = null;
  let resetting = false;
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

  let frameBuffer = [];
  let isPlayingBuffer = false;
  let animationFrameId = null;
  const TARGET_FPS = 24;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;
  const MIN_BUFFER_SIZE = 4;
  let lastFrameTime = 0;
  let canvas;
  let ctx;

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

  function scheduleReconnect() {
    if (reconnectTimer) return; // Already scheduled
    if (connectionTimeout) clearTimeout(connectionTimeout);

    connected = false;
    connecting = false;
    reconnectTimer = setTimeout(connectWebSocket, 2000);
  }

  function connectWebSocket() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (connectionTimeout) clearTimeout(connectionTimeout);
    reconnectTimer = null;
    connectionTimeout = null;

    connecting = true;
    connected = false;

    // Timeout if not connected within 5s
    connectionTimeout = setTimeout(() => {
      if (!connected && connecting && ws) {
        ws.close();
      }
    }, 5000);

    try {
      ws = new WebSocket('ws://localhost:8000/ws');

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        connected = true;
        connecting = false;
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
        } else if (data.type === 'reset_complete') {
          console.log('Reset complete');
          resetting = false;
          frameCount = 0;
        }
      };

      ws.onerror = () => {
        // Let onclose handle retry
      };

      ws.onclose = () => {
        clearTimeout(connectionTimeout);
        scheduleReconnect();
      };
    } catch (error) {
      scheduleReconnect();
    }
  }

  function sendInput(key) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ key }));
    }
  }

  function handleReset() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      resetting = true;
      ws.send(JSON.stringify({ type: 'reset' }));
    }
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

  onMount(() => {
    connectWebSocket();
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
  });

  onDestroy(() => {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (connectionTimeout) clearTimeout(connectionTimeout);
    if (ws) ws.close();
    window.removeEventListener('keydown', handleKeyDown);
    window.removeEventListener('keyup', handleKeyUp);
  });
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
    <h2>Matrix-Game 2.0</h2>
    <div class="status">
      {#if connected}
        <span class="status-connected">Connected</span>
      {:else if connecting}
        <span class="status-connecting">Connecting...</span>
      {:else}
        <span class="status-disconnected">Disconnected</span>
      {/if}
      <!-- | Blocks: {blockCount} / {maxBlocks} -->
      <!-- | Frames: {frameCount}
      | Buffer: {frameBuffer.length} -->
    </div>
    <button class="reset-btn" on:click={handleReset} disabled={!connected || resetting}>
      {#if resetting}
        <span class="spinner">⟳</span> Resetting...
      {:else}
        Reset Model
      {/if}
    </button>
  </div>

  <div class="game-container">
    <!-- <div class="benchmark">
      Decode: {benchmarkStats.decodeTime}ms
      | Render: {benchmarkStats.renderTime}ms
      | Frame: {benchmarkStats.frameSize}KB
    </div> -->

    <div class="canvas">
    <canvas bind:this={canvas} style="display: block; max-width: 100%; height: auto;"></canvas>
    {#if !canvas || frameBuffer.length === 0 && !isPlayingBuffer}
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

  .header-section h2 {
    justify-self: start;
  }

  .header-section .status {
    justify-self: center;
  }

  .header-section .reset-btn {
    justify-self: end;
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
</style>
