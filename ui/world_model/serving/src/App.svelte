<script>
  import { onMount, onDestroy } from 'svelte';

  let ws = null;
  let connected = false;
  let connecting = false;
  let frameSrc = '';
  let frameCount = 0;
  let reconnectTimer = null;
  let connectionTimeout = null;
  let pressedKeys = {
    up: false,
    down: false,
    left: false,
    right: false
  };

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

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'frame') {
          frameSrc = `data:image/jpeg;base64,${data.data}`;
          frameCount++;
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

  function handleKeyDown(event) {
    const validKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd'];
    if (validKeys.includes(event.key)) {
      event.preventDefault();

      // Update pressed state
      if (event.key === 'w' || event.key === 'ArrowUp') pressedKeys.up = true;
      if (event.key === 's' || event.key === 'ArrowDown') pressedKeys.down = true;
      if (event.key === 'a' || event.key === 'ArrowLeft') pressedKeys.left = true;
      if (event.key === 'd' || event.key === 'ArrowRight') pressedKeys.right = true;

      sendInput(event.key);
    }
  }

  function handleKeyUp(event) {
    if (event.key === 'w' || event.key === 'ArrowUp') pressedKeys.up = false;
    if (event.key === 's' || event.key === 'ArrowDown') pressedKeys.down = false;
    if (event.key === 'a' || event.key === 'ArrowLeft') pressedKeys.left = false;
    if (event.key === 'd' || event.key === 'ArrowRight') pressedKeys.right = false;
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
  <h1>Matrix Game</h1>
  <div class="status">
    {#if connected}
      <span class="status-connected">Connected</span>
    {:else if connecting}
      <span class="status-connecting">Connecting...</span>
    {:else}
      <span class="status-disconnected">Disconnected</span>
    {/if}
    | Frames: {frameCount}
  </div>

  <div class="canvas">
    {#if frameSrc}
      <img src={frameSrc} alt="Frame" />
    {:else}
      <div class="placeholder">Waiting for frame...</div>
    {/if}

    <div class="key-overlay">
      <div class="key-grid">
        <div class="key-spacer"></div>
        <div class="key {pressedKeys.up ? 'pressed' : ''}">▲</div>
        <div class="key-spacer"></div>
        <div class="key {pressedKeys.left ? 'pressed' : ''}">◀</div>
        <div class="key {pressedKeys.down ? 'pressed' : ''}">▼</div>
        <div class="key {pressedKeys.right ? 'pressed' : ''}">▶</div>
      </div>
    </div>
  </div>

  <div class="instructions">
    Use WASD or Arrow Keys to control
  </div>
</main>

<style>
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

  h1 {
    font-size: 2rem;
  }

  .status {
    font-size: 1rem;
    color: #aaa;
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
    left: 16px;
    pointer-events: none;
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
</style>
