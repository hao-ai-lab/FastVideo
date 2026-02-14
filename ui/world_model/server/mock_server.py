"""
Mock server to test latency perception.
Generates frames with a moving object based on keyboard input.
Configurable latency to find the acceptable threshold.
"""

import asyncio
import base64
import io
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import os

# Configuration
LATENCY_MS = 200  # Simulated model latency - adjust this to test
FRAME_WIDTH = 640
FRAME_HEIGHT = 352
NUM_FRAMES = 12  # Frames per block
FPS = 24
BATCH_SIZE = 4
JPEG_QUALITY = 85
SESSION_TIMEOUT_SECONDS = 90
MAX_BLOCKS = 50

# Mock game state
class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = FRAME_WIDTH // 2
        self.y = FRAME_HEIGHT // 2
        self.size = 40
        self.color = (100, 200, 255)  # Light blue
        self.trail = []  # Previous positions for motion blur effect

    def update(self, keyboard_vector, mouse_vector):
        """Store the movement direction for this block."""
        self.keyboard_vector = keyboard_vector
        self.mouse_vector = mouse_vector

        # Arrow keys for camera (just change color as visual feedback)
        if mouse_vector[0] != 0 or mouse_vector[1] != 0:
            r = int(100 + mouse_vector[0] * 50) % 256
            g = int(200 + mouse_vector[1] * 50) % 256
            self.color = (r, g, 255)

    def render_frame(self, frame_idx: int) -> np.ndarray:
        """Render a single frame with incremental movement."""
        speed_per_frame = 8

        # Apply movement for this frame
        if hasattr(self, 'keyboard_vector'):
            if self.keyboard_vector[0]:  # W - up
                self.y -= speed_per_frame
            if self.keyboard_vector[1]:  # A - left
                self.x -= speed_per_frame
            if self.keyboard_vector[2]:  # S - down
                self.y += speed_per_frame
            if self.keyboard_vector[3]:  # D - right
                self.x += speed_per_frame

        # Wrap around screen
        self.x = self.x % FRAME_WIDTH
        self.y = self.y % FRAME_HEIGHT

        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 5:
            self.trail.pop(0)
        """Render a single frame."""
        # Create frame with gradient background
        img = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), (20, 20, 30))
        draw = ImageDraw.Draw(img)

        # Draw grid for visual reference
        for x in range(0, FRAME_WIDTH, 40):
            draw.line([(x, 0), (x, FRAME_HEIGHT)], fill=(40, 40, 50), width=1)
        for y in range(0, FRAME_HEIGHT, 40):
            draw.line([(0, y), (FRAME_WIDTH, y)], fill=(40, 40, 50), width=1)

        # Interpolate position for smooth animation within block
        t = frame_idx / NUM_FRAMES

        # Draw trail (motion blur effect)
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 0.3
            trail_size = int(self.size * (0.5 + alpha * 0.5))
            trail_color = tuple(int(c * alpha) for c in self.color)
            draw.ellipse(
                [tx - trail_size//2, ty - trail_size//2,
                 tx + trail_size//2, ty + trail_size//2],
                fill=trail_color
            )

        # Draw main object (circle with glow)
        # Outer glow
        for r in range(3, 0, -1):
            glow_size = self.size + r * 8
            glow_alpha = 0.2 / r
            glow_color = tuple(int(c * glow_alpha) for c in self.color)
            draw.ellipse(
                [self.x - glow_size//2, self.y - glow_size//2,
                 self.x + glow_size//2, self.y + glow_size//2],
                fill=glow_color
            )

        # Main circle
        draw.ellipse(
            [self.x - self.size//2, self.y - self.size//2,
             self.x + self.size//2, self.y + self.size//2],
            fill=self.color
        )

        # Inner highlight
        highlight_size = self.size // 3
        highlight_offset = self.size // 6
        draw.ellipse(
            [self.x - highlight_offset - highlight_size//2,
             self.y - highlight_offset - highlight_size//2,
             self.x - highlight_offset + highlight_size//2,
             self.y - highlight_offset + highlight_size//2],
            fill=(255, 255, 255)
        )

        # Add latency indicator text
        draw.text((10, 10), f"Latency: {LATENCY_MS}ms", fill=(150, 150, 150))
        draw.text((10, 30), f"Frame: {frame_idx + 1}/{NUM_FRAMES}", fill=(100, 100, 100))

        return np.array(img)


def encode_frame(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_frames(game_state: GameState) -> list[str]:
    """Generate and encode a block of frames."""
    encoded = []
    for i in range(NUM_FRAMES):
        frame = game_state.render_frame(i)
        encoded.append(encode_frame(frame))
    return encoded


# Keyboard/camera mappings (same as real server)
KEYBOARD_MAP = {
    'w': [1, 0, 0, 0],
    'a': [0, 1, 0, 0],
    's': [0, 0, 1, 0],
    'd': [0, 0, 0, 1],
}

CAMERA_MAP = {
    'ArrowUp': [0, -1],
    'ArrowDown': [0, 1],
    'ArrowLeft': [-1, 0],
    'ArrowRight': [1, 0],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Mock server starting with {LATENCY_MS}ms simulated latency...")
    yield
    print("Mock server shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_frames(websocket: WebSocket, frames: list[str]):
    """Send frames in batches."""
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        await websocket.send_json({"type": "frame_batch", "frames": batch})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = str(uuid.uuid4())
    print(f"Client {client_id[:8]} connected")

    game_state = GameState()
    block_count = 0
    timeout_task = None

    async def session_timeout():
        await asyncio.sleep(SESSION_TIMEOUT_SECONDS)
        try:
            await websocket.send_json({
                "type": "session_timeout",
                "message": f"Session expired after {SESSION_TIMEOUT_SECONDS} seconds"
            })
            await websocket.close()
        except Exception:
            pass

    try:
        # Send initial status
        await websocket.send_json({
            "type": "queue_status",
            "position": 0,
            "total_gpus": 1,
            "available_gpus": 1
        })

        # Simulate GPU assignment delay
        await asyncio.sleep(0.1)

        # Start timeout
        timeout_task = asyncio.create_task(session_timeout())

        # Send GPU assigned
        await websocket.send_json({
            "type": "gpu_assigned",
            "gpu_id": 0,
            "session_timeout": SESSION_TIMEOUT_SECONDS,
            "image_url": None  # No initial image for mock
        })

        # Generate initial frame
        await websocket.send_json({"type": "block_count", "count": 0, "max": MAX_BLOCKS})

        # Simulate latency
        await asyncio.sleep(LATENCY_MS / 1000)

        frames = generate_frames(game_state)
        block_count = 1
        await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
        await send_frames(websocket, frames)

        print(f"[Mock] Initial frames sent")

        # Main loop
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "key")

            if message_type == "reset":
                print(f"[Mock] Reset requested")
                await websocket.send_json({"type": "reset_started"})

                game_state.reset()
                await asyncio.sleep(LATENCY_MS / 1000)

                frames = generate_frames(game_state)
                block_count = 1
                await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
                await send_frames(websocket, frames)
                await websocket.send_json({"type": "reset_complete"})
                continue

            key = data.get("key")
            if key and block_count < MAX_BLOCKS:
                if key in CAMERA_MAP:
                    keyboard_vector = [0, 0, 0, 0]
                    mouse_vector = CAMERA_MAP[key]
                else:
                    keyboard_vector = KEYBOARD_MAP.get(key, [0, 0, 0, 0])
                    mouse_vector = [0, 0]

                print(f"[Mock] Key '{key}' pressed, generating block {block_count + 1}...")

                t_start = time.time()

                # Simulate model latency
                await asyncio.sleep(LATENCY_MS / 1000)

                # Update game state and generate frames
                game_state.update(keyboard_vector, mouse_vector)
                frames = generate_frames(game_state)

                t_total = (time.time() - t_start) * 1000

                block_count += 1
                await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
                await send_frames(websocket, frames)

                print(f"[Mock] Block generated in {t_total:.0f}ms (target: {LATENCY_MS}ms)")

    except WebSocketDisconnect:
        print(f"Client {client_id[:8]} disconnected")
    except Exception as e:
        print(f"Client {client_id[:8]} error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()


# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "client", "dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Mock server for latency testing")
    parser.add_argument("--latency", type=int, default=200, help="Simulated latency in ms")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    LATENCY_MS = args.latency
    print(f"Starting mock server with {LATENCY_MS}ms latency on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
