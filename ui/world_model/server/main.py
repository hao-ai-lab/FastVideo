from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import base64
import cv2
import os
import time
import uuid

from config import (
    KEYBOARD_MAP, CAMERA_MAP, MAX_BLOCKS, JPEG_QUALITY, BATCH_SIZE,
    SESSION_TIMEOUT_SECONDS, MODEL_CONFIG, MODEL_REGISTRY, DEFAULT_MODEL_ID
)
from gpu_pool import GPUPool, GPUSlot, get_available_gpus

# Global GPU pool
gpu_pool: GPUPool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global gpu_pool

    print("Starting server...")

    # Get available GPUs
    gpu_ids = get_available_gpus()
    print(f"Detected GPUs: {gpu_ids}")

    # Initialize GPU pool (spawns subprocess per GPU)
    gpu_pool = GPUPool(gpu_ids)
    await gpu_pool.initialize()

    print("Server ready")
    yield

    print("Shutting down server...")
    await gpu_pool.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
async def get_status():
    """Get the current status of the GPU pool."""
    if gpu_pool is None:
        return {"error": "GPU pool not initialized"}
    return gpu_pool.get_status()


@app.get("/models")
async def get_models():
    """Get available models and the currently active model."""
    models = []
    for model_id, config in MODEL_REGISTRY.items():
        models.append({
            "id": model_id,
            "name": config["name"],
        })
    return {
        "models": models,
        "default_model_id": DEFAULT_MODEL_ID,
    }


def encode_frames(frames: list) -> list[str]:
    """Encode frames to base64 JPEG."""
    encoded = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        encoded.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
    return encoded


async def send_frames(websocket: WebSocket, frames: list):
    """Encode and send frames in batches."""
    if not frames:
        return
    encoded_frames = encode_frames(frames)
    for i in range(0, len(encoded_frames), BATCH_SIZE):
        batch = encoded_frames[i:i + BATCH_SIZE]
        await websocket.send_json({"type": "frame_batch", "frames": batch})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = str(uuid.uuid4())
    print(f"Client {client_id[:8]} connected")

    # Send initial queue status
    status = gpu_pool.get_status()
    await websocket.send_json({
        "type": "queue_status",
        "position": status["queue_size"] + 1 if status["available_gpus"] == 0 else 0,
        "total_gpus": status["total_gpus"],
        "available_gpus": status["available_gpus"]
    })

    gpu_id = None
    slot: GPUSlot = None
    timeout_task: asyncio.Task = None

    async def session_timeout():
        """Close the session after timeout."""
        await asyncio.sleep(SESSION_TIMEOUT_SECONDS)
        print(f"[GPU {gpu_id}] Session timeout for client {client_id[:8]}")
        try:
            await websocket.send_json({
                "type": "session_timeout",
                "message": f"Session expired after {SESSION_TIMEOUT_SECONDS} seconds"
            })
            await websocket.close(code=1000, reason="Session timeout")
        except Exception:
            pass  # WebSocket may already be closed

    try:
        # Wait for model selection from client
        model_id = DEFAULT_MODEL_ID
        model_config = MODEL_CONFIG
        try:
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            if init_data.get("type") == "select_model":
                model_id = init_data.get("model_id", DEFAULT_MODEL_ID)
                if model_id in MODEL_REGISTRY:
                    model_config = MODEL_REGISTRY[model_id]
                    print(f"Client {client_id[:8]} selected model: {model_id}")
        except asyncio.TimeoutError:
            pass  # Use default model config

        # Acquire a GPU slot (may wait in queue, may share with other users)
        gpu_id, slot = await gpu_pool.acquire(client_id, websocket)

        # Start session timeout
        timeout_task = asyncio.create_task(session_timeout())

        # Join the multi-user engine on this GPU (triggers reload if model differs)
        await slot.join_user(client_id, model_id=model_id)

        # Notify client they're connected to a GPU
        await websocket.send_json({
            "type": "gpu_assigned",
            "gpu_id": gpu_id,
            "session_timeout": SESSION_TIMEOUT_SECONDS,
            "image_url": model_config.get("image_url")
        })

        # Generate and send initial frame
        block_count = 0
        await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})

        try:
            frames, timings = await slot.user_step(
                client_id, [0, 0, 0, 0], [0, 0])
            block_count = 1
            await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
            await send_frames(websocket, frames)
            if timings:
                print(f"[GPU {gpu_id}] Initial ({client_id[:8]}): model={timings.get('model_step_ms', 0):.0f}ms")
        except Exception as e:
            print(f"[GPU {gpu_id}] Initial frame error for {client_id[:8]}: {e}")

        # Main event loop
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "key")

            if message_type == "reset":
                print(f"[GPU {gpu_id}] Reset requested by client {client_id[:8]}")
                await websocket.send_json({"type": "reset_started"})

                try:
                    # Leave and rejoin to reset user state
                    await slot.leave_user(client_id)
                    await slot.join_user(client_id, model_id=model_id)

                    frames, timings = await slot.user_step(
                        client_id, [0, 0, 0, 0], [0, 0])
                    block_count = 1
                    await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
                    await send_frames(websocket, frames)
                    await websocket.send_json({"type": "reset_complete"})
                    print(f"[GPU {gpu_id}] Reset complete for {client_id[:8]}")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reset error for {client_id[:8]}: {e}")
                    await websocket.send_json({"type": "reset_complete"})
                continue

            # Handle key press
            key = data.get("key")
            if key and block_count < MAX_BLOCKS:
                if key in CAMERA_MAP:
                    keyboard_vector = [0, 0, 0, 0]
                    mouse_vector = CAMERA_MAP[key]
                else:
                    keyboard_vector = KEYBOARD_MAP.get(key, [0, 0, 0, 0])
                    mouse_vector = [0, 0]

                print(f"[GPU {gpu_id}] Key '{key}' from {client_id[:8]}, generating block {block_count + 1}...")

                t_start = time.time()
                frames, timings = await slot.user_step(
                    client_id, keyboard_vector, mouse_vector)
                t_generation = time.time() - t_start

                block_count += 1
                await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})

                if frames:
                    t_encode_start = time.time()
                    await send_frames(websocket, frames)
                    t_encode = time.time() - t_encode_start

                    dit_ms = timings.get('dit_ms', 0)
                    vae_ms = timings.get('vae_ms', 0)
                    print(f"[GPU {gpu_id}] ({client_id[:8]}) DiT: {dit_ms:.0f}ms, VAE: {vae_ms:.0f}ms, Encode+Send: {t_encode*1000:.0f}ms, Total: {t_generation*1000:.0f}ms")

    except WebSocketDisconnect:
        print(f"Client {client_id[:8]} disconnected")
    except Exception as e:
        print(f"Client {client_id[:8]} error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cancel the timeout task if still running
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
        if client_id:
            await gpu_pool.release(client_id)


# Serve server-side assets (images, etc.)
server_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/server-assets", StaticFiles(directory=server_dir), name="server-assets")

# Serve built frontend (must be after API/WebSocket routes)
static_dir = os.path.join(server_dir, "..", "client", "dist")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
