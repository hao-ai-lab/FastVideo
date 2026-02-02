from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import torch
import os

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
from fastvideo.models.dits.matrixgame.utils import expand_action_to_frames

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model config
MODEL_CONFIG = {
    "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
    "keyboard_dim": 4,
    "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
}

# Keyboard mappings
KEYBOARD_MAP = {
    "w": [1, 0, 0, 0],
    "s": [0, 1, 0, 0],
    "a": [0, 0, 1, 0],
    "d": [0, 0, 0, 1],
}

# Camera mappings
CAM_VALUE = 0.1
CAMERA_MAP = {
    "ArrowUp": [CAM_VALUE, 0],
    "ArrowDown": [-CAM_VALUE, 0],
    "ArrowLeft": [0, -CAM_VALUE],
    "ArrowRight": [0, CAM_VALUE],
}

generator = None

MAX_BLOCKS = 50

@app.on_event("startup")
async def startup_event():
    global generator
    print("Loading model...")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    generator = StreamingVideoGenerator.from_pretrained(
        MODEL_CONFIG["model_path"],
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    num_frames = 597
    actions = {
        "keyboard": torch.zeros((num_frames, MODEL_CONFIG["keyboard_dim"])),
        "mouse": torch.zeros((num_frames, 2))
    }

    generator.reset(
        prompt="",
        image_path=MODEL_CONFIG["image_url"],
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=torch.tensor([150, 44, 80]),
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=10,
    )

    print("Ready")


async def soft_reset_and_send_frame(websocket: WebSocket):
    """Fast reset without reloading the model."""
    import asyncio
    from fastvideo.configs.sample.base import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import align_to, shallow_asdict

    print(f"Starting soft reset (queue_mode={generator._use_queue_mode}, streaming_enabled={generator.executor._streaming_enabled})...")

    # Clear local state
    generator.accumulated_frames = []
    generator.block_idx = 0
    generator.block_dir = None
    if generator.writer:
        generator.writer.close()
        generator.writer = None

    # Use queue mode to clear
    if generator._use_queue_mode and generator.executor._streaming_enabled:
        print("Using queue mode submit_clear...")
        generator.executor.submit_clear()
        clear_result = generator.executor.wait_result()
        print(f"Clear result: error={clear_result.error}, task_type={clear_result.task_type}")
        if clear_result.error:
            raise clear_result.error
    else:
        print("Skipping clear (not in queue streaming mode)")

    # Set up sampling parameters
    num_frames = 597
    actions = {
        "keyboard": torch.zeros((num_frames, MODEL_CONFIG["keyboard_dim"])),
        "mouse": torch.zeros((num_frames, 2))
    }

    if generator.sampling_param is None:
        generator.sampling_param = SamplingParam.from_pretrained(
            generator.fastvideo_args.model_path)

    generator.sampling_param.update({
        "prompt": "",
        "image_path": MODEL_CONFIG["image_url"],
        "mouse_cond": actions["mouse"].unsqueeze(0),
        "keyboard_cond": actions["keyboard"].unsqueeze(0),
        "grid_sizes": torch.tensor([150, 44, 80]),
        "num_frames": num_frames,
        "height": 352,
        "width": 640,
        "num_inference_steps": 10,
    })

    generator.sampling_param.height = align_to(generator.sampling_param.height, 16)
    generator.sampling_param.width = align_to(generator.sampling_param.width, 16)

    latents_size = [(generator.sampling_param.num_frames - 1) // 4 + 1,
                    generator.sampling_param.height // 8,
                    generator.sampling_param.width // 8]
    n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

    generator.sampling_param.return_frames = True
    generator.sampling_param.save_video = False

    # Create new batch
    generator.batch = ForwardBatch(
        **shallow_asdict(generator.sampling_param),
        eta=0.0,
        n_tokens=n_tokens,
        VSA_sparsity=generator.fastvideo_args.VSA_sparsity,
    )

    # Use queue mode to reset
    if generator._use_queue_mode:
        print("Using queue mode submit_reset...")
        generator.executor.submit_reset(generator.batch, generator.fastvideo_args)
        result = generator.executor.wait_result()
        print(f"Reset result: error={result.error}, output_batch={result.output_batch is not None}")
        if result.error:
            raise result.error
        print("Reset acknowledged by workers")
    else:
        print("Using RPC mode execute_streaming_reset...")
        generator.executor.execute_streaming_reset(generator.batch, generator.fastvideo_args)

    print("Waiting for workers to settle...")
    await asyncio.sleep(0.5)

    # Send initial frame
    try:
        action = {
            "keyboard": torch.tensor([0, 0, 0, 0]).cuda(),
            "mouse": torch.tensor([0, 0]).cuda()
        }
        keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
        print("Generating initial frames...")
        frames, _ = await generator.step_async(keyboard_cond, mouse_cond)

        if frames is None:
            print("Warning: Got None for frames, trying again...")
            await asyncio.sleep(0.5)
            frames, _ = await generator.step_async(keyboard_cond, mouse_cond)

        print(f"Got {len(frames) if frames else 0} initial frames")

        if frames is not None and len(frames) > 0:
            encoded_frames = []
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                encoded_frames.append(frame_b64)

            BATCH_SIZE = 12
            for i in range(0, len(encoded_frames), BATCH_SIZE):
                batch = encoded_frames[i:i + BATCH_SIZE]
                await websocket.send_json({"type": "frame_batch", "frames": batch})

        print("Fast reset complete and initial frames sent")
        return 1  # Return block count after initial frame
    except Exception as e:
        print(f"Error during fast reset: {e}")
        import traceback
        traceback.print_exc()
        return 0


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    block_count = 0

    # Send initial block count
    await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})

    # Send initial frame
    try:
        action = {
            "keyboard": torch.tensor([0, 0, 0, 0]).cuda(),
            "mouse": torch.tensor([0, 0]).cuda()
        }
        keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
        frames, _ = await generator.step_async(keyboard_cond, mouse_cond)

        block_count += 1
        await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})

        if frames is not None and len(frames) > 0:
            # Encode and send frames in batches
            encoded_frames = []
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                encoded_frames.append(frame_b64)

            BATCH_SIZE = 12
            for i in range(0, len(encoded_frames), BATCH_SIZE):
                batch = encoded_frames[i:i + BATCH_SIZE]
                await websocket.send_json({"type": "frame_batch", "frames": batch})
    except Exception as e:
        print(f"Initial frame error: {e}")

    # Handle key presses and reset requests
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "key")

            if message_type == "reset":
                # Handle reset request
                print("Reset requested by client...")
                await websocket.send_json({"type": "reset_started"})

                block_count = await soft_reset_and_send_frame(websocket)
                await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})
                await websocket.send_json({"type": "reset_complete"})
                print(f"Reset complete, block_count: {block_count}")

                continue

            # Handle key press
            key = data.get("key")
            if key and block_count < MAX_BLOCKS:
                # Check if it's a camera control (arrow keys) or movement (WASD)
                if key in CAMERA_MAP:
                    keyboard_vector = [0, 0, 0, 0]
                    mouse_vector = CAMERA_MAP[key]
                else:
                    keyboard_vector = KEYBOARD_MAP.get(key, [0, 0, 0, 0])
                    mouse_vector = [0, 0]

                action = {
                    "keyboard": torch.tensor(keyboard_vector).cuda(),
                    "mouse": torch.tensor(mouse_vector).cuda()
                }

                keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
                print(f"Key '{key}' pressed, generating frames for block {block_count + 1}...")

                import time
                t_start = time.time()
                frames, _ = await generator.step_async(keyboard_cond, mouse_cond)
                t_generation = time.time() - t_start
                print(f"Received {len(frames) if frames else 0} frames from generator in {t_generation:.3f}s")

                block_count += 1
                await websocket.send_json({"type": "block_count", "count": block_count, "max": MAX_BLOCKS})

                if frames is not None and len(frames) > 0:
                    # Batch encode all frames
                    print(f"Encoding {len(frames)} frames...")
                    t_encode_start = time.time()

                    encoded_frames = []
                    for frame in frames:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                        encoded_frames.append(frame_b64)

                    t_encode_total = time.time() - t_encode_start

                    BATCH_SIZE = 12
                    print(f"Sending {len(frames)} frames in batches of {BATCH_SIZE}...")
                    t_send_start = time.time()

                    for i in range(0, len(encoded_frames), BATCH_SIZE):
                        batch = encoded_frames[i:i + BATCH_SIZE]
                        await websocket.send_json({
                            "type": "frame_batch",
                            "frames": batch,
                            "timestamp": time.time()
                        })

                    t_send_total = time.time() - t_send_start

                    print(f"Finished sending {len(frames)} frames in {len(encoded_frames) // BATCH_SIZE + (1 if len(encoded_frames) % BATCH_SIZE else 0)} batches")
                    print(f"  Generation: {t_generation:.3f}s ({t_generation/len(frames)*1000:.1f}ms/frame)")
                    print(f"  Encoding:   {t_encode_total:.3f}s ({t_encode_total/len(frames)*1000:.1f}ms/frame)")
                    print(f"  Sending:    {t_send_total:.3f}s ({len(encoded_frames) // BATCH_SIZE + (1 if len(encoded_frames) % BATCH_SIZE else 0)} batches)")
                    print(f"  Total:      {t_generation + t_encode_total + t_send_total:.3f}s")

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
