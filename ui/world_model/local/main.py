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
    "w": [1, 0, 0, 0], "ArrowUp": [1, 0, 0, 0],
    "s": [0, 1, 0, 0], "ArrowDown": [0, 1, 0, 0],
    "a": [0, 0, 1, 0], "ArrowLeft": [0, 0, 1, 0],
    "d": [0, 0, 0, 1], "ArrowRight": [0, 0, 0, 1],
}

# Global generator
generator = None

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    # Send initial frame
    try:
        action = {
            "keyboard": torch.tensor([0, 0, 0, 0]).cuda(),
            "mouse": torch.tensor([0, 0]).cuda()
        }
        keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
        frames, _ = await generator.step_async(keyboard_cond, mouse_cond)

        if frames is not None and len(frames) > 0:
            # Send all initial frames
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                await websocket.send_json({"type": "frame", "data": frame_b64})
    except Exception as e:
        print(f"Initial frame error: {e}")

    # Handle key presses
    try:
        while True:
            data = await websocket.receive_json()
            key = data.get("key")

            if key:
                action_vector = KEYBOARD_MAP.get(key, [0, 0, 0, 0])
                action = {
                    "keyboard": torch.tensor(action_vector).cuda(),
                    "mouse": torch.tensor([0, 0]).cuda()
                }

                keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
                frames, _ = await generator.step_async(keyboard_cond, mouse_cond)

                if frames is not None and len(frames) > 0:
                    # Send all frames in sequence for smoother animation
                    for frame in frames:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                        await websocket.send_json({"type": "frame", "data": frame_b64})

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
