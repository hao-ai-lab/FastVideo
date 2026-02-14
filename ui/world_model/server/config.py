from pathlib import Path

# Repo root: ui/world_model/server/config.py -> go up 3 levels
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Model registry - add new models here
MODEL_REGISTRY = {
    "matrix-game-2.0-base": {
        "name": "Matrix-Game 2.0 Base",
        "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
        "keyboard_dim": 4,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
    },
    "wangame-1.3b-mc-w-only-still-7k": {
        "name": "WANGame 1.3B MC (W-Only, 7k)",
        "model_path": "weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers",
        "init_weights_from_safetensors": str(_REPO_ROOT / "checkpoints" / "wangame-1.3b-mc-w-only-still-7k" / "checkpoint-7000" / "transformer"),
        "override_transformer_cls_name": "WanGameActionTransformer3DModel",
        "override_pipeline_cls_name": "WanGameCausalDMDPipeline",
        "keyboard_dim": 4,
        #"image_url": "/server-assets/mc.png",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
        # "image_path": str(Path(__file__).resolve().parent / "mc.png"),
    },
}

DEFAULT_MODEL_ID = "matrix-game-2.0-base"

# Active model configuration (set by server or user selection)
MODEL_CONFIG = MODEL_REGISTRY[DEFAULT_MODEL_ID]

# Keyboard mappings (WASD)
KEYBOARD_MAP = {
    "w": [1, 0, 0, 0],
    "s": [0, 1, 0, 0],
    "a": [0, 0, 1, 0],
    "d": [0, 0, 0, 1],
}

# Camera mappings (Arrow keys)
CAM_VALUE = 0.1
CAMERA_MAP = {
    "ArrowUp": [CAM_VALUE, 0],
    "ArrowDown": [-CAM_VALUE, 0],
    "ArrowLeft": [0, -CAM_VALUE],
    "ArrowRight": [0, CAM_VALUE],
}

# Generation limits
MAX_BLOCKS = 50
SESSION_TIMEOUT_SECONDS = 900
MAX_USERS_PER_GPU = 16
DISABLE_BATCHING = False  # Process users sequentially (batch_size=1) instead of batching

# Frame settings
NUM_FRAMES = 597
FRAME_HEIGHT = 352
FRAME_WIDTH = 640
NUM_INFERENCE_STEPS = 4
JPEG_QUALITY = 85
BATCH_SIZE = 12
