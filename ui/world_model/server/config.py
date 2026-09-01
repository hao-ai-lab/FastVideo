# Model configuration
MODEL_CONFIG = {
    "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
    "keyboard_dim": 4,
    "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
}

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
SESSION_TIMEOUT_SECONDS = 90

# Frame settings
NUM_FRAMES = 597
FRAME_HEIGHT = 352
FRAME_WIDTH = 640
NUM_INFERENCE_STEPS = 10
JPEG_QUALITY = 85
BATCH_SIZE = 12
