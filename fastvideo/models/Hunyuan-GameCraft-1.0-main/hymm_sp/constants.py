import os
import torch

__all__ = [
    "PROMPT_TEMPLATE", "MODEL_BASE", "PRECISION_TO_TYPE",
    "PRECISIONS", "VAE_PATH", "TEXT_ENCODER_PATH", "TOKENIZER_PATH",
    "TEXT_PROJECTION",
]

# =================== Constant Values =====================

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE = {
    "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO, "crop_start": 95},
}

# ======================= Model ======================
PRECISIONS = {"fp32", "fp16", "bf16"}

# =================== Model Path =====================
MODEL_BASE = os.getenv("MODEL_BASE")

# 3D VAE
VAE_PATH = {
    "884-16c-hy0801": f"{MODEL_BASE}/vae_3d/hyvae",
}

# Text Encoder
TEXT_ENCODER_PATH = {
    "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
    "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1-transformers",
}

# Tokenizer
TOKENIZER_PATH = {
    "clipL": f"{MODEL_BASE}/openai_clip-vit-large-patch14",
    "llava-llama-3-8b": f"{MODEL_BASE}/llava-llama-3-8b-v1_1-transformers",
}

TEXT_PROJECTION = {
    "linear",                               # Default, an nn.Linear() layer
    "single_refiner",                       # Single TokenRefiner. Refer to LI-DiT
}
