import os
from pathlib import Path

import torch
import torch._inductor.config

from fastvideo.configs.sample.base import SamplingParam

LOCAL_DEMO_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("LTX2_CLASSIFIER_DIR", str(LOCAL_DEMO_DIR))
        )
    )
)

MODEL_ID = os.path.expandvars(
    os.path.expanduser(
        os.getenv("LTX2_3_MODEL_PATH", "FastVideo/LTX-2.3-Distilled-Diffusers")
    )
)
MODEL_PATH_MAPPING = {
    "FastLTX-2.3": MODEL_ID,
}

DEFAULT_HEIGHT = 1088
DEFAULT_WIDTH = 1920
DEFAULT_NUM_FRAMES = 121
DEFAULT_FPS = 24
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_NUM_INFERENCE_STEPS = 5
DEFAULT_SEED = 10
DEFAULT_NEGATIVE_PROMPT = ""
REFINE_UPSAMPLER_PATH = "converted/ltx2_spatial_upscaler"
REPO_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_DIR = REPO_ROOT / "outputs_video" / "ltx2_basic_new"
GENERATED_CLIP_ROOT = REPO_ROOT / "outputs_video"
MAX_SESSION_CLIPS = 24

config = torch._inductor.config
config.conv_1x1_as_mm = True
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False

def apply_ltx2_defaults(params: SamplingParam) -> SamplingParam:
    params.height = DEFAULT_HEIGHT
    params.width = DEFAULT_WIDTH
    params.num_frames = DEFAULT_NUM_FRAMES
    params.fps = DEFAULT_FPS
    params.guidance_scale = DEFAULT_GUIDANCE_SCALE
    params.num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
    params.seed = DEFAULT_SEED
    params.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    return params

def resolve_model_path(model_path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(model_path)))

def resolve_refine_upsampler_path(model_path: Path) -> Path:
    candidates = [
        model_path / "spatial_upscaler",
        model_path / "spatial_upsampler",
        Path(os.path.expandvars(os.path.expanduser(REFINE_UPSAMPLER_PATH))),
        REPO_ROOT / REFINE_UPSAMPLER_PATH,
    ]

    env_path = os.getenv("LTX2_REFINE_UPSAMPLER_PATH")
    if env_path:
        candidates.insert(
            0, Path(os.path.expandvars(os.path.expanduser(env_path)))
        )

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find an LTX2 refine upsampler directory.\n"
        "Checked:\n"
        f"{checked}\n"
        "Set LTX2_REFINE_UPSAMPLER_PATH or update REFINE_UPSAMPLER_PATH."
    )

def setup_model_environment(model_path: str) -> None:
    _ = model_path
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
