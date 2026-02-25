# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for GEN3C video generation.

Compares newly generated GEN3C videos against device-specific reference videos
using MS-SSIM to detect quality regressions across code changes.

Usage:
    # Requires 1+ GPU, converted GEN3C weights, and reference videos.
    pytest fastvideo/tests/ssim/test_gen3c_similarity.py -v

Environment variables:
    GEN3C_MODEL_PATH  - Path to converted GEN3C weights (default: nvidia/GEN3C-Cosmos-7B)
"""

import os
from pathlib import Path

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)


def _resolve_gen3c_model_path() -> str:
    """
    Resolve a Diffusers-format GEN3C model path for SSIM tests.

    Priority:
    1) GEN3C_MODEL_PATH env var
    2) Local converted checkpoint under repo root
    """
    env_model = os.getenv("GEN3C_MODEL_PATH")
    if env_model:
        return env_model

    repo_root = Path(__file__).resolve().parents[3]
    local_default = repo_root / "converted_weights" / "GEN3C-Cosmos-7B"
    return str(local_default)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
else:
    device_reference_folder = None
    logger.warning(f"Unsupported device for GEN3C SSIM tests: {device_name}")

# ---------------------------------------------------------------------------
# GEN3C generation parameters
# ---------------------------------------------------------------------------

GEN3C_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": _resolve_gen3c_model_path(),
    "height": 720,
    "width": 1280,
    "num_frames": 121,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 1.0,
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 24,
}

MODEL_TO_PARAMS = {
    "GEN3C-Cosmos-7B": GEN3C_T2V_PARAMS,
}

TEST_PROMPTS = [
    "A camera slowly orbits around a vase of flowers on a wooden table in a sunlit room. "
    "The warm afternoon light casts soft shadows across the petals. "
    "Cinematic, shallow depth of field, smooth camera motion.",
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    device_reference_folder is None,
    reason=f"No reference videos for device {device_name}",
)
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_gen3c_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Generate a GEN3C video and compare against the reference using MS-SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "generated_videos", model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    model_path = BASE_PARAMS["model_path"]

    # SSIM test requires a Diffusers-format model with model_index.json.
    if os.path.exists(model_path):
        model_index_path = os.path.join(model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            pytest.skip(
                f"GEN3C_MODEL_PATH is not Diffusers-format (missing model_index.json): {model_path}"
            )
    elif model_path.lower() == "nvidia/gen3c-cosmos-7b":
        pytest.skip(
            "nvidia/GEN3C-Cosmos-7B is the official raw checkpoint repo, not Diffusers format. "
            "Set GEN3C_MODEL_PATH to converted_weights/GEN3C-Cosmos-7B."
        )

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
    }
    if "flow_shift" in BASE_PARAMS:
        init_kwargs["flow_shift"] = BASE_PARAMS["flow_shift"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
    }

    generator = VideoGenerator.from_pretrained(
        model_path=model_path, **init_kwargs
    )
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(output_dir), f"Output not generated at {output_dir}"

    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    # Find matching reference video by prompt substring
    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video not found for prompt: {prompt[:60]}..."
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(f"Computing SSIM: {reference_video_path} vs {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"GEN3C SSIM mean: {mean_ssim}")

    write_ssim_results(
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt,
    )

    # GEN3C uses flow-matching with 50 steps; expect high consistency.
    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM {mean_ssim:.4f} < {min_acceptable_ssim} for {model_id} / {ATTENTION_BACKEND}"
    )
