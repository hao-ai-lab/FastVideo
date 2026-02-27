# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for GEN3C video generation.

Compares newly generated GEN3C videos against device-specific reference videos
using MS-SSIM to detect quality regressions across code changes.

Usage:
    # Requires 1+ GPU and reference videos.
    pytest fastvideo/tests/ssim/test_gen3c_similarity.py -v

Environment variables:
    GEN3C_MODEL_PATH  - Diffusers-format GEN3C model path/repo id.
                        Default: vbharath/GEN3C-Cosmos-7B-Diffusers
                        (local converted path also supported)
"""

import os
import glob
from pathlib import Path

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)


def _resolve_gen3c_test_image_path() -> str:
    """
    Resolve image path for GEN3C I2V SSIM tests.

    Priority:
    1) GEN3C_TEST_IMAGE_PATH env var
    2) Repo asset image
    """
    env_image = os.getenv("GEN3C_TEST_IMAGE_PATH")
    if env_image:
        return env_image

    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "assets" / "girl.png")


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
    "model_path": os.getenv("GEN3C_MODEL_PATH",
                            "vbharath/GEN3C-Cosmos-7B-Diffusers"),
    "height": 720,
    "width": 1280,
    "num_frames": 121,
    "num_inference_steps": 12,
    "guidance_scale": 6.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 1.0,
    "seed": 1024,
    "image_path": _resolve_gen3c_test_image_path(),
    "sp_size": 1,
    "tp_size": 1,
    "fps": 24,
}

MODEL_TO_PARAMS = {
    "GEN3C-Cosmos-7B": GEN3C_T2V_PARAMS,
}

TEST_PROMPTS = [
    "A camera slowly orbits around a young woman sitting at a table with a coffee mug with coffee in it in front of her, "
    "looking away naturally. Soft indoor lighting, cinematic framing, shallow depth of field, smooth camera motion.",
]

BASELINE_VIDEO_NAME = "gen3c_ssim_baseline.mp4"
CANDIDATE_VIDEO_NAME = "gen3c_ssim_candidate.mp4"


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
    output_video_name = CANDIDATE_VIDEO_NAME
    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    model_path = BASE_PARAMS["model_path"]

    # Guard common misconfigurations to keep CI behavior explicit.
    if model_path.lower() == "nvidia/gen3c-cosmos-7b":
        pytest.skip(
            "nvidia/GEN3C-Cosmos-7B is the official raw checkpoint repo, not Diffusers format. "
            "Use GEN3C_MODEL_PATH=vbharath/GEN3C-Cosmos-7B-Diffusers or a local converted path."
        )

    local_like = model_path.startswith(("/", "./", "../"))
    if local_like and not os.path.exists(model_path):
        pytest.skip(
            f"Local GEN3C model path not found: {model_path}. "
            "Set GEN3C_MODEL_PATH to a valid local path or HF Diffusers repo id."
        )

    if os.path.exists(model_path):
        model_index_path = os.path.join(model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            pytest.skip(
                f"GEN3C_MODEL_PATH is not Diffusers-format (missing model_index.json): {model_path}"
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
        "output_path": os.path.join(output_dir, output_video_name),
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
        "seed": BASE_PARAMS["seed"],
        "image_path": BASE_PARAMS["image_path"],
        "fps": BASE_PARAMS["fps"],
    }

    if not os.path.exists(generation_kwargs["image_path"]):
        pytest.skip(
            f"GEN3C test image not found: {generation_kwargs['image_path']}. "
            "Set GEN3C_TEST_IMAGE_PATH to a valid local image."
        )

    # Keep local reruns deterministic: remove prior candidate outputs so
    # VideoGenerator does not auto-suffix (_1, _2, ...).
    stale_pattern = os.path.join(output_dir, "gen3c_ssim_candidate*.mp4")
    for stale_video in glob.glob(stale_pattern):
        os.remove(stale_video)

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

    reference_video_path = os.path.join(reference_folder, BASELINE_VIDEO_NAME)
    if not os.path.exists(reference_video_path):
        raise FileNotFoundError(
            f"Reference video not found: {reference_video_path}"
        )

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

    # GEN3C SSIM threshold for stable L40S reference comparisons.
    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM {mean_ssim:.4f} < {min_acceptable_ssim} for {model_id} / {ATTENTION_BACKEND}"
    )
