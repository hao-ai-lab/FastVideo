# SPDX-License-Identifier: Apache-2.0
"""SSIM-based similarity test for LingBot-World-Fast causal I2V.

The camera trajectory and source image are the tracked copies of official
LingBot example 03, matching the released `run_fast.sh` workflow.

Note: this checkpoint is 4-step distilled, so the step count is fixed by the
model rather than reduced for CI.
"""

import os

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results

logger = init_logger(__name__)

REQUIRED_GPUS = 2

# The released checkpoint is 4-step distilled; see LingBotWorldFastArchConfig.
NUM_DISTILLED_STEPS = 4


def _find_lingbotworld_action_path() -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    candidate = os.path.join(repo_root, "examples", "datasets", "lingbotworld2")
    if (os.path.exists(os.path.join(candidate, "image.jpg"))
            and os.path.exists(os.path.join(candidate, "poses.npy"))
            and os.path.exists(os.path.join(candidate, "intrinsics.npy"))):
        return os.path.abspath(candidate)
    return None


device_name = get_cuda_device_name()
device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
    ),
    device_name=device_name,
    logger=logger,
)

LINGBOT_FAST_PARAMS = {
    "model_path": "FastVideo/LingBot-World-Fast-Diffusers",
    "num_gpus": 2,
    "height": 480,
    "width": 832,
    "num_frames": 25,  # must be 4k+1; trimmed to a whole number of chunks
    "seed": 42,
    "fps": 16,
}
LINGBOT_FAST_FULL_QUALITY_PARAMS = {
    **LINGBOT_FAST_PARAMS,
    "num_frames": 81,
}

TEST_PROMPTS = [
    "A serene lakeside scene with a lone tree standing in calm water, surrounded "
    "by distant snow-capped mountains under a bright blue sky with drifting white "
    "clouds — gentle ripples reflect the tree and sky, creating a tranquil, "
    "meditative atmosphere.",
]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_lingbot_fast_i2v_similarity(prompt: str, ATTENTION_BACKEND: str):
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    params = select_ssim_params(LINGBOT_FAST_PARAMS,
                               LINGBOT_FAST_FULL_QUALITY_PARAMS)

    if device_reference_folder is None:
        pytest.skip(
            f"Unsupported device for LingBot-World-Fast SSIM test: {device_name}"
        )
    if torch.cuda.device_count() < REQUIRED_GPUS:
        pytest.skip(
            f"LingBot-World-Fast SSIM test requires {REQUIRED_GPUS} GPUs, "
            f"but only {torch.cuda.device_count()} detected.")

    action_path = _find_lingbotworld_action_path()
    if action_path is None:
        pytest.skip(
            "LingBot camera assets not found under examples/datasets/lingbotworld2."
        )
    image_path = os.path.join(action_path, "image.jpg")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LingBot-World-Fast-Diffusers"
    output_dir = build_generated_output_dir(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    init_kwargs = {
        "num_gpus": params["num_gpus"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "pin_cpu_memory": True,
    }
    generation_kwargs = {
        "output_path": output_dir,
        "image_path": image_path,
        "action_path": action_path,
        "height": params["height"],
        "width": params["width"],
        "num_frames": params["num_frames"],
        "seed": params["seed"],
        "fps": params["fps"],
    }

    generator: VideoGenerator | None = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=params["model_path"], **init_kwargs)
        generator.generate_video(prompt, **generation_kwargs)
    finally:
        if generator is not None:
            generator.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}")

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break
    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video missing for prompt/backend under {reference_folder}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    logger.info("Computing SSIM between %s and %s", reference_video_path,
                generated_video_path)
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)
    mean_ssim = ssim_values[0]
    logger.info("SSIM mean value: %s", mean_ssim)

    write_ssim_results(output_dir, ssim_values, reference_video_path,
                       generated_video_path, NUM_DISTILLED_STEPS, prompt)

    min_acceptable_ssim = 0.70
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")
