# SPDX-License-Identifier: Apache-2.0
"""
SSIM-based similarity tests for Lucy-Edit-Dev video editing.

Generates a video using the Lucy-Edit-Dev pipeline from a synthetic
input video and compares the output against a pre-generated reference
video using MS-SSIM. This tests self-consistency of the pipeline.

Note: num_inference_steps is reduced for CI speed (10 steps vs 50 recommended).
"""
import os
import tempfile

import imageio
import numpy as np
import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results

logger = init_logger(__name__)

# Device-specific reference folder
device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
elif "H100" in device_name:
    device_reference_folder = "H100" + device_reference_folder_suffix
else:
    logger.warning(f"Unsupported device for ssim tests: {device_name}")

# =============================================================================
# Lucy-Edit-Dev Parameters
# =============================================================================
LUCY_EDIT_PARAMS = {
    "num_gpus": 1,
    "model_path": "official_weights/Lucy-Edit-Dev",
    "height": 480,
    "width": 832,
    "num_frames": 17,
    "num_inference_steps": 10,  # Reduced from 50 for CI speed
    "guidance_scale": 5.0,
    "fps": 24,
    "seed": 42,
}

# Test prompt
LUCY_EDIT_TEST_PROMPTS = [
    "Change the shirt to a bright red leather jacket with a glossy finish, "
    "add aviator sunglasses.",
]


def _create_synthetic_video(num_frames, height, width, seed=0):
    """Create a deterministic synthetic video for the editing pipeline."""
    rng = np.random.RandomState(seed)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    writer = imageio.get_writer(tmp.name, fps=24, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for _ in range(num_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.append_data(frame)
    writer.close()

    return tmp.name


@pytest.mark.parametrize("prompt", LUCY_EDIT_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
def test_lucy_edit_similarity(prompt: str, ATTENTION_BACKEND: str):
    """
    Test Lucy-Edit-Dev inference and compare output to reference videos
    using MS-SSIM.

    Parameters derived from examples/inference/basic/basic_lucy_edit.py
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    weights_path = LUCY_EDIT_PARAMS["model_path"]
    if not os.path.isdir(weights_path):
        pytest.skip(
            f"Missing Lucy-Edit-Dev weights at {weights_path}. "
            "Download with: python scripts/huggingface/download_hf.py "
            "--repo_id decart-ai/Lucy-Edit-Dev "
            "--local_dir official_weights/Lucy-Edit-Dev --repo_type model"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "Lucy-Edit-Dev"

    output_dir = os.path.join(script_dir, "generated_videos", model_id,
                              ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip().rstrip(chr(46))}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    # Create deterministic synthetic input video
    video_path = _create_synthetic_video(
        LUCY_EDIT_PARAMS["num_frames"],
        LUCY_EDIT_PARAMS["height"],
        LUCY_EDIT_PARAMS["width"],
        seed=0,
    )

    try:
        generator = VideoGenerator.from_pretrained(
            model_path=weights_path,
            num_gpus=LUCY_EDIT_PARAMS["num_gpus"],
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
        )

        generator.generate_video(
            prompt,
            video_path=video_path,
            output_path=output_dir,
            output_video_name=output_video_name,
            save_video=True,
            height=LUCY_EDIT_PARAMS["height"],
            width=LUCY_EDIT_PARAMS["width"],
            num_frames=LUCY_EDIT_PARAMS["num_frames"],
            num_inference_steps=LUCY_EDIT_PARAMS["num_inference_steps"],
            guidance_scale=LUCY_EDIT_PARAMS["guidance_scale"],
            fps=LUCY_EDIT_PARAMS["fps"],
            seed=LUCY_EDIT_PARAMS["seed"],
        )
        generator.shutdown()
    finally:
        os.unlink(video_path)

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}"
    )

    # Find reference video
    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )
    if not os.path.exists(reference_folder):
        pytest.skip(
            f"Reference video folder does not exist: {reference_folder}. "
            "Generate reference videos first."
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        pytest.skip(
            f"Reference video not found for prompt: {prompt[:50]}... "
            f"with backend: {ATTENTION_BACKEND}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    write_ssim_results(
        output_dir, ssim_values, reference_video_path, generated_video_path,
        LUCY_EDIT_PARAMS["num_inference_steps"], prompt
    )

    min_acceptable_ssim = 0.80
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}"
    )
