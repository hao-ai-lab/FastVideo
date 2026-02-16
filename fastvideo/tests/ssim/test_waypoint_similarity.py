# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for Waypoint-1-Small.

Uses StreamingVideoGenerator (reset/step/finalize). Reference videos must be
generated first: run this test, then `bash update_reference_videos.sh` from
this directory (adjust REFERENCE_DIR to your device, e.g. A40_reference_videos).
"""
# Avoid PyTorch 2.10 + Triton "duplicate template name" on some CI/RunPod images.
# Must be set before torch/fastvideo are imported.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import pytest

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import (
    compute_video_ssim_torchvision,
    write_ssim_results,
)

logger = init_logger(__name__)

device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
else:
    logger.warning(f"Unsupported device for ssim tests: {device_name}")

# Owl-Control keycode: W=forward
KEY_FORWARD = 17

WAYPOINT_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/Waypoint-1-Small-Diffusers",
    "height": 368,
    "width": 640,
    "num_steps": 60,
    "frames_per_step": 1,
    "num_inference_steps": 4,
    "seed": 1024,
    "video_quality": 8,
}

MODEL_TO_PARAMS = {
    "Waypoint-1-Small-Diffusers": WAYPOINT_PARAMS,
}

TEST_PROMPTS = [
    "A first-person view of walking through a grassy field.",
]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_waypoint_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs Waypoint inference via StreamingVideoGenerator and compares
    the output to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, "generated_videos", model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    output_video_name = "video.mp4"
    output_path = os.path.join(output_dir, output_video_name)

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    total_frames = BASE_PARAMS["num_steps"] * BASE_PARAMS["frames_per_step"]
    num_frames_cap = max(32, total_frames + 8)

    generator = StreamingVideoGenerator.from_pretrained(
        BASE_PARAMS["model_path"],
        num_gpus=BASE_PARAMS["num_gpus"],
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )

    reset_kw = dict(
        prompt=prompt,
        num_frames=num_frames_cap,
        height=BASE_PARAMS["height"],
        width=BASE_PARAMS["width"],
        num_inference_steps=num_inference_steps,
        output_path=output_path,
        save_video=True,
        video_quality=BASE_PARAMS["video_quality"],
        seed=BASE_PARAMS["seed"],
    )
    generator.reset(**reset_kw)

    if BASE_PARAMS["seed"] is not None:
        torch.manual_seed(BASE_PARAMS["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_PARAMS["seed"])

    keyboard_cond = torch.zeros(
        (1, BASE_PARAMS["frames_per_step"], 256), dtype=torch.float32
    )
    keyboard_cond[:, :, KEY_FORWARD] = 1.0
    mouse_cond = torch.zeros(
        (1, BASE_PARAMS["frames_per_step"], 2), dtype=torch.float32
    )

    for _ in range(BASE_PARAMS["num_steps"]):
        generator.step(keyboard_cond=keyboard_cond, mouse_cond=mouse_cond)

    saved_path = generator.finalize()
    generator.shutdown()

    assert saved_path and os.path.exists(saved_path), (
        f"Output video was not generated at {output_path}"
    )

    reference_folder = os.path.join(
        script_dir, device_reference_folder, model_id, ATTENTION_BACKEND
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}. "
            "Run this test to generate a video, then "
            "bash update_reference_videos.sh from this directory."
        )

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4"):
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for model: {model_id} "
            f"with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError("Reference video missing")

    reference_video_path = os.path.join(
        reference_folder, reference_video_name
    )
    generated_video_path = saved_path

    logger.info(
        f"Computing SSIM between {reference_video_path} and "
        f"{generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path, generated_video_path, use_ms_ssim=True
    )

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt,
    )

    if not success:
        logger.error("Failed to write SSIM results to file")

    # Waypoint may have run-to-run variance (cuDNN, etc.). Use a relaxed
    # threshold until determinism is verified. Other models use 0.93–0.98.
    min_acceptable_ssim = 0.35
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend: {ATTENTION_BACKEND}"
    )
