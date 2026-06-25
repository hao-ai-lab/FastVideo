# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for Waypoint-1-Small.

Drives ``VideoGenerator`` and compares against a device-specific reference,
using the shared ``reference_utils`` helpers (same as the Matrix-Game tests) so
the reference path matches the tiered layout the HF download CLI populates
(``reference_videos/<tier>/<GPU>_reference_videos/...``), with a flat-legacy
fallback. To create/update references: run this test once (it generates the
video and fails because the reference is missing), then
`python reference_videos_cli.py copy-local --quality-tier default
--device-folder <GPU>_reference_videos` from this directory to promote the
freshly generated video, then re-run to compare. Canonical CI references are
generated on L40S.
"""
# Avoid PyTorch 2.10 + Triton "duplicate template name" on some CI/RunPod images.
# Must be set before torch/fastvideo are imported.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
)
from fastvideo.tests.utils import (
    compute_video_ssim_torchvision,
    write_ssim_results,
)

logger = init_logger(__name__)

# Resolve the device-specific reference folder via the shared helper so this
# test reads the same tiered layout the HF download CLI populates
# (reference_videos/<tier>/<GPU>_reference_videos/...), with a flat-legacy
# fallback. Canonical CI references are generated on L40S/A40.
_device_name = get_cuda_device_name()
device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H200", "H200"),
    ),
    device_name=_device_name,
    # Keep "any device works": fall back to a folder derived from the GPU name
    # (e.g. "NVIDIA RTX 4090" -> "RTX_4090_reference_videos").
    unknown_device_prefix=_device_name.replace("NVIDIA ", "").strip().replace(
        " ", "_"),
    logger=logger,
)

# Owl-Control keycode: W=forward
KEY_FORWARD = 17

# WorldEngineVAE always decodes to 360x640, so frame count (num_steps) is the
# main cost lever; keep it small for a fast regression pass.
WAYPOINT_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/Waypoint-1-Small-Diffusers",
    "height": 368,
    "width": 640,
    "num_steps": 16,
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
    Run Waypoint inference via VideoGenerator and compare the output to the
    device's reference video using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = build_generated_output_dir(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    num_frames = BASE_PARAMS["num_steps"] * BASE_PARAMS["frames_per_step"]

    if BASE_PARAMS["seed"] is not None:
        torch.manual_seed(BASE_PARAMS["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_PARAMS["seed"])

    generator = VideoGenerator.from_pretrained(
        BASE_PARAMS["model_path"],
        num_gpus=BASE_PARAMS["num_gpus"],
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )

    keyboard_cond = torch.zeros((num_frames, 256), dtype=torch.float32)
    keyboard_cond[:, KEY_FORWARD] = 1.0
    mouse_cond = torch.zeros((num_frames, 2), dtype=torch.float32)

    generator.generate_video(
        prompt=prompt,
        keyboard_cond=keyboard_cond.unsqueeze(0),
        mouse_cond=mouse_cond.unsqueeze(0),
        num_frames=num_frames,
        height=BASE_PARAMS["height"],
        width=BASE_PARAMS["width"],
        num_inference_steps=num_inference_steps,
        fps=BASE_PARAMS.get("fps", 60),
        seed=BASE_PARAMS["seed"],
        output_path=output_dir,
        save_video=True,
    )

    # generate_video writes "<prompt>.mp4" under output_dir; take the newest mp4.
    import glob
    mp4s = sorted(glob.glob(os.path.join(output_dir, "*.mp4")),
                  key=os.path.getmtime)
    saved_path = mp4s[-1] if mp4s else None

    assert saved_path and os.path.exists(saved_path), (
        f"Output video was not generated at {output_path}"
    )

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}. "
            "Run this test to generate a video, then "
            "`python reference_videos_cli.py copy-local` from this directory."
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

    # Determinism verified locally with TORCH_COMPILE_DISABLE + fixed seed
    # (same-device regen vs reference gives SSIM=1.0), so use a tight threshold.
    min_acceptable_ssim = 0.95
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend: {ATTENTION_BACKEND}"
    )
