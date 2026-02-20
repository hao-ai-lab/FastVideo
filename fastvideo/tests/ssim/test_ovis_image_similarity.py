# SPDX-License-Identifier: Apache-2.0
"""
SSIM regression test for Ovis-Image-7B text-to-image pipeline.

Generates a 256×256 image with a fixed seed and compares it against a
committed reference image using MS-SSIM (threshold ≥ 0.98).

How to create reference images (first time):
    1. Run this test once — it will fail with a FileNotFoundError that
       includes the exact cp command needed to bless the output.
    2. Inspect the generated image under
       fastvideo/tests/ssim/generated_videos/Ovis-Image-7B/TORCH_SDPA/
    3. Copy it to the reference folder (command printed by the test).
    4. Commit the reference image and re-run — the test should now pass.

Usage:
    OVIS_WEIGHTS=official_weights/ovis_image \
        pytest fastvideo/tests/ssim/test_ovis_image_similarity.py -vs
"""

from __future__ import annotations

import os
import shlex
import logging

import pytest
import torch

logger = logging.getLogger(__name__)

# OVIS_WEIGHTS = os.getenv("OVIS_WEIGHTS", "AIDC-AI/Ovis-Image-7B")
OVIS_WEIGHTS = os.getenv("OVIS_WEIGHTS", "AIDC-AI/Ovis-Image-7B")
MODEL_ID = "Ovis-Image-7B"

TEST_PROMPTS = [
    'A vibrant poster with the text "FAST VIDEO" written in bold red letters '
    "on a clean white background. Professional design, high contrast, 4k quality.",
]


def _device_reference_folder() -> str:
    suffix = "_reference_videos"
    device_name = torch.cuda.get_device_name(0)
    if "A40" in device_name:
        return "A40" + suffix
    if "L40S" in device_name:
        return "L40S" + suffix
    if "H100" in device_name:
        return "H100" + suffix
    logger.warning(
        "Unsupported device for ssim tests: %s; using L40S references",
        device_name,
    )
    return "L40S" + suffix


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script_method.*:DeprecationWarning",
)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Ovis-Image SSIM test requires CUDA.")
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_ovis_image_similarity(prompt: str, ATTENTION_BACKEND: str) -> None:
    from fastvideo import VideoGenerator
    from fastvideo.tests.utils import (
        compute_video_ssim_torchvision,
        write_ssim_results,
    )

    if not os.path.isdir(OVIS_WEIGHTS):
        pytest.skip(
            f"Ovis-Image weights not found at {OVIS_WEIGHTS} "
            "(set OVIS_WEIGHTS env var to local model path or use HF hub ID)")

    old_backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "generated_videos", MODEL_ID,
                                  ATTENTION_BACKEND)
        os.makedirs(output_dir, exist_ok=True)

        prompt_prefix = prompt[:100].strip()
        output_video_name = f"{prompt_prefix}.mp4"
        expected_video_path = os.path.join(output_dir, output_video_name)

        # Remove stale output to avoid comparing against a previous run.
        for filename in os.listdir(output_dir):
            if filename.endswith(".mp4") and filename.startswith(prompt_prefix):
                try:
                    os.remove(os.path.join(output_dir, filename))
                except FileNotFoundError:
                    pass

        num_inference_steps = 20
        generator = VideoGenerator.from_pretrained(
            model_path=OVIS_WEIGHTS,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
            sp_size=1,
            tp_size=1,
        )
        try:
            generator.generate_video(
                prompt,
                output_path=output_dir,
                save_video=True,
                height=256,
                width=256,
                num_frames=1,
                fps=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=5.0,
                seed=42,
            )
        finally:
            generator.shutdown()

        # Locate the generated file.
        generated_video_path = None
        if os.path.exists(expected_video_path):
            generated_video_path = expected_video_path
        else:
            candidates = [
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if f.endswith(".mp4") and f.startswith(prompt_prefix)
            ]
            if candidates:
                generated_video_path = max(candidates, key=os.path.getmtime)

        assert generated_video_path is not None and os.path.exists(
            generated_video_path), (
                f"Output video was not generated under {output_dir} "
                f"for prompt '{prompt}'")

        # Locate reference.
        device_reference_folder = _device_reference_folder()
        reference_folder = os.path.join(script_dir, device_reference_folder,
                                        MODEL_ID, ATTENTION_BACKEND)

        if not os.path.exists(reference_folder):
            bless_cmd = (
                f"mkdir -p {shlex.quote(reference_folder)} && "
                f"cp {shlex.quote(generated_video_path)} "
                f"{shlex.quote(reference_folder)}/")
            pytest.fail(
                f"Reference folder does not exist: {reference_folder}\n"
                f"Generated image saved at: {generated_video_path}\n"
                "To bless references, run:\n"
                f"  {bless_cmd}")

        reference_video_path = os.path.join(reference_folder, output_video_name)
        if not os.path.exists(reference_video_path):
            reference_video_name = None
            for filename in os.listdir(reference_folder):
                if filename.endswith(".mp4") and filename.startswith(
                        prompt_prefix):
                    reference_video_name = filename
                    break
            if not reference_video_name:
                bless_cmd = (
                    f"cp {shlex.quote(generated_video_path)} "
                    f"{shlex.quote(reference_folder)}/")
                pytest.fail(
                    f"Reference image not found for prompt '{prompt}' "
                    f"under: {reference_folder}\n"
                    f"Expected name: {output_video_name}\n"
                    f"Generated image saved at: {generated_video_path}\n"
                    f"To bless references, run:\n  {bless_cmd}")
            reference_video_path = os.path.join(reference_folder,
                                                 reference_video_name)

        logger.info("Computing SSIM between %s and %s", reference_video_path,
                    generated_video_path)
        ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                     generated_video_path,
                                                     use_ms_ssim=True)

        mean_ssim = ssim_values[0]
        logger.info("SSIM mean value: %s", mean_ssim)

        write_ssim_results(
            output_dir,
            ssim_values,
            reference_video_path,
            generated_video_path,
            num_inference_steps,
            prompt,
        )

        min_acceptable_ssim = 0.98
        assert mean_ssim >= min_acceptable_ssim, (
            f"SSIM {mean_ssim:.4f} below threshold {min_acceptable_ssim} "
            f"for {MODEL_ID} with backend {ATTENTION_BACKEND}")

    finally:
        if old_backend is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = old_backend
