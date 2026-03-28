# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import logging

import pytest
import torch

from fastvideo.configs.sample.sd35 import SD35SamplingParam
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    resolve_device_reference_folder,
    select_ssim_params,
)

from fastvideo.tests.ssim.reference_utils import get_cuda_device_name

logger = logging.getLogger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H100", "H100"),
        ("H200", "H200"),
        ("RTX 4090", "RTX4090"),
        ("4090", "RTX4090"),
    ),
    device_name=get_cuda_device_name(),
    fallback_device_prefix="L40S",
    logger=logger,
)


SD35_MODEL_PATH = os.getenv(
    "SD35_MODEL_DIR",
    "stabilityai/stable-diffusion-3.5-medium",
)

MODEL_ID = "stabilityai__stable-diffusion-3.5-medium"

TEST_PROMPTS = [
    "a photo of a cat",
]

SD35_PARAMS = {
    "height": 256,
    "width": 256,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 8,
    "guidance_scale": 6.0,
    "seed": 0,
    "negative_prompt": "lowres, blurry",
}
_SD35_FULL_QUALITY_DEFAULTS = SD35SamplingParam()
SD35_FULL_QUALITY_PARAMS = {
    "height": _SD35_FULL_QUALITY_DEFAULTS.height,
    "width": _SD35_FULL_QUALITY_DEFAULTS.width,
    "num_frames": SD35_PARAMS["num_frames"],  # default num_frames: 1
    "fps": _SD35_FULL_QUALITY_DEFAULTS.fps,
    "num_inference_steps": _SD35_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _SD35_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _SD35_FULL_QUALITY_DEFAULTS.seed,
    "negative_prompt": _SD35_FULL_QUALITY_DEFAULTS.negative_prompt,
}

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script_method.*:DeprecationWarning",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SD3.5 SSIM test requires CUDA")
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_sd35_similarity(prompt: str, ATTENTION_BACKEND: str) -> None:
    from fastvideo import VideoGenerator
    from fastvideo.tests.utils import (
        compute_video_ssim_torchvision,
        write_ssim_results,
    )

    is_hf_repo = "/" in SD35_MODEL_PATH and not SD35_MODEL_PATH.startswith("/")
    if not is_hf_repo and not os.path.isdir(SD35_MODEL_PATH):
        pytest.skip(f"SD3.5 weights not found at {SD35_MODEL_PATH} (set SD35_MODEL_DIR to override)")

    old_backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    old_transformers_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    try:
        params = select_ssim_params(SD35_PARAMS, SD35_FULL_QUALITY_PARAMS)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            MODEL_ID,
            ATTENTION_BACKEND,
        )
        os.makedirs(output_dir, exist_ok=True)

        prompt_prefix = prompt[:100].strip()
        output_video_name = f"{prompt_prefix}.mp4"
        expected_video_path = os.path.join(output_dir, output_video_name)

        for filename in os.listdir(output_dir):
            if filename.endswith(".mp4") and filename.startswith(prompt_prefix):
                try:
                    os.remove(os.path.join(output_dir, filename))
                except FileNotFoundError:
                    pass

        init_kwargs = {
            "num_gpus": 1,
            # Save a single-frame MP4 to reuse the existing video-SSIM harness.
            "workload_type": "t2v",
            "sp_size": 1,
            "tp_size": 1,
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "image_encoder_cpu_offload": False,
            "pin_cpu_memory": False,
            "use_fsdp_inference": False,
        }

        num_inference_steps = params["num_inference_steps"]
        generation_kwargs = {
            "output_path": output_dir,
            "height": params["height"],
            "width": params["width"],
            "num_frames": params["num_frames"],
            "fps": params["fps"],
            "num_inference_steps": num_inference_steps,
            "guidance_scale": params["guidance_scale"],
            "seed": params["seed"],
            "negative_prompt": params["negative_prompt"],
            "save_video": True,
        }

        generator = VideoGenerator.from_pretrained(model_path=SD35_MODEL_PATH, **init_kwargs)
        try:
            generator.generate_video(prompt, **generation_kwargs)
        finally:
            generator.shutdown()

        generated_video_path = None
        if os.path.exists(expected_video_path):
            generated_video_path = expected_video_path
        else:
            candidates = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith(".mp4") and f.startswith(prompt_prefix)
            ]
            if candidates:
                generated_video_path = max(candidates, key=os.path.getmtime)

        assert generated_video_path is not None and os.path.exists(generated_video_path), (
            f"Output video was not generated under {output_dir} for prompt '{prompt}'"
        )

        reference_folder = build_reference_folder_path(
            script_dir,
            device_reference_folder,
            MODEL_ID,
            ATTENTION_BACKEND,
        )

        if not os.path.exists(reference_folder):
            bless_cmd = (
                f"mkdir -p {shlex.quote(reference_folder)} && "
                f"cp {shlex.quote(generated_video_path)} {shlex.quote(reference_folder)}/"
            )
            pytest.fail(
                f"Reference folder does not exist: {reference_folder}\n"
                f"Generated video saved at: {generated_video_path}\n"
                "To bless references, copy the generated mp4 into the reference folder with a matching name.\n"
                f"Example:\n  {bless_cmd}"
            )

        reference_video_path = os.path.join(reference_folder, output_video_name)
        if not os.path.exists(reference_video_path):
            reference_video_name = None
            for filename in os.listdir(reference_folder):
                if filename.endswith(".mp4") and filename.startswith(prompt_prefix):
                    reference_video_name = filename
                    break
            if not reference_video_name:
                bless_cmd = f"cp {shlex.quote(generated_video_path)} {shlex.quote(reference_folder)}/"
                pytest.fail(
                    f"Reference video not found for prompt '{prompt}' under: {reference_folder}\n"
                    f"Expected name: {output_video_name}\n"
                    f"Generated video saved at: {generated_video_path}\n"
                    "To bless references, copy the generated mp4 into the reference folder.\n"
                    f"Example:\n  {bless_cmd}"
                )
            reference_video_path = os.path.join(reference_folder, reference_video_name)

        logger.info("Computing SSIM between %s and %s", reference_video_path, generated_video_path)
        ssim_values = compute_video_ssim_torchvision(reference_video_path, generated_video_path, use_ms_ssim=True)

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
            f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
            f"for {MODEL_ID} with backend {ATTENTION_BACKEND}"
        )

    finally:
        if old_backend is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = old_backend
        if old_transformers_verbosity is None:
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        else:
            os.environ["TRANSFORMERS_VERBOSITY"] = old_transformers_verbosity
