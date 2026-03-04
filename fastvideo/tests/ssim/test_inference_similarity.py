# SPDX-License-Identifier: Apache-2.0
import json
import os
from contextlib import contextmanager
from typing import Iterator

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.configs.sample.hunyuan import FastHunyuanSamplingParam
from fastvideo.configs.sample.ltx2 import LTX2DistilledSamplingParam
from fastvideo.configs.sample.wan import (
    WanI2V_14B_480P_SamplingParam,
    WanT2V_1_3B_SamplingParam,
)
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

REQUIRED_GPUS = 2


@contextmanager
def _attention_backend(backend: str) -> Iterator[None]:
    previous = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = previous


def _shutdown_executor(generator: VideoGenerator | None) -> None:
    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

device_reference_folder = resolve_device_reference_folder(
    (
        ("A40", "A40"),
        ("L40S", "L40S"),
        ("H200", "H200"),
    ),
    device_name=get_cuda_device_name(),
    logger=logger,
)

# Base parameters from the shell script
HUNYUAN_PARAMS = {
    "num_gpus": 4,
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 45,
    "num_inference_steps": 2,
    "guidance_scale": 1,
    "embedded_cfg_scale": 6,
    "flow_shift": 17,
    "seed": 1024,
    "sp_size": 4,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
}
_HUNYUAN_FULL_QUALITY_DEFAULTS = FastHunyuanSamplingParam()
HUNYUAN_FULL_QUALITY_PARAMS = {
    "num_gpus": HUNYUAN_PARAMS["num_gpus"],
    "model_path": HUNYUAN_PARAMS["model_path"],
    "height": _HUNYUAN_FULL_QUALITY_DEFAULTS.height,
    "width": _HUNYUAN_FULL_QUALITY_DEFAULTS.width,
    "num_frames": HUNYUAN_PARAMS["num_frames"],  # default num_frames: 125
    "num_inference_steps": _HUNYUAN_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _HUNYUAN_FULL_QUALITY_DEFAULTS.guidance_scale,
    "embedded_cfg_scale": HUNYUAN_PARAMS["embedded_cfg_scale"],
    "flow_shift": HUNYUAN_PARAMS["flow_shift"],
    "seed": _HUNYUAN_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": HUNYUAN_PARAMS["sp_size"],
    "tp_size": HUNYUAN_PARAMS["tp_size"],
    "vae_sp": HUNYUAN_PARAMS["vae_sp"],
    "fps": _HUNYUAN_FULL_QUALITY_DEFAULTS.fps,
}

WAN_T2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    "text-encoder-precision": ("fp32",)
}
_WAN_T2V_FULL_QUALITY_DEFAULTS = WanT2V_1_3B_SamplingParam()
WAN_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": WAN_T2V_PARAMS["num_gpus"],
    "model_path": WAN_T2V_PARAMS["model_path"],
    "height": _WAN_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _WAN_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": WAN_T2V_PARAMS["num_frames"],  # default num_frames: 81
    "num_inference_steps": _WAN_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _WAN_T2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "embedded_cfg_scale": WAN_T2V_PARAMS["embedded_cfg_scale"],
    "flow_shift": WAN_T2V_PARAMS["flow_shift"],
    "seed": _WAN_T2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": WAN_T2V_PARAMS["sp_size"],
    "tp_size": WAN_T2V_PARAMS["tp_size"],
    "vae_sp": WAN_T2V_PARAMS["vae_sp"],
    "fps": _WAN_T2V_FULL_QUALITY_DEFAULTS.fps,
    "neg_prompt": _WAN_T2V_FULL_QUALITY_DEFAULTS.negative_prompt,
    "text-encoder-precision": WAN_T2V_PARAMS["text-encoder-precision"],
}

WAN_I2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 2,
    "guidance_scale": 5.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    "text-encoder-precision": ("fp32",)
}
_WAN_I2V_FULL_QUALITY_DEFAULTS = WanI2V_14B_480P_SamplingParam()
WAN_I2V_FULL_QUALITY_PARAMS = {
    "num_gpus": WAN_I2V_PARAMS["num_gpus"],
    "model_path": WAN_I2V_PARAMS["model_path"],
    "height": _WAN_I2V_FULL_QUALITY_DEFAULTS.height,
    "width": _WAN_I2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": WAN_I2V_PARAMS["num_frames"],  # default num_frames: 81
    "num_inference_steps": _WAN_I2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _WAN_I2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "embedded_cfg_scale": WAN_I2V_PARAMS["embedded_cfg_scale"],
    "flow_shift": WAN_I2V_PARAMS["flow_shift"],
    "seed": _WAN_I2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": WAN_I2V_PARAMS["sp_size"],
    "tp_size": WAN_I2V_PARAMS["tp_size"],
    "vae_sp": WAN_I2V_PARAMS["vae_sp"],
    "fps": _WAN_I2V_FULL_QUALITY_DEFAULTS.fps,
    "neg_prompt": _WAN_I2V_FULL_QUALITY_DEFAULTS.negative_prompt,
    "text-encoder-precision": WAN_I2V_PARAMS["text-encoder-precision"],
}

# LTX-2 distilled one-stage params (no refine/upscale)
# Official defaults: height=512, width=768, num_frames=121, fps=24, seed=10
# Using num_frames=41 for faster CI (still valid: 41 = 8×5 + 1)
LTX2_T2V_PARAMS = {
    "num_gpus": 2,
    "model_path": "FastVideo/LTX2-Distilled-Diffusers",
    "height": 512,
    "width": 768,
    "num_frames": 41,  # Shorter for CI; official default is 121
    "num_inference_steps": 8,  # Distilled uses 8 steps
    "guidance_scale": 1.0,  # No CFG for distilled
    "embedded_cfg_scale": 6,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 1,
    "fps": 24,
    "neg_prompt": (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
        "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
        "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
        "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
        "around text, inconsistent perspective, camera shake, incorrect depth of field, "
        "background too sharp, background clutter, distracting reflections, harsh shadows, "
        "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
        "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, "
        "exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted "
        "audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
        "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
        "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    ),
    "ltx2_vae_tiling": True,
    "ltx2_vae_spatial_tile_size_in_pixels": 512,
    "ltx2_vae_spatial_tile_overlap_in_pixels": 64,
    "ltx2_vae_temporal_tile_size_in_frames": 64,
    "ltx2_vae_temporal_tile_overlap_in_frames": 24,
}
_LTX2_T2V_FULL_QUALITY_DEFAULTS = LTX2DistilledSamplingParam()
LTX2_T2V_FULL_QUALITY_PARAMS = {
    "num_gpus": LTX2_T2V_PARAMS["num_gpus"],
    "model_path": LTX2_T2V_PARAMS["model_path"],
    "height": _LTX2_T2V_FULL_QUALITY_DEFAULTS.height,
    "width": _LTX2_T2V_FULL_QUALITY_DEFAULTS.width,
    "num_frames": LTX2_T2V_PARAMS["num_frames"],  # default num_frames: 121
    "num_inference_steps": _LTX2_T2V_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _LTX2_T2V_FULL_QUALITY_DEFAULTS.guidance_scale,
    "embedded_cfg_scale": LTX2_T2V_PARAMS["embedded_cfg_scale"],
    "seed": _LTX2_T2V_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": LTX2_T2V_PARAMS["sp_size"],
    "tp_size": LTX2_T2V_PARAMS["tp_size"],
    "fps": _LTX2_T2V_FULL_QUALITY_DEFAULTS.fps,
    "neg_prompt": _LTX2_T2V_FULL_QUALITY_DEFAULTS.negative_prompt,
    "ltx2_vae_tiling": LTX2_T2V_PARAMS["ltx2_vae_tiling"],
    "ltx2_vae_spatial_tile_size_in_pixels": (
        LTX2_T2V_PARAMS["ltx2_vae_spatial_tile_size_in_pixels"]
    ),
    "ltx2_vae_spatial_tile_overlap_in_pixels": (
        LTX2_T2V_PARAMS["ltx2_vae_spatial_tile_overlap_in_pixels"]
    ),
    "ltx2_vae_temporal_tile_size_in_frames": (
        LTX2_T2V_PARAMS["ltx2_vae_temporal_tile_size_in_frames"]
    ),
    "ltx2_vae_temporal_tile_overlap_in_frames": (
        LTX2_T2V_PARAMS["ltx2_vae_temporal_tile_overlap_in_frames"]
    ),
}

MODEL_TO_PARAMS = {
    # "FastHunyuan-diffusers": HUNYUAN_PARAMS,
    "Wan2.1-T2V-1.3B-Diffusers": WAN_T2V_PARAMS,
    # "ltx2_diffusers": LTX2_T2V_PARAMS,
}
FULL_QUALITY_MODEL_TO_PARAMS = {
    # "FastHunyuan-diffusers": HUNYUAN_FULL_QUALITY_PARAMS,
    "Wan2.1-T2V-1.3B-Diffusers": WAN_T2V_FULL_QUALITY_PARAMS,
    # "ltx2_diffusers": LTX2_T2V_FULL_QUALITY_PARAMS,
}

I2V_MODEL_TO_PARAMS = {
    "Wan2.1-I2V-14B-480P-Diffusers": WAN_I2V_PARAMS,
}
FULL_QUALITY_I2V_MODEL_TO_PARAMS = {
    "Wan2.1-I2V-14B-480P-Diffusers": WAN_I2V_FULL_QUALITY_PARAMS,
}

TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    # "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]

I2V_TEST_PROMPTS = [
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
]

I2V_IMAGE_PATHS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
]



@pytest.mark.parametrize("prompt", I2V_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(I2V_MODEL_TO_PARAMS.keys()))
def test_i2v_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    assert len(I2V_TEST_PROMPTS) == len(I2V_IMAGE_PATHS), "Expect number of prompts equal to number of images"
    with _attention_backend(ATTENTION_BACKEND):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            model_id,
            ATTENTION_BACKEND,
        )
        output_video_name = f"{prompt[:100].strip()}.mp4"

        os.makedirs(output_dir, exist_ok=True)

        params_map = select_ssim_params(
            I2V_MODEL_TO_PARAMS,
            FULL_QUALITY_I2V_MODEL_TO_PARAMS,
        )
        BASE_PARAMS = params_map[model_id]
        num_inference_steps = BASE_PARAMS["num_inference_steps"]
        image_path = I2V_IMAGE_PATHS[I2V_TEST_PROMPTS.index(prompt)]

        init_kwargs = {
            "num_gpus": BASE_PARAMS["num_gpus"],
            "flow_shift": BASE_PARAMS["flow_shift"],
            "sp_size": BASE_PARAMS["sp_size"],
            "tp_size": BASE_PARAMS["tp_size"],
        }
        if BASE_PARAMS.get("vae_sp"):
            init_kwargs["vae_sp"] = True
            init_kwargs["vae_tiling"] = True
        if "text-encoder-precision" in BASE_PARAMS:
            init_kwargs["text_encoder_precisions"] = BASE_PARAMS[
                "text-encoder-precision"]

        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "output_path": output_dir,
            "image_path": image_path,
            "height": BASE_PARAMS["height"],
            "width": BASE_PARAMS["width"],
            "num_frames": BASE_PARAMS["num_frames"],
            "guidance_scale": BASE_PARAMS["guidance_scale"],
            "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
            "seed": BASE_PARAMS["seed"],
            "fps": BASE_PARAMS["fps"],
        }
        if "neg_prompt" in BASE_PARAMS:
            generation_kwargs["neg_prompt"] = BASE_PARAMS["neg_prompt"]

        generator: VideoGenerator | None = None
        try:
            generator = VideoGenerator.from_pretrained(
                model_path=BASE_PARAMS["model_path"], **init_kwargs)
            generator.generate_video(prompt, **generation_kwargs)
        finally:
            _shutdown_executor(generator)

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path, num_inference_steps,
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.97
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"

@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN", "TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_inference_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    with _attention_backend(ATTENTION_BACKEND):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            model_id,
            ATTENTION_BACKEND,
        )
        output_video_name = f"{prompt[:100].strip()}.mp4"

        os.makedirs(output_dir, exist_ok=True)

        params_map = select_ssim_params(
            MODEL_TO_PARAMS,
            FULL_QUALITY_MODEL_TO_PARAMS,
        )
        BASE_PARAMS = params_map[model_id]
        num_inference_steps = BASE_PARAMS["num_inference_steps"]

        init_kwargs = {
            "num_gpus": BASE_PARAMS["num_gpus"],
            "sp_size": BASE_PARAMS["sp_size"],
            "tp_size": BASE_PARAMS["tp_size"],
            "use_fsdp_inference": True,
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
        }
        if "flow_shift" in BASE_PARAMS:
            init_kwargs["flow_shift"] = BASE_PARAMS["flow_shift"]
        if BASE_PARAMS.get("vae_sp"):
            init_kwargs["vae_sp"] = True
            init_kwargs["vae_tiling"] = True
        if "text-encoder-precision" in BASE_PARAMS:
            init_kwargs["text_encoder_precisions"] = BASE_PARAMS[
                "text-encoder-precision"]
        # LTX2-specific VAE tiling parameters
        if BASE_PARAMS.get("ltx2_vae_tiling"):
            init_kwargs["ltx2_vae_tiling"] = True
            init_kwargs["ltx2_vae_spatial_tile_size_in_pixels"] = BASE_PARAMS.get(
                "ltx2_vae_spatial_tile_size_in_pixels", 512)
            init_kwargs["ltx2_vae_spatial_tile_overlap_in_pixels"] = BASE_PARAMS.get(
                "ltx2_vae_spatial_tile_overlap_in_pixels", 64)
            init_kwargs["ltx2_vae_temporal_tile_size_in_frames"] = BASE_PARAMS.get(
                "ltx2_vae_temporal_tile_size_in_frames", 64)
            init_kwargs[
                "ltx2_vae_temporal_tile_overlap_in_frames"] = BASE_PARAMS.get(
                    "ltx2_vae_temporal_tile_overlap_in_frames", 24)

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
        if "neg_prompt" in BASE_PARAMS:
            generation_kwargs["neg_prompt"] = BASE_PARAMS["neg_prompt"]

        generator: VideoGenerator | None = None
        try:
            generator = VideoGenerator.from_pretrained(
                model_path=BASE_PARAMS["model_path"], **init_kwargs)
            generator.generate_video(prompt, **generation_kwargs)
        finally:
            _shutdown_executor(generator)

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100].strip() in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")
    logger.info(f"Writing SSIM results to directory: {output_dir}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path, num_inference_steps,
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"
