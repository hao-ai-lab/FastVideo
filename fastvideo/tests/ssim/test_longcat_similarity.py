# SPDX-License-Identifier: Apache-2.0

import os
import re

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results

logger = init_logger(__name__)


def _sanitize_filename_component(name: str) -> str:
    """Sanitize filename to remove invalid characters (same logic as VideoGenerator)."""
    sanitized = re.sub(r'[\\/:*?"<>|]', '', name)
    sanitized = sanitized.strip().strip('.')
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized or "video"


def _looks_like_local_path(path: str) -> bool:
    return path.startswith(("/", "./", "../", "weights/", "data/"))


def _resolve_longcat_paths() -> tuple[str, str | None, str | None]:
    """Return (model_path, distill_lora_path_or_none, refine_lora_path_or_none).

    Skips if required local paths are missing.
    """
    model_path = os.getenv("FASTVIDEO_LONGCAT_MODEL_PATH", "weights/longcat-native")
    if _looks_like_local_path(model_path) and not os.path.exists(model_path):
        pytest.skip(
            f"LongCat model path not found: {model_path}. "
            "Set FASTVIDEO_LONGCAT_MODEL_PATH to your converted weights folder."
        )

    default_distill_lora = os.path.join(model_path, "lora", "distilled")
    distill_lora_path = os.getenv("FASTVIDEO_LONGCAT_LORA_PATH", default_distill_lora)

    if distill_lora_path and _looks_like_local_path(distill_lora_path) and not os.path.exists(distill_lora_path):
        # Distill SSIM test is intended to exercise the distilled LoRA;
        # if it's missing locally, skip rather than silently testing base.
        pytest.skip(
            f"LongCat distilled LoRA path not found: {distill_lora_path}. "
            "Set FASTVIDEO_LONGCAT_LORA_PATH or generate the distilled adapter."
        )

    default_refine_lora = os.path.join(model_path, "lora", "refinement")
    refine_lora_path = os.getenv("FASTVIDEO_LONGCAT_REFINE_LORA_PATH", default_refine_lora)
    # Refinement SSIM test needs the refinement LoRA; if missing, it will skip at runtime.
    if refine_lora_path and _looks_like_local_path(refine_lora_path) and not os.path.exists(refine_lora_path):
        refine_lora_path = None

    return model_path, distill_lora_path, refine_lora_path


device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
else:
    raise ValueError(f"Unsupported device for ssim tests: {device_name}")


LONGCAT_DISTILL_PARAMS = {
    "num_gpus": 1,
    "sp_size": 1,
    "tp_size": 1,
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 16,
    "guidance_scale": 1.0,
    "fps": 15,
    "seed": 42,
    "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
}

LONGCAT_REFINE_PARAMS = {
    "num_gpus": 1,
    "sp_size": 1,
    "tp_size": 1,
    "height": 720,
    "width": 1280,
    # LongCat refine doubles frames when spatial_refine_only=False
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "fps": 30,
    "seed": 42,
    "t_thresh": 0.5,
    "spatial_refine_only": False,
    "num_cond_frames": 0,
    # BSA settings from the official script
    "enable_bsa": True,
    "bsa_sparsity": 0.875,
    "bsa_chunk_q": [4, 4, 8],
    "bsa_chunk_k": [4, 4, 8],
}

LONGCAT_TEST_PROMPTS = [
    "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.",
]


@pytest.mark.parametrize("prompt", LONGCAT_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN", "TORCH_SDPA"])
def test_longcat_distill_similarity(prompt: str, ATTENTION_BACKEND: str):
    """Generate LongCat (distilled) video and compare with reference via MS-SSIM."""
    model_path, distill_lora_path, _ = _resolve_longcat_paths()

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_id = "LongCat-native-distill"
    base_output_dir = os.path.join(script_dir, "generated_videos", model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND)
    os.makedirs(output_dir, exist_ok=True)

    # Use an explicit file path to match VideoGenerator's sanitization behavior.
    output_video_stem = _sanitize_filename_component(prompt[:100].strip())
    generated_video_path = os.path.join(output_dir, f"{output_video_stem}.mp4")

    init_kwargs = {
        "num_gpus": LONGCAT_DISTILL_PARAMS["num_gpus"],
        "sp_size": LONGCAT_DISTILL_PARAMS["sp_size"],
        "tp_size": LONGCAT_DISTILL_PARAMS["tp_size"],
        # Keep defaults consistent with most tests (memory friendly).
        "dit_cpu_offload": True,
        # Explicitly disable BSA for the distill stage (matches official scripts).
        "enable_bsa": False,
    }

    generator = VideoGenerator.from_pretrained(model_path=model_path, **init_kwargs)
    if distill_lora_path:
        generator.set_lora_adapter(lora_nickname="distilled", lora_path=distill_lora_path)

    generator.generate_video(
        prompt,
        output_path=generated_video_path,
        height=LONGCAT_DISTILL_PARAMS["height"],
        width=LONGCAT_DISTILL_PARAMS["width"],
        num_frames=LONGCAT_DISTILL_PARAMS["num_frames"],
        num_inference_steps=LONGCAT_DISTILL_PARAMS["num_inference_steps"],
        guidance_scale=LONGCAT_DISTILL_PARAMS["guidance_scale"],
        fps=LONGCAT_DISTILL_PARAMS["fps"],
        seed=LONGCAT_DISTILL_PARAMS["seed"],
        negative_prompt=LONGCAT_DISTILL_PARAMS["negative_prompt"],
    )

    generator.shutdown()

    assert os.path.exists(
        generated_video_path), f"Output video was not generated at {generated_video_path}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id,
                                    ATTENTION_BACKEND)

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if not filename.endswith(".mp4"):
            continue
        base_filename = filename[:-4]
        if base_filename.startswith(output_video_stem) or _sanitize_filename_component(
                base_filename) == output_video_stem:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt[:100]} with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError("Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)

    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path,
                                 LONGCAT_DISTILL_PARAMS["num_inference_steps"],
                                 prompt)

    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.93
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")


@pytest.mark.parametrize("prompt", LONGCAT_TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_longcat_refine_similarity(prompt: str, ATTENTION_BACKEND: str):
    """Generate LongCat refinement (480p->720p) and compare with reference via MS-SSIM."""
    model_path, distill_lora_path, refine_lora_path = _resolve_longcat_paths()
    if refine_lora_path is None:
        pytest.skip(
            "LongCat refinement LoRA path not found. "
            "Set FASTVIDEO_LONGCAT_REFINE_LORA_PATH or place it under "
            "<model_path>/lora/refinement."
        )

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Stage 1 (distill): generate a 480p base video to refine from ---
    stage1_model_id = "LongCat-native-distill"
    stage1_dir = os.path.join(script_dir, "generated_videos", stage1_model_id,
                              ATTENTION_BACKEND)
    os.makedirs(stage1_dir, exist_ok=True)

    stage1_video_stem = _sanitize_filename_component(prompt[:100].strip())
    stage1_video_path = os.path.join(stage1_dir, f"{stage1_video_stem}.mp4")

    stage1_init_kwargs = {
        "num_gpus": LONGCAT_DISTILL_PARAMS["num_gpus"],
        "sp_size": LONGCAT_DISTILL_PARAMS["sp_size"],
        "tp_size": LONGCAT_DISTILL_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "enable_bsa": False,
    }
    stage1_generator = VideoGenerator.from_pretrained(model_path=model_path,
                                                      **stage1_init_kwargs)
    stage1_generator.set_lora_adapter(lora_nickname="distilled",
                                      lora_path=distill_lora_path)
    stage1_generator.generate_video(
        prompt,
        output_path=stage1_video_path,
        height=LONGCAT_DISTILL_PARAMS["height"],
        width=LONGCAT_DISTILL_PARAMS["width"],
        num_frames=LONGCAT_DISTILL_PARAMS["num_frames"],
        num_inference_steps=LONGCAT_DISTILL_PARAMS["num_inference_steps"],
        guidance_scale=LONGCAT_DISTILL_PARAMS["guidance_scale"],
        fps=LONGCAT_DISTILL_PARAMS["fps"],
        seed=LONGCAT_DISTILL_PARAMS["seed"],
        negative_prompt=LONGCAT_DISTILL_PARAMS["negative_prompt"],
    )
    stage1_generator.shutdown()

    assert os.path.exists(stage1_video_path), (
        f"Stage1 video was not generated at {stage1_video_path}")

    # --- Stage 2 (refine): 480p -> 720p (spatial + temporal by default) ---
    model_id = "LongCat-native-refine"
    output_dir = os.path.join(script_dir, "generated_videos", model_id,
                              ATTENTION_BACKEND)
    os.makedirs(output_dir, exist_ok=True)

    refine_video_stem = stage1_video_stem
    generated_video_path = os.path.join(output_dir, f"{refine_video_stem}.mp4")

    refine_init_kwargs = {
        "num_gpus": LONGCAT_REFINE_PARAMS["num_gpus"],
        "sp_size": LONGCAT_REFINE_PARAMS["sp_size"],
        "tp_size": LONGCAT_REFINE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "vae_cpu_offload": False,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": False,
        "enable_bsa": LONGCAT_REFINE_PARAMS["enable_bsa"],
        "bsa_sparsity": LONGCAT_REFINE_PARAMS["bsa_sparsity"],
        "bsa_chunk_q": LONGCAT_REFINE_PARAMS["bsa_chunk_q"],
        "bsa_chunk_k": LONGCAT_REFINE_PARAMS["bsa_chunk_k"],
    }
    refine_generator = VideoGenerator.from_pretrained(model_path=model_path,
                                                      **refine_init_kwargs)
    refine_generator.set_lora_adapter(lora_nickname="refinement",
                                      lora_path=refine_lora_path)
    refine_generator.generate_video(
        prompt,
        output_path=generated_video_path,
        refine_from=stage1_video_path,
        t_thresh=LONGCAT_REFINE_PARAMS["t_thresh"],
        spatial_refine_only=LONGCAT_REFINE_PARAMS["spatial_refine_only"],
        num_cond_frames=LONGCAT_REFINE_PARAMS["num_cond_frames"],
        height=LONGCAT_REFINE_PARAMS["height"],
        width=LONGCAT_REFINE_PARAMS["width"],
        num_inference_steps=LONGCAT_REFINE_PARAMS["num_inference_steps"],
        fps=LONGCAT_REFINE_PARAMS["fps"],
        guidance_scale=LONGCAT_REFINE_PARAMS["guidance_scale"],
        seed=LONGCAT_REFINE_PARAMS["seed"],
    )
    refine_generator.shutdown()

    assert os.path.exists(
        generated_video_path), f"Output video was not generated at {generated_video_path}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id,
                                    ATTENTION_BACKEND)
    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if not filename.endswith(".mp4"):
            continue
        base_filename = filename[:-4]
        if base_filename.startswith(refine_video_stem) or _sanitize_filename_component(
                base_filename) == refine_video_stem:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(
            f"Reference video not found for prompt: {prompt[:100]} with backend: {ATTENTION_BACKEND}"
        )
        raise FileNotFoundError("Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)

    logger.info(
        f"Computing SSIM between {reference_video_path} and {generated_video_path}"
    )
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)
    mean_ssim = ssim_values[0]
    logger.info(f"SSIM mean value: {mean_ssim}")

    success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                 generated_video_path,
                                 LONGCAT_REFINE_PARAMS["num_inference_steps"],
                                 prompt)
    if not success:
        logger.error("Failed to write SSIM results to file")

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")
