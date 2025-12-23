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


def _resolve_longcat_model_path() -> str:
    """Resolve LongCat model path.

    Logic: Get base path from MODEL_PATH and append 'longcat-native'.
    """
    base_path = os.getenv("MODEL_PATH", "../weights")
    
    model_path = os.path.join(base_path, "longcat-native")

    if _looks_like_local_path(model_path) and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LongCat model path not found: {model_path}. "
            f"Please ensure MODEL_PATH is set correctly (Current base: {base_path})."
        )
    return model_path

def _resolve_longcat_distill_lora_path(model_path: str) -> str:
    """Resolve distilled LoRA path for LongCat distill/refine stage1.

    Strict: if the path looks local but does not exist, raise immediately.
    """
    default_distill_lora = os.path.join(model_path, "lora", "distilled")
    lora_path = os.getenv("FASTVIDEO_LONGCAT_LORA_PATH", default_distill_lora)
    if _looks_like_local_path(lora_path) and not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LongCat distilled LoRA path not found: {lora_path}. "
            "Set FASTVIDEO_LONGCAT_LORA_PATH or place it under <model_path>/lora/distilled."
        )
    return lora_path


def _resolve_longcat_refine_lora_path(model_path: str) -> str:
    """Resolve refinement LoRA path for LongCat refine stage.

    Strict: if the path looks local but does not exist, raise immediately.
    """
    default_refine_lora = os.path.join(model_path, "lora", "refinement")
    lora_path = os.getenv("FASTVIDEO_LONGCAT_REFINE_LORA_PATH", default_refine_lora)
    if _looks_like_local_path(lora_path) and not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LongCat refinement LoRA path not found: {lora_path}. "
            "Set FASTVIDEO_LONGCAT_REFINE_LORA_PATH or place it under <model_path>/lora/refinement."
        )
    return lora_path


def _get_generated_video_path(script_dir: str, model_id: str, attention_backend: str,
                             prompt: str):
    output_dir = os.path.join(script_dir, "generated_videos", model_id, attention_backend)
    output_video_stem = _sanitize_filename_component(prompt[:100].strip())
    generated_video_path = os.path.join(output_dir, f"{output_video_stem}.mp4")
    return output_dir, output_video_stem, generated_video_path


def _should_reuse_existing_video(path: str) -> bool:
    if not os.path.exists(path):
        return False
    # Avoid reusing empty/corrupted files from failed runs.
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _ensure_longcat_distill_video(prompt: str, attention_backend: str, model_path: str,
                                  distill_lora_path: str):
    """Ensure the LongCat distill (stage1) video exists and return its path.

    This is shared by the distill SSIM test and the refine SSIM test (stage1 input).
    Set FASTVIDEO_SSIM_FORCE_REGEN=1 to always regenerate.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LongCat-native-distill"

    output_dir, output_video_stem, generated_video_path = _get_generated_video_path(
        script_dir, model_id, attention_backend, prompt
    )
    os.makedirs(output_dir, exist_ok=True)

    force_regen = os.getenv("FASTVIDEO_SSIM_FORCE_REGEN", "").strip() in {"1", "true", "True"}
    if not force_regen and _should_reuse_existing_video(generated_video_path):
        logger.info(f"Reusing existing distill video: {generated_video_path}")
        return output_dir, output_video_stem, generated_video_path

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

    return output_dir, output_video_stem, generated_video_path


device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
elif "H100" in device_name: # temporary
    device_reference_folder = "H100" + device_reference_folder_suffix
else:
    raise ValueError(f"Unsupported device for ssim tests: {device_name}")


LONGCAT_DISTILL_PARAMS = {
    "num_gpus": 4,
    "sp_size": 1,
    "tp_size": 1,
    # NOTE: Speed-optimized settings for SSIM tests.
    # We intentionally reduce denoising steps and resolution (must be multiples of 4)
    # to keep runtime manageable in CI.
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 2,
    "guidance_scale": 1.0,
    "fps": 15,
    "seed": 42,
    "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
}

LONGCAT_BASE_PARAMS = {
    "num_gpus": 4,
    "sp_size": 1,
    "tp_size": 1,
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 4,
    "guidance_scale": 4.0,
    "fps": 15,
    "seed": 42,
    "negative_prompt": LONGCAT_DISTILL_PARAMS["negative_prompt"],
}

LONGCAT_REFINE_PARAMS = {
    "num_gpus": 4,
    "sp_size": 1,
    "tp_size": 1,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 2,
    "guidance_scale": 1.0,
    "fps": 15,
    "seed": 42,
    "t_thresh": 0.5,
    "spatial_refine_only": True,
    "num_cond_frames": 0,
    # BSA settings from the official script
    "enable_bsa": True,
    "bsa_sparsity": 0.875,
    "bsa_chunk_q": [4, 4, 8],
    "bsa_chunk_k": [4, 4, 8],
}

LONGCAT_TEST_PROMPTS = [
    "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.",
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature.",
    "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. Mid-shot, warm and cheerful tones.",
    "A superintelligent humanoid robot waking up. The robot has a sleek metallic body with futuristic design features. Its glowing red eyes are the focal point, emanating a sharp, intense light as it powers on. The scene is set in a dimly lit, high-tech laboratory filled with glowing control panels, robotic arms, and holographic screens. The setting emphasizes advanced technology and an atmosphere of mystery. The ambiance is eerie and dramatic, highlighting the moment of awakening and the robots immense intelligence. Photorealistic style with a cinematic, dark sci-fi aesthetic. Aspect ratio: 16:9 --v 6.1",
    "fox in the forest close-up quickly turned its head to the left",
    "Man walking his dog in the woods on a hot sunny day",
    "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.",
    "A serene mountain lake reflects the towering peaks of a distant mountain range, creating a mirror-like surface. The water is calm and clear, with gentle ripples created by a small boat drifting on the surface. The scene is bathed in soft, golden light, casting a serene and tranquil atmosphere. Mid-shot framing, with a focus on the calm water and the distant mountains.",
    "A young woman with long, flowing hair sits on a park bench, her eyes closed and her face relaxed. The background features a lush green park with trees and flowers, creating a peaceful and serene atmosphere. The lighting is soft and natural, with a warm glow from the sun filtering through the trees. Mid-shot framing, with a focus on the woman's face and the peaceful park.",
    "A group of people gathered around a table, each holding a drink in their hand. The background features a vibrant cityscape with tall buildings and neon lights, creating a lively and energetic atmosphere. The lighting is bright and colorful, with a focus on the people and the drinks. Mid-shot framing, with a focus on the people and the drinks.",
    "A young woman with long, flowing hair sits on a park bench, her eyes closed and her face relaxed. The background features a lush green park with trees and flowers, creating a peaceful and serene atmosphere. The lighting is soft and natural, with a warm glow from the sun filtering through the trees. Mid-shot framing, with a focus on the woman's face and the peaceful park.",
]

_MAX_PROMPTS = int(os.getenv("FASTVIDEO_SSIM_MAX_PROMPTS", "1"))
if _MAX_PROMPTS <= 0:
    _MAX_PROMPTS = 1
TEST_PROMPTS = LONGCAT_TEST_PROMPTS[:_MAX_PROMPTS]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN", "TORCH_SDPA"])
def test_longcat_distill_similarity(prompt: str, ATTENTION_BACKEND: str):
    """Generate LongCat (distilled) video and compare with reference via MS-SSIM."""
    model_path = _resolve_longcat_model_path()
    distill_lora_path = _resolve_longcat_distill_lora_path(model_path)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LongCat-native-distill"
    output_dir, output_video_stem, generated_video_path = _ensure_longcat_distill_video(
        prompt=prompt,
        attention_backend=ATTENTION_BACKEND,
        model_path=model_path,
        distill_lora_path=distill_lora_path,
    )

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

    # With reduced steps/resolution, allow a slightly lower SSIM threshold.
    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN", "TORCH_SDPA"])
def test_longcat_base_similarity(prompt: str, ATTENTION_BACKEND: str):
    """Generate LongCat base (no LoRA) video and compare with reference via MS-SSIM."""
    model_path = _resolve_longcat_model_path()

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_id = "LongCat-native-base"
    output_dir = os.path.join(script_dir, "generated_videos", model_id, ATTENTION_BACKEND)
    os.makedirs(output_dir, exist_ok=True)

    output_video_stem = _sanitize_filename_component(prompt[:100].strip())
    generated_video_path = os.path.join(output_dir, f"{output_video_stem}.mp4")

    init_kwargs = {
        "num_gpus": LONGCAT_BASE_PARAMS["num_gpus"],
        "sp_size": LONGCAT_BASE_PARAMS["sp_size"],
        "tp_size": LONGCAT_BASE_PARAMS["tp_size"],
        "dit_cpu_offload": True,
        "enable_bsa": False,
    }
    generator = VideoGenerator.from_pretrained(model_path=model_path, **init_kwargs)
    generator.generate_video(
        prompt,
        output_path=generated_video_path,
        height=LONGCAT_BASE_PARAMS["height"],
        width=LONGCAT_BASE_PARAMS["width"],
        num_frames=LONGCAT_BASE_PARAMS["num_frames"],
        num_inference_steps=LONGCAT_BASE_PARAMS["num_inference_steps"],
        guidance_scale=LONGCAT_BASE_PARAMS["guidance_scale"],
        fps=LONGCAT_BASE_PARAMS["fps"],
        seed=LONGCAT_BASE_PARAMS["seed"],
        negative_prompt=LONGCAT_BASE_PARAMS["negative_prompt"],
    )
    generator.shutdown()

    assert os.path.exists(
        generated_video_path), f"Output video was not generated at {generated_video_path}"

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id, ATTENTION_BACKEND)
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
                                 LONGCAT_BASE_PARAMS["num_inference_steps"],
                                 prompt)
    if not success:
        logger.error("Failed to write SSIM results to file")

    # With reduced steps/resolution, allow a slightly lower SSIM threshold.
    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_longcat_refine_similarity(prompt: str, ATTENTION_BACKEND: str):
    """Generate LongCat refinement (480p->720p) and compare with reference via MS-SSIM."""
    model_path = _resolve_longcat_model_path()
    distill_lora_path = _resolve_longcat_distill_lora_path(model_path)
    refine_lora_path = _resolve_longcat_refine_lora_path(model_path)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Stage 1 (distill): reuse if already generated; otherwise generate ---
    _, stage1_video_stem, stage1_video_path = _ensure_longcat_distill_video(
        prompt=prompt,
        attention_backend=ATTENTION_BACKEND,
        model_path=model_path,
        distill_lora_path=distill_lora_path,
    )

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
        # VAE can be large at 720p decode/encode paths; keep it CPU-offloaded.
        "vae_cpu_offload": False,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": True,
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

    # Refinement is especially sensitive; keep a modest threshold for the fast config.
    min_acceptable_ssim = 0.88
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")
