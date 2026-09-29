# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

WAN_VACE_MODEL_PATH = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

WAN_VACE_PARAMS = {
    "num_gpus": 1,
    "model_path": WAN_VACE_MODEL_PATH,
    "height": 480,
    "width": 832,
    "num_frames": 21,
    "num_inference_steps": 4,
    "guidance_scale": 5.0,
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
    "fps": 16,
    "neg_prompt": ("Bright tones, overexposed, static, blurred details, subtitles, style, "
                   "works, paintings, images, static, overall gray, worst quality, low "
                   "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                   "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                   "misshapen limbs, fused fingers, still picture, messy background, "
                   "three legs, many people in the background, walking backwards"),
    "text-encoder-precision": ("fp32", ),
}
_WAN_VACE_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(WAN_VACE_MODEL_PATH)
WAN_VACE_FULL_QUALITY_PARAMS = {
    "num_gpus": WAN_VACE_PARAMS["num_gpus"],
    "model_path": WAN_VACE_PARAMS["model_path"],
    "height": _WAN_VACE_FULL_QUALITY_DEFAULTS.height,
    "width": _WAN_VACE_FULL_QUALITY_DEFAULTS.width,
    "num_frames": _WAN_VACE_FULL_QUALITY_DEFAULTS.num_frames,
    "num_inference_steps": _WAN_VACE_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _WAN_VACE_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _WAN_VACE_FULL_QUALITY_DEFAULTS.seed,
    "sp_size": WAN_VACE_PARAMS["sp_size"],
    "tp_size": WAN_VACE_PARAMS["tp_size"],
    "fps": _WAN_VACE_FULL_QUALITY_DEFAULTS.fps,
    "neg_prompt": _WAN_VACE_FULL_QUALITY_DEFAULTS.negative_prompt,
    "text-encoder-precision": WAN_VACE_PARAMS["text-encoder-precision"],
}

WAN_VACE_MODEL_TO_PARAMS = {
    "Wan2.1-VACE-1.3B-diffusers": WAN_VACE_PARAMS,
}
FULL_QUALITY_WAN_VACE_MODEL_TO_PARAMS = {
    "Wan2.1-VACE-1.3B-diffusers": WAN_VACE_FULL_QUALITY_PARAMS,
}

# Reference-image conditioning: the image is VAE-encoded into the VACE control
# latents and prepended as an extra latent frame.
WAN_VACE_TEST_CASES = [
    (
        "An astronaut walking across the surface of the moon, the darkness and depth of space realised in the "
        "background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
    ),
]


@pytest.mark.parametrize(("prompt", "reference_image"), WAN_VACE_TEST_CASES)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(WAN_VACE_MODEL_TO_PARAMS.keys()))
def test_wan_vace_inference_similarity(
    prompt: str,
    reference_image: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=WAN_VACE_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_WAN_VACE_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.97,
        generation_kwargs_override={"references": [reference_image]},
    )


VIDEO_MASK_PROMPT = ("An astronaut stands on the moon next to a small silver lunar rover, the darkness and depth of "
                     "space realised in the background. High quality, ultrarealistic detail.")
VIDEO_MASK_SOURCE_IMAGE = WAN_VACE_TEST_CASES[0][1]


def _write_video(path: str, frames: np.ndarray) -> None:
    import imageio

    # Lossless H.264 keeps the control video identical across runs.
    imageio.mimwrite(path, frames, fps=16, codec="libx264", output_params=["-qp", "0"])


@pytest.fixture(scope="module")
def video_mask_inputs(tmp_path_factory) -> tuple[str, str]:
    """A slow pan over a still photo as control video, with a centre box to regenerate."""
    from fastvideo.models.vision_utils import load_image

    # The loader truncates to num_frames, so cover the longer (full-quality) setting.
    num_frames = max(int(WAN_VACE_PARAMS["num_frames"]), int(WAN_VACE_FULL_QUALITY_PARAMS["num_frames"]))
    height, width = int(WAN_VACE_PARAMS["height"]), int(WAN_VACE_PARAMS["width"])
    pan = 64
    image = np.asarray(load_image(VIDEO_MASK_SOURCE_IMAGE).resize((width + pan, height)))
    video = np.stack([image[:, i * pan // (num_frames - 1):][:, :width] for i in range(num_frames)])
    mask = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    mask[:, height // 4:3 * height // 4, width // 4:3 * width // 4] = 255
    root = tmp_path_factory.mktemp("wan_vace_video_mask")
    video_path, mask_path = str(root / "video.mp4"), str(root / "mask.mp4")
    _write_video(video_path, video)
    _write_video(mask_path, mask)
    return video_path, mask_path


@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(WAN_VACE_MODEL_TO_PARAMS.keys()))
def test_wan_vace_video_mask_inference_similarity(
    attention_backend_name: str,
    model_id: str,
    video_mask_inputs: tuple[str, str],
) -> None:
    video_path, mask_path = video_mask_inputs
    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=VIDEO_MASK_PROMPT,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=WAN_VACE_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_WAN_VACE_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.97,
        generation_kwargs_override={"video_path": video_path, "mask_path": mask_path},
    )
