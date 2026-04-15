# SPDX-License-Identifier: Apache-2.0
import os

import torch
import pytest

from fastvideo import VideoGenerator
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.logger import init_logger
from fastvideo.models.dits.matrixgame.utils import create_action_presets
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import (
    compute_video_ssim_torchvision,
    write_ssim_results,
)
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

REQUIRED_GPUS = 1

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

MATRIXGAME_PARAMS = {
    "num_gpus": 1,
    "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
    "height": 352,
    "width": 640,
    "num_frames": 117,
    "num_inference_steps": 10,
    "seed": 1024,
    "keyboard_dim": 4,
}
_MATRIXGAME_FULL_QUALITY_DEFAULTS = SamplingParam.from_pretrained(
    MATRIXGAME_PARAMS["model_path"])
MATRIXGAME_FULL_QUALITY_PARAMS = {
    "num_gpus": MATRIXGAME_PARAMS["num_gpus"],
    "model_path": MATRIXGAME_PARAMS["model_path"],
    "height": _MATRIXGAME_FULL_QUALITY_DEFAULTS.height,
    "width": _MATRIXGAME_FULL_QUALITY_DEFAULTS.width,
    "num_frames": MATRIXGAME_PARAMS["num_frames"],  # default num_frames: 57
    "num_inference_steps": _MATRIXGAME_FULL_QUALITY_DEFAULTS.num_inference_steps,
    "guidance_scale": _MATRIXGAME_FULL_QUALITY_DEFAULTS.guidance_scale,
    "seed": _MATRIXGAME_FULL_QUALITY_DEFAULTS.seed,
    "keyboard_dim": MATRIXGAME_PARAMS["keyboard_dim"],
}


MODEL_TO_PARAMS = {
    "Matrix-Game-2.0-Diffusers-Base": MATRIXGAME_PARAMS,
}
FULL_QUALITY_MODEL_TO_PARAMS = {
    "Matrix-Game-2.0-Diffusers-Base": MATRIXGAME_FULL_QUALITY_PARAMS,
}

I2V_MODEL_TO_PARAMS = {}

TEST_PROMPTS = [
    "",  # MatrixGame is I2V with action conditions, no text prompt
]

TEST_IMAGE_PATHS = [
    "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0002.png",
]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_matrixgame_similarity(prompt, ATTENTION_BACKEND, model_id):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = build_generated_output_dir(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )
    output_video_name = "output.mp4"

    os.makedirs(output_dir, exist_ok=True)

    params_map = select_ssim_params(
        MODEL_TO_PARAMS,
        FULL_QUALITY_MODEL_TO_PARAMS,
    )
    BASE_PARAMS = params_map[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    # Create action conditions for MatrixGame
    actions = create_action_presets(
        BASE_PARAMS["num_frames"], keyboard_dim=BASE_PARAMS["keyboard_dim"], seed=BASE_PARAMS["seed"]
    )
    latent_frames = (BASE_PARAMS["num_frames"] - 1) // 4 + 1
    grid_sizes = torch.tensor([latent_frames, 44, 80])

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "use_fsdp_inference": True,
        "dit_layerwise_offload": False,
        "dit_cpu_offload": False,
        "vae_cpu_offload": False,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": True,
    }

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "image_path": TEST_IMAGE_PATHS[0],
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "seed": BASE_PARAMS["seed"],
        "mouse_cond": actions["mouse"].unsqueeze(0),
        "keyboard_cond": actions["keyboard"].unsqueeze(0),
        "grid_sizes": grid_sizes,
        "save_video": True,
    }

    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    generator.generate_video(prompt, **generation_kwargs)

    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()

    assert os.path.exists(output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        ATTENTION_BACKEND,
    )

    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4"):
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for model: {model_id} with backend: {ATTENTION_BACKEND}")
        raise FileNotFoundError("Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(f"Computing SSIM between {reference_video_path} and {generated_video_path}")
    ssim_values = compute_video_ssim_torchvision(reference_video_path, generated_video_path, use_ms_ssim=True)

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

    min_acceptable_ssim = 0.98
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for {model_id} with backend {ATTENTION_BACKEND}"
    )
