# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_image_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

_LOCAL_CONVERTED_MODEL = Path("converted_weights/dreamx_world")
_MODEL_PATH = os.getenv(
    "DREAMX_WORLD_SSIM_MODEL_PATH",
    str(_LOCAL_CONVERTED_MODEL),
)

DREAMX_WORLD_PARAMS = {
    "num_gpus": 1,
    "model_path": _MODEL_PATH,
    "height": 64,
    "width": 64,
    "num_frames": 9,
    "num_inference_steps": 1,
    "guidance_scale": 1.0,
    "seed": 1024,
    "fps": 16,
}

DREAMX_WORLD_FULL_QUALITY_PARAMS = {
    **DREAMX_WORLD_PARAMS,
    "height": 480,
    "width": 832,
    "num_frames": 161,
    "num_inference_steps": 30,
    "guidance_scale": 5.0,
}

DREAMX_WORLD_MODEL_TO_PARAMS = {
    "DreamX-World-5B-Cam": DREAMX_WORLD_PARAMS,
}

FULL_QUALITY_DREAMX_WORLD_MODEL_TO_PARAMS = {
    "DreamX-World-5B-Cam": DREAMX_WORLD_FULL_QUALITY_PARAMS,
}

DREAMX_WORLD_TEST_CASES = [
    (
        "A cinematic first-person drive through a futuristic coastal city at sunrise, "
        "reflective glass towers, clean streets, soft volumetric light.",
        ("w", "d", "w"),
        (4.0, 2.0, 4.0),
    ),
]


def _write_deterministic_reference_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), color=(42, 76, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 56, 96, 96), fill=(36, 44, 52))
    draw.polygon([(0, 56), (48, 30), (96, 56)], fill=(120, 142, 158))
    draw.rectangle((34, 42, 62, 72), fill=(178, 198, 212))
    draw.line((0, 80, 96, 66), fill=(238, 209, 124), width=3)
    image.save(path)


@pytest.mark.parametrize(("prompt", "action_list", "action_speed_list"), DREAMX_WORLD_TEST_CASES)
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(DREAMX_WORLD_MODEL_TO_PARAMS.keys()))
def test_dreamx_world_inference_similarity(
    prompt: str,
    action_list: tuple[str, ...],
    action_speed_list: tuple[float, ...],
    attention_backend_name: str,
    model_id: str,
    tmp_path: Path,
) -> None:
    model_path = Path(str(DREAMX_WORLD_MODEL_TO_PARAMS[model_id]["model_path"]))
    if not model_path.exists():
        pytest.skip(
            f"DreamX-World converted model path is missing: {model_path}. "
            "Set DREAMX_WORLD_SSIM_MODEL_PATH to a FastVideo-loadable converted root."
        )

    image_path = tmp_path / "dreamx_world_ssim_input.png"
    _write_deterministic_reference_image(image_path)

    run_image_to_video_similarity_test(
        logger=logger,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        image_path=str(image_path),
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=DREAMX_WORLD_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_QUALITY_DREAMX_WORLD_MODEL_TO_PARAMS,
        min_acceptable_ssim=0.98,
        init_kwargs_override={
            "use_fsdp_inference": False,
            "dit_cpu_offload": False,
            "vae_cpu_offload": True,
            "text_encoder_cpu_offload": True,
            "pin_cpu_memory": False,
            "override_pipeline_cls_name": "DreamXWorldPipeline",
        },
        generation_kwargs_override={
            "action_list": list(action_list),
            "action_speed_list": list(action_speed_list),
        },
    )
