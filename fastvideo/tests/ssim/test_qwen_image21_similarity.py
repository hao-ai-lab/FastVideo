# SPDX-License-Identifier: Apache-2.0
"""Single-GPU Qwen T2I and editing regressions, including exported alpha.

Requires the Qwen checkpoint, FlashAttention, and device-matched PNG references.
The shared SSIM harness compares all four RGBA channels and records metrics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import (
    DEVICE_MAPPINGS,
    run_text_to_video_similarity_test,
)
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
MIN_ACCEPTABLE_SSIM = 0.98

QWEN_IMAGE21_DEFAULT_PARAMS: dict[str, object] = {
    "model_path": MODEL_ID,
    "num_gpus": REQUIRED_GPUS,
    "sp_size": 1,
    "tp_size": 1,
    "height": 512,
    "width": 512,
    "num_frames": 1,
    "fps": 1,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 21,
}

QWEN_IMAGE21_FULL_QUALITY_PARAMS: dict[str, object] = {
    **QWEN_IMAGE21_DEFAULT_PARAMS,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 40,
}

QWEN_IMAGE21_MODEL_TO_PARAMS = {"Qwen/Qwen-Image-2.1": QWEN_IMAGE21_DEFAULT_PARAMS}
QWEN_IMAGE21_FULL_QUALITY_MODEL_TO_PARAMS = {"Qwen/Qwen-Image-2.1": QWEN_IMAGE21_FULL_QUALITY_PARAMS}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen image SSIM requires one CUDA GPU")


def _run_similarity(model_id: str, attention_backend_name: str, prompt: str,
                    references: list[str] | None = None) -> None:
    device_reference_folder = resolve_device_reference_folder(
        DEVICE_MAPPINGS + (("RTX 5090", "RTX5090"), ("RTX 4090", "RTX4090")),
        device_name=get_cuda_device_name(),
        logger=logger,
    )
    if device_reference_folder is None:
        pytest.skip(f"No Qwen image SSIM reference mapping for {get_cuda_device_name()}")

    script_dir = str(Path(__file__).resolve().parent)
    selected_params = select_ssim_params(QWEN_IMAGE21_MODEL_TO_PARAMS,
                                        QWEN_IMAGE21_FULL_QUALITY_MODEL_TO_PARAMS)[model_id]
    run_text_to_video_similarity_test(
        logger=logger,
        script_dir=script_dir,
        device_reference_folder=device_reference_folder,
        prompt=prompt,
        attention_backend_name=attention_backend_name,
        model_id=model_id,
        default_params_map=QWEN_IMAGE21_MODEL_TO_PARAMS,
        full_quality_params_map=QWEN_IMAGE21_FULL_QUALITY_MODEL_TO_PARAMS,
        min_acceptable_ssim=MIN_ACCEPTABLE_SSIM,
        media_extension=".png",
        init_kwargs_override={
            "pipeline_config": QwenImage21PipelineConfig(
                reference_resolution=selected_params["height"],
                use_kv_cache=True,
            ),
            "revision": MODEL_REVISION,
            "workload_type": "i2i" if references else "t2i",
            "use_fsdp_inference": False,
            "dit_cpu_offload": True,
            "dit_layerwise_offload": True,
            "text_encoder_cpu_offload": True,
            "vae_cpu_offload": True,
            "lazy_module_load": True,
            "pin_cpu_memory": False,
            "vae_tiling": True,
        },
        generation_kwargs_override={
            "references": references,
            "negative_prompt": None,
            "true_cfg_scale": 1.0,
            "return_frames": True,
            "save_video": True,
        },
    )

    output_dir = build_generated_output_dir(script_dir, device_reference_folder, model_id, attention_backend_name)
    with Image.open(Path(output_dir) / f"{prompt[:100].strip()}.png") as generated:
        assert generated.mode == "RGBA", "Qwen PNG export must retain its alpha channel"
        assert generated.size == (selected_params["width"], selected_params["height"])


@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(QWEN_IMAGE21_MODEL_TO_PARAMS))
def test_qwen_image21_t2i_similarity(model_id: str, attention_backend_name: str) -> None:
    _run_similarity(
        model_id,
        attention_backend_name,
        "A studio photograph of a blue ceramic vase on a white background",
    )


@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(QWEN_IMAGE21_MODEL_TO_PARAMS))
def test_qwen_image21_edit_similarity(model_id: str, attention_backend_name: str, tmp_path: Path) -> None:
    reference = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(reference)
    draw.ellipse((176, 160, 336, 432), fill=(32, 96, 224))
    draw.rectangle((224, 96, 288, 192), fill=(32, 96, 224))
    draw.ellipse((224, 88, 288, 112), fill=(32, 96, 224))
    reference_path = tmp_path / "blue-vase.png"
    reference.save(reference_path)
    _run_similarity(
        model_id,
        attention_backend_name,
        "Change the blue vase to red and keep the white background",
        references=[str(reference_path)],
    )
