# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen3-VL image-conditioning parity for LingBot-Video TI2V."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from fastvideo.configs.models.encoders.lingbot_video import LingBotVideoQwen3VLConfig
from fastvideo.configs.pipelines.lingbot_video import preprocess_lingbot_video_prompt
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from fastvideo.models.encoders.lingbot_video import LingBotVideoQwen3VLModel
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.pipelines.basic.lingbot_video.stages import (
    LINGBOT_VIDEO_IMAGE_TEMPLATE,
    _condition_tensor_to_vlm_image,
    _preprocess_condition_image,
)
from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    OFFICIAL_DENSE,
    download_components,
)


def _load_native(device: torch.device, checkpoint: Path) -> LingBotVideoQwen3VLModel:
    """Load the converted compound encoder through FastVideo's production loader."""
    args = SimpleNamespace(
        text_encoder_cpu_offload=False,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
    )
    model = TextEncoderLoader().load_model(
        str(checkpoint / "text_encoder"),
        LingBotVideoQwen3VLConfig(),
        device,
        args,
        dtype="bf16",
        use_text_encoder_override=True,
    )
    return model.eval()


def _assert_exact(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Print exact drift metrics and require zero differing hidden-state values."""
    difference = (actual.float() - expected.float()).abs()
    print(
        f"{name}: equal={torch.equal(actual, expected)} "
        f"differing={torch.count_nonzero(actual != expected).item()} "
        f"max_abs={difference.max().item():.8f} mean_abs={difference.mean().item():.8f}"
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3-VL parity requires CUDA")
def test_lingbot_video_ti2v_text_encoder_exact_parity() -> None:
    """Require exact text-only, image-conditioned, and long-prompt hidden states."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on a scheduled GPU")
    device = torch.device("cuda")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
    local_official = os.environ.get("LINGBOT_VIDEO_OFFICIAL_CHECKPOINT")
    official_checkpoint = (
        Path(local_official)
        if local_official
        else download_components(OFFICIAL_DENSE, "text_encoder", "processor")
    )
    local_converted = os.environ.get("LINGBOT_VIDEO_TI2V_CHECKPOINT")
    converted_checkpoint = Path(local_converted) if local_converted else download_components(FASTVIDEO_DENSE, "text_encoder")
    processor = AutoProcessor.from_pretrained(official_checkpoint / "processor", local_files_only=True)
    y, x = np.mgrid[0:941, 0:1672]
    image = Image.fromarray(
        np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8),
        mode="RGB",
    )
    condition = _preprocess_condition_image(image, 480, 832)
    vlm_image = _condition_tensor_to_vlm_image(condition, patch_size=16)
    prompt = preprocess_lingbot_video_prompt(
        LINGBOT_VIDEO_IMAGE_TEMPLATE + "A runner and a white robot move toward the camera."
    )
    inputs = processor(
        text=[prompt],
        images=[vlm_image],
        videos=None,
        video_metadata=None,
        do_resize=False,
        truncation=True,
        max_length=37698,
        padding="longest",
        return_tensors="pt",
    ).to(device)
    official = Qwen3VLForConditionalGeneration.from_pretrained(
        official_checkpoint / "text_encoder",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    native = _load_native(device, converted_checkpoint)
    with torch.no_grad():
        expected = official(**inputs, output_hidden_states=True, use_cache=False).hidden_states[-1]
        actual = native(**inputs, output_hidden_states=True).hidden_states[-1]
    _assert_exact("image_conditioned", expected, actual)
    text_inputs = processor(
        text=[preprocess_lingbot_video_prompt("A runner and a white robot move toward the camera.")],
        images=None,
        videos=None,
        do_resize=False,
        truncation=True,
        max_length=37698,
        padding="longest",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        expected_text = official(
            **text_inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1]
        actual_text = native(
            **text_inputs,
            output_hidden_states=True,
        ).hidden_states[-1]
    _assert_exact("text_only", expected_text, actual_text)
    long_prompt = " ".join(
        ["A runner and a white robot move toward the camera under cherry blossoms."] * 160
    )
    long_inputs = processor(
        text=[preprocess_lingbot_video_prompt(LINGBOT_VIDEO_IMAGE_TEMPLATE + long_prompt)],
        images=[vlm_image],
        videos=None,
        video_metadata=None,
        do_resize=False,
        truncation=True,
        max_length=37698,
        padding="longest",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        expected_long = official(
            **long_inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1]
        actual_long = native(
            **long_inputs,
            output_hidden_states=True,
        ).hidden_states[-1]
    _assert_exact("long_image_conditioned", expected_long, actual_long)
