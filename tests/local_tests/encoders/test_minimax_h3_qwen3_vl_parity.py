# SPDX-License-Identifier: Apache-2.0
"""Production-loader parity for MiniMax-H3's Qwen3-VL conditioner.

The test compares the exact Transformers base model used by the official
pipeline with FastVideo's production ``TextEncoderLoader`` path.  It covers
the three numerical branches the H3 pipelines exercise: text-only tokens,
image features, and video features.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch.testing import assert_close

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.loader.component_loader import TextEncoderLoader


PARITY_SCOPE = "production_loader"
MINIMAX_H3_TEXT_ENCODER_LAYER = 50


def _require_assets() -> tuple[torch.device, Path]:
    if os.environ.get("MINIMAX_H3_RUN_ENCODER_PARITY") != "1":
        pytest.skip("set MINIMAX_H3_RUN_ENCODER_PARITY=1 on an allocated GPU node")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.fail("MiniMax-H3 Qwen3-VL parity requires a bf16-capable CUDA GPU", pytrace=False)

    model_root = os.environ.get("MINIMAX_H3_MODEL_ROOT")
    if not model_root:
        pytest.fail("set MINIMAX_H3_MODEL_ROOT to the downloaded MiniMax-H3 checkpoint", pytrace=False)
    root = Path(model_root)
    required = (root / "text_encoder", root / "tokenizer", root / "processor")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        pytest.fail(f"MiniMax-H3 component directories are missing: {missing}", pytrace=False)
    return torch.device("cuda"), root


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if os.environ.get("MINIMAX_H3_RUN_ENCODER_PARITY") != "1":
        yield
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29623")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _reclaim_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"MiniMax-H3 tokenizer does not define {token!r}")
    return int(token_id)


def _mm_token_type_ids(processor: Any, token_ids: list[int]) -> torch.Tensor:
    create_ids = getattr(processor, "create_mm_token_type_ids", None)
    if callable(create_ids):
        return torch.tensor(create_ids([token_ids]), dtype=torch.long)

    image_token_id = int(getattr(processor, "image_token_id", -1))
    video_token_id = int(getattr(processor, "video_token_id", -1))
    return torch.tensor(
        [[1 if token == image_token_id else 2 if token == video_token_id else 0 for token in token_ids]],
        dtype=torch.long,
    )


def _make_cases(root: Path) -> dict[str, dict[str, torch.Tensor]]:
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(root / "processor", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    prompt_ids = tokenizer("A red fox crosses fresh snow at sunrise.", add_special_tokens=False)["input_ids"]

    text_ids = list(prompt_ids)
    cases: dict[str, dict[str, torch.Tensor]] = {
        "text": {
            "input_ids": torch.tensor([text_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(text_ids), dtype=torch.long),
            "mm_token_type_ids": _mm_token_type_ids(processor, text_ids),
        }
    }

    height, width = 64, 96
    rows = np.arange(height, dtype=np.uint16)[:, None, None]
    cols = np.arange(width, dtype=np.uint16)[None, :, None]
    channels = np.arange(3, dtype=np.uint16)[None, None, :]
    pixels = ((rows * 7 + cols * 11 + channels * 53) % 256).astype(np.uint8)
    image = Image.fromarray(pixels)
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    image_grid_thw = image_inputs["image_grid_thw"]
    merge_area = int(processor.image_processor.merge_size)**2
    image_token_count = int(image_grid_thw[0].prod().item()) // merge_area
    image_ids = (
        tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
        + [_token_id(tokenizer, "<|vision_start|>")]
        + [_token_id(tokenizer, "<|image_pad|>")] * image_token_count
        + [_token_id(tokenizer, "<|vision_end|>")]
        + prompt_ids
    )
    cases["image"] = {
        "input_ids": torch.tensor([image_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(image_ids), dtype=torch.long),
        "mm_token_type_ids": _mm_token_type_ids(processor, image_ids),
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_grid_thw,
    }

    frames = np.stack([np.roll(pixels, shift=index * 3, axis=1) for index in range(4)])
    video_inputs = processor.video_processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
    video_grid_thw = video_inputs["video_grid_thw"]
    temporal_blocks, grid_height, grid_width = (int(value) for value in video_grid_thw[0])
    video_tokens_per_block = grid_height * grid_width // int(processor.video_processor.merge_size)**2
    video_ids = tokenizer("<Video 1>: ", add_special_tokens=False)["input_ids"]
    for _ in range(temporal_blocks):
        video_ids += (
            [_token_id(tokenizer, "<|vision_start|>")]
            + [_token_id(tokenizer, "<|video_pad|>")] * video_tokens_per_block
            + [_token_id(tokenizer, "<|vision_end|>")]
        )
    video_ids += prompt_ids
    cases["video"] = {
        "input_ids": torch.tensor([video_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(video_ids), dtype=torch.long),
        "mm_token_type_ids": _mm_token_type_ids(processor, video_ids),
        "pixel_values_videos": video_inputs["pixel_values_videos"],
        "video_grid_thw": video_grid_thw,
    }
    return cases


def _run_cases(
    model: torch.nn.Module,
    cases: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, ...]]:
    dtype = next(model.parameters()).dtype
    outputs: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, case in cases.items():
        inputs = {
            key: value.to(device=device, dtype=dtype if key.startswith("pixel_values") else value.dtype)
            for key, value in case.items()
        }
        with torch.inference_mode():
            result = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
            )
        assert result.hidden_states is not None
        assert len(result.hidden_states) > MINIMAX_H3_TEXT_ENCODER_LAYER
        outputs[name] = tuple(hidden_state.detach().cpu() for hidden_state in result.hidden_states)
    return outputs


def _production_loader_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            text_encoder_configs=(MiniMaxH3Qwen3VLConfig(),),
            text_encoder_precisions=("bf16",),
        ),
        text_encoder_cpu_offload=False,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
    )


def test_minimax_h3_qwen3_vl_parity() -> None:
    """Match official layer-50 states for text, image, and video inputs."""
    from transformers import Qwen3VLForConditionalGeneration

    device, root = _require_assets()
    cases = _make_cases(root)
    requested_cases = os.environ.get("MINIMAX_H3_ENCODER_PARITY_CASES")
    if requested_cases:
        names = {name.strip() for name in requested_cases.split(",") if name.strip()}
        unknown = names - cases.keys()
        if unknown:
            raise ValueError(f"Unknown MiniMax-H3 encoder parity cases: {sorted(unknown)}")
        cases = {name: case for name, case in cases.items() if name in names}

    official_full, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder",
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    load_errors = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    assert not load_errors, f"Official Qwen3-VL checkpoint did not load strictly: {load_errors}"
    official = official_full.model.eval().to(device)
    del official_full
    expected = _run_cases(official, cases, device)
    del official
    _reclaim_vram()

    production = TextEncoderLoader().load(str(root / "text_encoder"), _production_loader_args())
    assert getattr(production, "_fastvideo_input_device", device) == device
    actual = _run_cases(production, cases, device)

    assert actual.keys() == expected.keys()
    for name in expected:
        assert len(actual[name]) == len(expected[name])
        for layer, (result, reference) in enumerate(zip(actual[name], expected[name], strict=True)):
            assert_close(result, reference, atol=0.0, rtol=0.0, msg=lambda message: f"{name} layer {layer}: {message}")
        result = actual[name][MINIMAX_H3_TEXT_ENCODER_LAYER]
        reference = expected[name][MINIMAX_H3_TEXT_ENCODER_LAYER]
        drift = (result.float() - reference.float()).abs()
        print(
            f"{name}: max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f}",
            flush=True,
        )
