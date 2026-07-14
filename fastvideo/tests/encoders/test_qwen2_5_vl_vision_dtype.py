# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.models.encoders.qwen2_5_vl_custom import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGenerationSimple,
)


def _vision_config(torch_dtype=None):
    return SimpleNamespace(
        torch_dtype=torch_dtype,
        spatial_merge_size=2,
        patch_size=14,
        fullatt_block_indexes=[],
        window_size=112,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
        num_heads=2,
        depth=0,
        _attn_implementation="sdpa",
        out_hidden_size=8,
    )


def _full_config(torch_dtype):
    return SimpleNamespace(
        vision_config=_vision_config(),
        torch_dtype=torch_dtype,
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        pad_token_id=0,
        _attn_implementation="sdpa",
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rope_scaling=None,
    )


def test_vision_dtype_uses_parent_bfloat16_string_when_vision_dtype_missing():
    model = Qwen2_5_VisionTransformerPretrainedModel(
        _vision_config(),
        parent_torch_dtype="bfloat16",
    )

    assert model.dtype == torch.bfloat16


def test_vision_dtype_accepts_parent_torch_dtype_object():
    model = Qwen2_5_VisionTransformerPretrainedModel(
        _vision_config(),
        parent_torch_dtype=torch.bfloat16,
    )

    assert model.dtype == torch.bfloat16


def test_vision_dtype_strips_parent_dtype_string_whitespace():
    model = Qwen2_5_VisionTransformerPretrainedModel(
        _vision_config(),
        parent_torch_dtype=" torch.bfloat16 ",
    )

    assert model.dtype == torch.bfloat16


def test_vision_dtype_prefers_explicit_vision_dtype_over_parent_dtype():
    model = Qwen2_5_VisionTransformerPretrainedModel(
        _vision_config(torch_dtype="float16"),
        parent_torch_dtype="bfloat16",
    )

    assert model.dtype == torch.float16


def test_vision_dtype_falls_back_to_float32_for_missing_or_unknown_dtype():
    missing = Qwen2_5_VisionTransformerPretrainedModel(_vision_config())
    unknown = Qwen2_5_VisionTransformerPretrainedModel(
        _vision_config(),
        parent_torch_dtype="not-a-real-dtype",
    )

    assert missing.dtype == torch.float32
    assert unknown.dtype == torch.float32


def test_conditional_generation_passes_parent_dtype_to_visual_tower():
    model = Qwen2_5_VLForConditionalGenerationSimple(_full_config("torch.bfloat16"))

    assert model.visual.dtype == torch.bfloat16
