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


def test_sliding_window_cache_check_survives_transformers_5():
    """transformers 5.0-5.5 removed `cache_utils.SlidingWindowCache`.

    The import here was unguarded and at module scope, so the whole encoder became
    unimportable on those versions -- which `pyproject.toml` admits via `transformers>=5.0.0`
    -- taking `Reason1TextEncoder`, and with it the Cosmos 2.5 and Kandinsky 5 pipelines,
    with it. transformers 5.6 restored the name as an alias for `StaticCache`.

    Whatever the guard bound, `isinstance()` has to accept it and answer False for a
    `DynamicCache`: a sliding window is a property of the cache's layers now, so no cache
    object is an instance of the old class.
    """
    from transformers import cache_utils
    from transformers.cache_utils import DynamicCache

    from fastvideo.models.encoders import qwen2_5_vl_custom

    bound = qwen2_5_vl_custom.SlidingWindowCache
    if hasattr(cache_utils, "SlidingWindowCache"):
        # The guard must not shadow a name transformers still exports.
        assert bound is cache_utils.SlidingWindowCache
    else:
        # The fallback arm: an empty tuple keeps isinstance() answering False.
        assert bound == ()
    assert isinstance(DynamicCache(), bound) is False
