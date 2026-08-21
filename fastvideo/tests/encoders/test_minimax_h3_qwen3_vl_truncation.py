# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 Qwen3-VL layer truncation and slim-forward tests."""
from __future__ import annotations

import inspect
import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLLanguageModel
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER


def _small_arch(**overrides) -> MiniMaxH3Qwen3VLArchConfig:
    kwargs: dict = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=8,
        output_hidden_state_index=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
        vision_out_hidden_size=16,
    )
    kwargs.update(overrides)
    return MiniMaxH3Qwen3VLArchConfig(**kwargs)


def _small_config(**overrides) -> MiniMaxH3Qwen3VLConfig:
    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config = _small_arch(**overrides)
    return config


def test_default_matches_the_index_the_pipeline_reads() -> None:
    config = MiniMaxH3Qwen3VLArchConfig()

    assert config.output_hidden_state_index == MINIMAX_H3_TEXT_ENCODER_LAYER
    assert config.num_hidden_layers_override == MINIMAX_H3_TEXT_ENCODER_LAYER


def test_rejects_build_depth_that_cannot_reach_the_output() -> None:
    for override in (0, 4):
        with pytest.raises(ValueError, match="num_hidden_layers_override"):
            _small_arch(num_hidden_layers_override=override)


def test_rejects_output_index_above_the_checkpoint_depth() -> None:
    with pytest.raises(ValueError, match="output_hidden_state_index"):
        _small_arch(output_hidden_state_index=9, num_hidden_layers_override=None)


def test_builds_only_up_to_the_override(distributed_setup) -> None:
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=5))

    assert model.num_layers == 5
    assert len(model.layers) == 5
    assert model.norm is None


def test_override_none_keeps_the_full_stack(distributed_setup) -> None:
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=None))

    assert model.num_layers == 8
    assert model.norm is not None


def test_nominal_and_built_depths_remain_distinct(distributed_setup) -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=5))

    assert conditioner.num_hidden_layers == 8
    assert conditioner.num_built_hidden_layers == 5


def test_override_above_the_stack_does_not_over_build(distributed_setup) -> None:
    model = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=99))

    assert model.num_layers == 8
    assert model.norm is not None


def test_tapped_hidden_state_is_unchanged_by_truncation(distributed_setup) -> None:
    """The slim model returns the raw output at the selected layer."""
    tap = 5
    full = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=None))
    cut = MiniMaxH3Qwen3VLLanguageModel(_small_config(num_hidden_layers_override=tap))

    torch.manual_seed(0)
    for parameter in full.parameters():
        parameter.data.normal_(std=0.02)
    for (_, a), (_, b) in zip(full.layers[:tap].named_parameters(),
                              cut.layers[:tap].named_parameters(),
                              strict=True):
        b.data.copy_(a.data)
    torch.manual_seed(1)
    inputs_embeds = torch.randn(1, 6, 16)
    position_ids = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    with torch.no_grad():
        expected = inputs_embeds
        position_embeddings = full.rotary_emb(inputs_embeds, position_ids)
        for layer in full.layers[:tap]:
            expected = layer(expected, position_embeddings, None)
        full_out = full(inputs_embeds, position_ids, None, None, None)
        cut_out = cut(inputs_embeds, position_ids, None, None, None)

    assert torch.equal(expected, full_out)
    assert torch.equal(expected, cut_out)


def test_conditioning_stage_adapts_slim_sequence_output() -> None:
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage

    class FakeConditioner:

        dtype = torch.float32

        def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
            assert input_ids.ndim == 1
            assert not kwargs
            return torch.ones(input_ids.shape[0], 4)

    stage = MiniMaxH3ConditioningStage(conditioner=FakeConditioner(), tokenizer=None, processor=None, ref2va=False)
    embeddings, tags = stage._encode_tokens([1, 2, 3], [0, 0, 0], torch.device("cpu"))

    assert embeddings.shape == (1, 3, 4)
    assert tags.shape == (3, )


def test_conditioner_exposes_only_the_slim_forward_contract() -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    assert tuple(inspect.signature(MiniMaxH3Qwen3VLConditioner.forward).parameters) == (
        "self",
        "input_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    )


def test_truncated_model_drops_the_surplus_checkpoint_keys(distributed_setup) -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=5))

    assert conditioner._is_omitted_checkpoint_key("language_model.layers.5.mlp.gate_proj.weight")
    assert conditioner._is_omitted_checkpoint_key("language_model.layers.7.self_attn.q_proj.weight")
    assert conditioner._is_omitted_checkpoint_key("language_model.norm.weight")
    assert not conditioner._is_omitted_checkpoint_key("language_model.layers.4.mlp.gate_proj.weight")
    assert not conditioner._is_omitted_checkpoint_key("language_model.embed_tokens.weight")
    assert not conditioner._is_omitted_checkpoint_key("visual.blocks.0.attn.qkv.weight")
    assert not conditioner._is_omitted_checkpoint_key("language_model.layers.8.mlp.gate_proj.weight")


def test_full_stack_filters_nothing(distributed_setup) -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=None))

    assert not conditioner._is_omitted_checkpoint_key("language_model.layers.7.mlp.gate_proj.weight")
    assert not conditioner._is_omitted_checkpoint_key("language_model.norm.weight")


def test_corrupt_layer_above_checkpoint_depth_remains_unexpected(distributed_setup) -> None:
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    conditioner = MiniMaxH3Qwen3VLConditioner(_small_config(num_hidden_layers_override=5))

    with pytest.raises(ValueError, match="Unexpected"):
        conditioner.load_weights([("language_model.layers.8.mlp.gate_proj.weight", torch.empty(1))])
