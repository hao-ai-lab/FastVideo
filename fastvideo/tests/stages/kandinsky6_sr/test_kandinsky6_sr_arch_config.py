# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR DiT arch config: release configs parse, unsupported configurations fail by name."""
from __future__ import annotations

import copy

import pytest
from k6_sr_release import DISTILLED_TRANSFORMER_CONFIG, FLOW_MATCHING_TRANSFORMER_CONFIG

from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRArchConfig, Kandinsky6SRConfig


def _arch(**overrides) -> Kandinsky6SRArchConfig:
    config = Kandinsky6SRConfig()
    config.update_model_arch({**copy.deepcopy(FLOW_MATCHING_TRANSFORMER_CONFIG), **overrides})
    return config.arch_config


@pytest.mark.parametrize("release", [FLOW_MATCHING_TRANSFORMER_CONFIG, DISTILLED_TRANSFORMER_CONFIG])
def test_release_configs_expose_the_sampling_parameters(release):
    config = Kandinsky6SRConfig()
    config.update_model_arch(copy.deepcopy(release))
    arch = config.arch_config
    assert arch.visual_size == 512
    assert arch.rope_scale_factor == (1.0, 2.0, 2.0)
    assert arch.lq_noise_scale == 0.7
    assert arch.input_channels == 2 * 64 + 1
    assert arch.num_attention_heads == 1792 // 64
    assert set(release) <= arch.known_config_keys()


@pytest.mark.parametrize("overrides, match", [
    ({"use_text": True}, "use_text"),
    ({"attribute_overrides": {"visual_cond": False}}, "attribute_overrides"),
    ({"instruct_type": "noise"}, "hybrid_anchor"),
    ({"visual_cond": False}, "hybrid_anchor"),
    ({"sr_params": {"lq_noise_type": "linear"}}, "lq_noise_type"),
    ({"sr_params": {"lq_channel_noise_scale": 0.1}}, "lq_channel_noise_scale"),
    ({"sr_params": {"cap_noise_timestep": True}}, "cap_noise_timestep"),
])
def test_unsupported_configurations_raise_and_name_the_field(overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        _arch(**overrides)


def test_temporal_patching_is_rejected():
    with pytest.raises(ValueError, match="patch_size"):
        _arch(patch_size=[2, 1, 1])


def test_param_names_mapping_maps_the_official_diffusers_names():
    import re

    mapping = Kandinsky6SRArchConfig().param_names_mapping

    def rename(key: str) -> str:
        for pattern, replacement in mapping.items():
            key = re.sub(pattern, replacement, key)
        return key

    assert rename("visual_transformer_blocks.0.feed_forward.net.0.proj.weight") == (
        "visual_transformer_blocks.0.feed_forward.mlp.fc_in.weight")
    assert rename("visual_transformer_blocks.0.feed_forward.net.2.bias") == (
        "visual_transformer_blocks.0.feed_forward.mlp.fc_out.bias")
    assert rename("time_embeddings.timestep_embedder.linear_1.weight") == "time_embeddings.in_layer.weight"
    assert rename("time_embeddings.timestep_embedder.linear_2.bias") == "time_embeddings.out_layer.bias"
    assert rename("visual_transformer_blocks.0.self_attention.to_query.weight") == (
        "visual_transformer_blocks.0.self_attention.to_query.weight")
