# SPDX-License-Identifier: Apache-2.0
"""DreamX-World-5B autoregressive conversion smoke tests."""

import json

from fastvideo.configs.pipelines.dreamx_world import make_dreamx_world_5b_ar_dit_config
from fastvideo.models.registry import _LEGACY_FAST_VIDEO_MODELS
from scripts.checkpoint_conversion.dreamx_world_ar_to_diffusers import (
    MODEL_INDEX,
    REUSED_COMPONENTS,
    TRANSFORMER_CONFIG,
    convert_transformer,
    write_model_index,
)


def test_dreamx_world_ar_converter_writes_symlinked_transformer_and_model_index(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    raw_tensor = source / "model.safetensors"
    raw_tensor.write_bytes(b"placeholder")
    component_source = tmp_path / "wan22"
    output = tmp_path / "dreamx_ar"

    for component in REUSED_COMPONENTS:
        component_dir = component_source / component
        component_dir.mkdir(parents=True)
        (component_dir / "config.json").write_text("{}\n")

    convert_transformer(source, output, symlink_transformer=True)
    write_model_index(output, component_source, symlink_components=True)

    assert (output / "transformer" / "model.safetensors").is_symlink()
    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index == MODEL_INDEX
    assert model_index["_class_name"] == "DreamXWorldARPipeline"
    assert model_index["transformer"] == ["diffusers", "DreamXWorldARTransformer3DModel"]


def test_dreamx_world_ar_transformer_config_matches_pipeline_dit_config():
    dit_config = make_dreamx_world_5b_ar_dit_config()
    assert TRANSFORMER_CONFIG["_class_name"] == "DreamXWorldARTransformer3DModel"
    assert TRANSFORMER_CONFIG["num_attention_heads"] == dit_config.num_attention_heads
    assert TRANSFORMER_CONFIG["attention_head_dim"] == dit_config.attention_head_dim
    assert TRANSFORMER_CONFIG["in_channels"] == dit_config.in_channels
    assert TRANSFORMER_CONFIG["out_channels"] == dit_config.out_channels
    assert TRANSFORMER_CONFIG["ffn_dim"] == dit_config.ffn_dim
    assert TRANSFORMER_CONFIG["num_layers"] == dit_config.num_layers
    assert TRANSFORMER_CONFIG["local_attn_size"] == dit_config.local_attn_size
    assert TRANSFORMER_CONFIG["sink_size"] == dit_config.sink_size
    assert TRANSFORMER_CONFIG["attn_compress"] == dit_config.attn_compress
    assert tuple(TRANSFORMER_CONFIG["cam_self_attn_layers"]) == dit_config.cam_self_attn_layers


def test_dreamx_world_ar_model_index_component_classes_are_registered():
    for component in ("scheduler", "text_encoder", "transformer", "vae"):
        class_name = MODEL_INDEX[component][1]
        assert class_name in _LEGACY_FAST_VIDEO_MODELS
