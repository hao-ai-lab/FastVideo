# SPDX-License-Identifier: Apache-2.0
"""DreamX-World conversion script smoke tests."""

import json

from fastvideo.configs.pipelines.dreamx_world import make_dreamx_world_5b_cam_dit_config
from fastvideo.models.registry import _LEGACY_FAST_VIDEO_MODELS

from scripts.checkpoint_conversion.dreamx_world_to_diffusers import (
    MODEL_INDEX,
    REUSED_COMPONENTS,
    TRANSFORMER_CONFIG,
    _copy_or_link_component,
    write_model_index,
)


def test_dreamx_world_converter_writes_full_model_index_with_reused_components(tmp_path):
    component_source = tmp_path / "wan22"
    output = tmp_path / "dreamx"
    output.mkdir()

    for component in REUSED_COMPONENTS:
        component_dir = component_source / component
        component_dir.mkdir(parents=True)
        (component_dir / "config.json").write_text("{}\n")

    write_model_index(output, component_source, symlink_components=True)

    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index == MODEL_INDEX
    assert model_index["_class_name"] == "DreamXWorldPipeline"
    assert model_index["transformer"] == ["diffusers", "DreamXWorldTransformer3DModel"]
    for component in REUSED_COMPONENTS:
        assert (output / component).is_symlink()


def test_dreamx_world_transformer_config_has_camera_adapter_enabled():
    assert TRANSFORMER_CONFIG["_class_name"] == "DreamXWorldTransformer3DModel"
    assert TRANSFORMER_CONFIG["add_control_adapter"] is True
    assert TRANSFORMER_CONFIG["cam_method"] == "prope"
    assert TRANSFORMER_CONFIG["num_layers"] == 30


def test_dreamx_world_converter_transformer_config_matches_pipeline_dit_config():
    dit_config = make_dreamx_world_5b_cam_dit_config()

    assert TRANSFORMER_CONFIG["num_attention_heads"] == dit_config.num_attention_heads
    assert TRANSFORMER_CONFIG["attention_head_dim"] == dit_config.attention_head_dim
    assert TRANSFORMER_CONFIG["in_channels"] == dit_config.in_channels
    assert TRANSFORMER_CONFIG["out_channels"] == dit_config.out_channels
    assert TRANSFORMER_CONFIG["ffn_dim"] == dit_config.ffn_dim
    assert TRANSFORMER_CONFIG["num_layers"] == dit_config.num_layers
    assert TRANSFORMER_CONFIG["cross_attn_norm"] == dit_config.cross_attn_norm
    assert TRANSFORMER_CONFIG["qk_norm"] == dit_config.qk_norm
    assert TRANSFORMER_CONFIG["add_control_adapter"] == dit_config.add_control_adapter
    assert TRANSFORMER_CONFIG["cam_method"] == dit_config.cam_method
    assert TRANSFORMER_CONFIG["attn_compress"] == dit_config.attn_compress
    assert TRANSFORMER_CONFIG["cam_self_attn_layers"] == dit_config.cam_self_attn_layers


def test_dreamx_world_model_index_component_classes_are_registered():
    for component in ("scheduler", "text_encoder", "transformer", "vae"):
        class_name = MODEL_INDEX[component][1]
        assert class_name in _LEGACY_FAST_VIDEO_MODELS


def test_dreamx_world_copy_or_link_component_keeps_broken_symlink(tmp_path):
    component_source = tmp_path / "wan22"
    src = component_source / "scheduler"
    src.mkdir(parents=True)
    output = tmp_path / "dreamx"
    output.mkdir()
    dst = output / "scheduler"
    dst.symlink_to(tmp_path / "missing_scheduler", target_is_directory=True)

    _copy_or_link_component("scheduler", component_source, output, symlink=True)

    assert dst.is_symlink()
