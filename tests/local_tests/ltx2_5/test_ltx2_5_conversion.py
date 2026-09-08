# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from fastvideo.configs.models.encoders.gemma import LTX2GemmaConfig
from fastvideo.models.encoders.gemma import LTX2GemmaTextEncoderModel


REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERTER_PATH = REPO_ROOT / "scripts" / "checkpoint_conversion" / "convert_ltx2_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_ltx2_weights", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _uint8(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


def _save(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    config: dict | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    safetensors_metadata = dict(metadata or {})
    if config is not None:
        safetensors_metadata["config"] = json.dumps(config)
    save_file(tensors, str(path), metadata=safetensors_metadata)


def _split_sources(tmp_path: Path) -> dict[str, Path]:
    transformer_config = {
        "transformer": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "cross_attention_dim": 8,
            "caption_channels": 6,
            "audio_num_attention_heads": 1,
            "audio_attention_head_dim": 3,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_cross_attention_dim": 3,
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "apply_gated_attention": True,
            "use_prompt_adaln_single": False,
            "ff_bias": False,
            "audio_ff_bias": False,
            "use_keyframes_abs_pos_embedding": False,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": True,
            "caption_projection_second_linear": False,
            "connector_num_attention_heads": 2,
            "connector_attention_head_dim": 4,
            "connector_num_layers": 1,
            "audio_connector_num_attention_heads": 1,
            "audio_connector_attention_head_dim": 3,
            "audio_connector_num_layers": 1,
            "connector_positional_embedding_max_pos": [2048],
            "connector_apply_gated_attention": True,
            "connector_ff_bias": False,
            "frequencies_precision": "float64",
        }
    }
    transformer = tmp_path / "transformer.safetensors"
    _save(
        transformer,
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": torch.ones(2, 2),
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.weight": torch.ones(1),
            "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.weight": torch.ones(1),
        },
        config=transformer_config,
    )

    gemma_config = {
        "model_type": "gemma4_unified",
        "pad_token_id": 0,
        "eos_token_id": 2,
        "text_config": {
            "hidden_size": 6,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 1024,
            "bos_token_id": 2,
        },
    }
    text_encoder = tmp_path / "text_encoder.safetensors"
    _save(
        text_encoder,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "vision_model.embeddings.patch_embedding.weight": torch.ones(1),
            "multi_modal_projector.embedding_projection.weight": torch.ones(1),
            "audio_projector.proj.weight": torch.ones(1),
            "text_embedding_projection.video_aggregate_embed.weight": torch.ones(8, 18),
            "text_embedding_projection.video_aggregate_embed.bias": torch.ones(8),
            "text_embedding_projection.audio_aggregate_embed.weight": torch.ones(3, 18),
            "text_embedding_projection.audio_aggregate_embed.bias": torch.ones(3),
            "tokenizer_json": _uint8("{}"),
            "hf_asset__tokenizer_config.json": _uint8('{"tokenizer_class":"PreTrainedTokenizerFast"}'),
            "hf_asset__processor_config.json": _uint8("{}"),
        },
        metadata={"gemma_config": json.dumps(gemma_config)},
    )

    vae = tmp_path / "vae.safetensors"
    _save(
        vae,
        {"vae.encoder.conv.weight": torch.ones(1)},
        config={"vae": {"_class_name": "CausalVideoAutoencoder", "dims": 3}},
    )

    audio_vae = tmp_path / "audio_vae.safetensors"
    _save(
        audio_vae,
        {
            "audio_vae.decoder.conv_in.weight": torch.ones(1),
            "audio_vae.per_channel_statistics.mean-of-means": torch.ones(1),
            "vocoder.vocoder.conv_pre.weight": torch.ones(1),
            "vocoder.bwe_generator.conv_pre.weight": torch.ones(1),
        },
        config={
            "audio_vae": {"model": {"params": {"ddconfig": {}}}},
            "vocoder": {"vocoder": {"resblock": "AMP1"}, "bwe": {"resblock": "AMP1"}},
        },
    )

    spatial_upscaler = tmp_path / "spatial_upscaler.safetensors"
    _save(
        spatial_upscaler,
        {
            "model.initial_conv.weight": torch.ones(1),
            "model.0.weight": torch.ones(1),
        },
        config={
            "in_channels": 128,
            "mid_channels": 512,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
        },
    )

    distilled_lora = tmp_path / "distilled_lora.safetensors"
    _save(
        distilled_lora,
        {"diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1)},
    )
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "audio_vae": audio_vae,
        "spatial_upscaler": spatial_upscaler,
        "distilled_lora": distilled_lora,
    }


def test_split_ltx2_5_conversion_routes_components_and_metadata(tmp_path: Path) -> None:
    sources = _split_sources(tmp_path)
    output = tmp_path / "converted"

    converter.convert_split_components(
        transformer_source=sources["transformer"],
        text_encoder_source=sources["text_encoder"],
        vae_source=sources["vae"],
        audio_vae_source=sources["audio_vae"],
        spatial_upscaler_source=sources["spatial_upscaler"],
        distilled_lora_source=sources["distilled_lora"],
        output_dir=output,
        transformer_class_name="LTX2Transformer3DModel",
        variant="dev",
    )

    transformer_weights = load_file(str(output / "transformer" / "model.safetensors"))
    assert set(transformer_weights) == {"transformer_blocks.0.ff.net.0.proj.weight"}
    transformer_config = json.loads((output / "transformer" / "config.json").read_text())
    assert transformer_config["cross_attention_adaln"] is True
    assert transformer_config["use_prompt_adaln_single"] is False
    assert transformer_config["ff_bias"] is False
    assert transformer_config["audio_ff_bias"] is False
    assert transformer_config["use_keyframes_abs_pos_embedding"] is False
    assert transformer_config["connector_ff_bias"] is False
    assert transformer_config["double_precision_rope"] is True

    text_weights = load_file(str(output / "text_encoder" / "model.safetensors"))
    assert set(text_weights) == {
        "video_feature_extractor_linear.weight",
        "video_feature_extractor_linear.bias",
        "audio_feature_extractor_linear.weight",
        "audio_feature_extractor_linear.bias",
        "embeddings_connector.transformer_1d_blocks.0.weight",
        "audio_embeddings_connector.transformer_1d_blocks.0.weight",
    }
    text_config = json.loads((output / "text_encoder" / "config.json").read_text())
    assert text_config["gemma_model_path"] == "gemma"
    assert text_config["hidden_size"] == 6
    assert text_config["num_hidden_layers"] == 2
    assert text_config["feature_extractor_in_features"] == 18
    assert text_config["text_len"] == 1024
    assert text_config["bos_token_id"] == 2
    assert text_config["eos_token_id"] == 2
    assert text_config["video_feature_extractor_out_features"] == 8
    assert text_config["audio_feature_extractor_out_features"] == 3
    assert text_config["caption_proj_before_connector"] is True
    assert text_config["connector_rope_type"] == "split"
    assert text_config["connector_ff_bias"] is False

    # Validate the emitted config through the production FastVideo model
    # constructor. The Gemma backbone remains lazy, so this is lightweight.
    encoder_config = LTX2GemmaConfig()
    encoder_config.update_model_arch(text_config)
    encoder = LTX2GemmaTextEncoderModel(encoder_config)
    assert encoder.video_feature_extractor_linear.in_features == 18
    assert encoder.video_feature_extractor_linear.out_features == 8
    assert encoder.audio_feature_extractor_linear.out_features == 3
    assert encoder.embeddings_connector.transformer_1d_blocks[0].ff.net[0].proj.bias is None

    gemma_weights = load_file(str(output / "text_encoder" / "gemma" / "model.safetensors"))
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in gemma_weights
    assert "model.vision_embedder.embeddings.patch_embedding.weight" in gemma_weights
    assert "model.embed_vision.embedding_projection.weight" in gemma_weights
    assert "model.embed_audio.proj.weight" in gemma_weights
    assert (output / "text_encoder" / "gemma" / "tokenizer.json").read_text() == "{}"
    assert (output / "tokenizer" / "tokenizer_config.json").is_file()

    assert set(load_file(str(output / "vae" / "model.safetensors"))) == {"encoder.conv.weight"}
    assert set(load_file(str(output / "audio_vae" / "model.safetensors"))) == {
        "decoder.conv_in.weight",
        "per_channel_statistics.mean-of-means",
    }
    assert set(load_file(str(output / "vocoder" / "model.safetensors"))) == {
        "vocoder.conv_pre.weight",
        "bwe_generator.conv_pre.weight",
    }
    assert set(load_file(str(output / "spatial_upsampler" / "model.safetensors"))) == {
        "initial_conv.weight",
        "upsampler.0.weight",
    }
    assert set(load_file(str(output / "distilled_lora" / "model.safetensors"))) == {
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"
    }

    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index["transformer"] == ["diffusers", "LTX2Transformer3DModel"]
    assert model_index["spatial_upsampler"] == ["diffusers", "LTX2LatentUpsampler"]
    assert model_index["fastvideo_ltx2_variant"] == "ltx2.5-dev"
    assert model_index["fastvideo_refine_enabled"] is True
    assert model_index["fastvideo_refine_upsampler_path"] == "spatial_upsampler"
    assert model_index["fastvideo_refine_lora_path"] == "distilled_lora/model.safetensors"


def test_split_model_index_enables_distilled_refine_without_lora() -> None:
    model_index = converter._build_split_model_index(
        transformer_class_name="LTX2Transformer3DModel",
        pipeline_class_name="LTX2Pipeline",
        diffusers_version="0.33.0.dev0",
        variant="distilled",
        distilled_lora=False,
    )
    assert model_index["fastvideo_refine_enabled"] is True
    assert "fastvideo_refine_lora_path" not in model_index


def test_split_ltx2_5_conversion_preserves_enabled_transformer_gates(tmp_path: Path) -> None:
    """Transformer config filtering retains True-valued 2.5 architecture gates."""
    # Create a fresh fixture with all 2.5 gates enabled
    transformer_config = {
        "transformer": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "cross_attention_dim": 8,
            "caption_channels": 6,
            "audio_num_attention_heads": 1,
            "audio_attention_head_dim": 3,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_cross_attention_dim": 3,
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "apply_gated_attention": True,
            "use_prompt_adaln_single": True,
            "ff_bias": True,
            "audio_ff_bias": True,
            "use_keyframes_abs_pos_embedding": True,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": True,
            "caption_projection_second_linear": False,
            "connector_num_attention_heads": 2,
            "connector_attention_head_dim": 4,
            "connector_num_layers": 1,
            "audio_connector_num_attention_heads": 1,
            "audio_connector_attention_head_dim": 3,
            "audio_connector_num_layers": 1,
            "connector_positional_embedding_max_pos": [2048],
            "connector_apply_gated_attention": True,
            "connector_ff_bias": False,
            "frequencies_precision": "float64",
        }
    }
    transformer = tmp_path / "transformer_gates_enabled.safetensors"
    _save(
        transformer,
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": torch.ones(2, 2),
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.weight": torch.ones(1),
            "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.weight": torch.ones(1),
        },
        config=transformer_config,
    )

    # Reuse other fixture sources
    sources = _split_sources(tmp_path)
    output = tmp_path / "converted_gates_enabled"

    converter.convert_split_components(
        transformer_source=transformer,
        text_encoder_source=sources["text_encoder"],
        vae_source=sources["vae"],
        audio_vae_source=sources["audio_vae"],
        spatial_upscaler_source=sources["spatial_upscaler"],
        distilled_lora_source=sources["distilled_lora"],
        output_dir=output,
        transformer_class_name="LTX2Transformer3DModel",
        variant="dev",
    )

    transformer_config_out = json.loads((output / "transformer" / "config.json").read_text())
    assert transformer_config_out["use_prompt_adaln_single"] is True
    assert transformer_config_out["ff_bias"] is True
    assert transformer_config_out["audio_ff_bias"] is True
    assert transformer_config_out["use_keyframes_abs_pos_embedding"] is True


def test_legacy_monolithic_conversion_behavior_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "legacy.safetensors"
    metadata_config = {
        "transformer": {
            "num_attention_heads": 2,
            "frequencies_precision": "float64",
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "apply_gated_attention": True,
            "use_prompt_adaln_single": False,
            "ff_bias": False,
            "audio_ff_bias": False,
            "use_keyframes_abs_pos_embedding": False,
        },
        "vae": {},
        "audio_vae": {},
        "vocoder": {},
    }
    _save(
        source,
        {
            "model.diffusion_model.transformer_blocks.0.weight": torch.ones(1),
            "vae.encoder.weight": torch.ones(1),
            "audio_vae.decoder.weight": torch.ones(1),
            "vocoder.conv_pre.weight": torch.ones(1),
            "text_embedding_projection.video_aggregate_embed.weight": torch.ones(1),
        },
        config=metadata_config,
    )
    output = tmp_path / "legacy-converted"
    converter.convert_components(
        source,
        output,
        metadata_config,
        "LTX2Transformer3DModel",
    )

    assert set(load_file(str(output / "transformer" / "model.safetensors"))) == {
        "transformer_blocks.0.weight"
    }
    # Legacy monolith keeps its historical projection key surface.
    assert set(load_file(str(output / "text_encoder" / "model.safetensors"))) == {
        "video_aggregate_embed.weight"
    }
    config = json.loads((output / "transformer" / "config.json").read_text())
    assert config["cross_attention_adaln"] is True
    assert config["use_prompt_adaln_single"] is False
    assert config["ff_bias"] is False
    assert config["audio_ff_bias"] is False
    assert config["use_keyframes_abs_pos_embedding"] is False
    assert config["double_precision_rope"] is True
    assert (output / "model_index.json").is_file()
