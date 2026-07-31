# SPDX-License-Identifier: Apache-2.0
"""Checks for single-file LTX metadata parsing and prefix routing.

Builds a tiny synthetic bundle shaped like the real one -- same metadata keys,
same top-level tensor prefixes -- so no checkpoint is needed. Run directly
(``python fastvideo/tests/api/test_ltx_single_file.py``) or under pytest.
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from fastvideo.models.loader.ltx_single_file import (
    LTXCheckpointMetadata,
    resolve_text_encoder_root,
    build_dit_config,
    bundle_model_index,
    component_weights,
    read_ltx_metadata,
)

# Subset of the transformer section as the checkpoint declares it.
TRANSFORMER_SECTION = {
    "_class_name": "AVTransformer3DModel",
    "num_layers": 48,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "cross_attention_dim": 4096,
    "audio_cross_attention_dim": 2048,
    "caption_channels": 3840,
    "ff_bias": False,
    "cross_attention_adaln": True,
    "caption_proj_before_connector": True,
    "apply_gated_attention": True,
    "rope_type": "split",
    "frequencies_precision": "float64",
}


def _write_bundle(path: Path,
                  *,
                  with_gemma_source: bool,
                  transformer_cls: str = "AVTransformer3DModel") -> None:
    tensors = {
        "model.diffusion_model.patchify_proj.weight": torch.zeros(2, 2),
        "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": torch.zeros(2, 2),
        # Stored under the transformer prefix but owned by the text stack.
        "model.diffusion_model.video_embeddings_connector.x.weight": torch.zeros(2),
        "model.diffusion_model.audio_embeddings_connector.x.weight": torch.zeros(2),
        "vae.encoder.conv.weight": torch.zeros(2),
        "audio_vae.decoder.conv.weight": torch.zeros(2),
        "vocoder.conv.weight": torch.zeros(2),
        "text_embedding_projection.video_aggregate_embed.weight": torch.zeros(2),
        "duration_head.linear.weight": torch.zeros(2),
    }
    section = dict(TRANSFORMER_SECTION, _class_name=transformer_cls)
    config = {
        "transformer": section,
        # Declares its class: a component that can be built from the bundle.
        "vae": {"_class_name": "CausalVideoAutoencoder", "dims": 3},
        # Present and carrying weights, but naming no class.
        "audio_vae": {},
    }
    metadata = {
        "config": json.dumps(config),
        "model_version": "9.9.9",
        "license": "x" * 128,
    }
    if with_gemma_source:
        metadata["gemma_source_checkpoint"] = json.dumps(
            {"ltx_version": "9.9.9", "gemma_version": "fake-encoder-v0"})
    save_file(tensors, str(path), metadata=metadata)


def test_read_metadata_and_routing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.safetensors"
        _write_bundle(path, with_gemma_source=True)

        meta = read_ltx_metadata(str(path))
        assert meta.model_version == "9.9.9"
        assert meta.gemma_source_checkpoint == {
            "ltx_version": "9.9.9",
            "gemma_version": "fake-encoder-v0",
        }
        assert set(meta.config) == {"transformer", "vae", "audio_vae"}
        # frequencies_precision is the checkpoint's name for double_precision_rope.
        assert meta.config["transformer"]["double_precision_rope"] is True

        # Prefix stripped, connectors excluded, nothing from other components.
        transformer_keys = {n for n, _ in component_weights(str(path), "transformer")}
        assert transformer_keys == {
            "patchify_proj.weight",
            "transformer_blocks.0.ff.net.2.weight",
        }
        for component, expected in (
            ("vae", {"encoder.conv.weight"}),
            ("audio_vae", {"decoder.conv.weight"}),
            ("vocoder", {"conv.weight"}),
            # The connectors come off the transformer prefix with their
            # sub-tree name intact; the text encoder renames from there.
            ("text_encoder", {
                "video_aggregate_embed.weight",
                "video_embeddings_connector.x.weight",
                "audio_embeddings_connector.x.weight",
            }),
        ):
            assert {n for n, _ in component_weights(str(path), component)} == expected


def test_bundle_model_index_declares_what_the_metadata_declares() -> None:
    """The model_index a bundle stands in for: same shape, same libraries."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.safetensors"
        _write_bundle(path, with_gemma_source=True)
        index = bundle_model_index(str(path))

        # A section that names a class is a component we can build.
        assert index["transformer"] == ["diffusers", "AVTransformer3DModel"]
        assert index["vae"] == ["diffusers", "CausalVideoAutoencoder"]
        # A section that names none stays declared with a null library, so
        # `load_modules` drops it from the required set on its own. Omitting
        # the key instead would trip its required-module check.
        assert index["audio_vae"] == [None, None]
        # Both live outside the bundle, but the pipeline requires them.
        assert index["text_encoder"] == [
            "transformers", "LTX2GemmaTextEncoderModel"
        ]
        assert index["tokenizer"] == ["transformers", "AutoTokenizer"]
        # `load_modules` pops both of these without a default.
        assert "_class_name" in index and "_diffusers_version" in index


def test_missing_gemma_source_is_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.safetensors"
        _write_bundle(path, with_gemma_source=False)
        assert read_ltx_metadata(str(path)).gemma_source_checkpoint is None


def test_dit_config_takes_ff_bias_from_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.safetensors"
        _write_bundle(path, with_gemma_source=True)
        arch = build_dit_config(read_ltx_metadata(str(path))).arch_config

        # ff_bias comes from metadata; audio_ff_bias has no metadata key and
        # must keep the default, matching the audio_ff.*.bias tensors that the
        # checkpoint does carry.
        assert arch.ff_bias is False
        assert arch.audio_ff_bias is True
        assert arch.num_layers == 48
        assert arch.cross_attention_adaln is True
        assert arch.caption_channels == 3840


def test_defaults_unchanged_without_metadata() -> None:
    from fastvideo.configs.models.dits.ltx2 import LTX2VideoArchConfig

    arch = LTX2VideoArchConfig()
    assert arch.ff_bias is True
    assert arch.audio_ff_bias is True


def test_encoder_root_requires_an_explicit_declaration() -> None:
    """Neither source set -> raise, naming both. Never guess from the filesystem."""
    import pytest
    with pytest.raises(ValueError, match="must be declared"):
        resolve_text_encoder_root(configured_path=None, override_path=None)


def test_encoder_root_override_beats_config() -> None:
    assert resolve_text_encoder_root("/from/config", "/from/override") == "/from/override"
    assert resolve_text_encoder_root("/from/config", None) == "/from/config"


def test_encoder_root_is_not_discovered_from_a_sibling_directory() -> None:
    """A plausible encoder sitting next to the bundle must NOT be picked up."""
    with tempfile.TemporaryDirectory() as tmp:
        sibling = Path(tmp) / "gemma"
        sibling.mkdir()
        (sibling / "config.json").write_text("{}")
        import pytest
        with pytest.raises(ValueError):
            resolve_text_encoder_root(configured_path=None, override_path=None)


def test_declared_root_survives_a_version_mismatch() -> None:
    """Pairing metadata validates a declared root; it never overrides it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "encoder"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"gemma_version": "other-v0"}))
        meta = LTXCheckpointMetadata(
            config={}, model_version="9.9.9",
            gemma_source_checkpoint={"gemma_version": "fake-encoder-v0"})
        assert resolve_text_encoder_root(str(root), None, meta) == str(root)


def test_bundle_resolves_its_pipeline_config_from_declared_transformer_class() -> None:
    """A bundle path resolves through the alias table, not through model_index.json."""
    from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig
    from fastvideo.registry import get_pipeline_config_cls_from_name

    with tempfile.TemporaryDirectory() as tmp:
        # Deliberately uninformative name: resolution must not read the file name.
        path = Path(tmp) / "anonymous.safetensors"
        _write_bundle(path, with_gemma_source=True)
        assert get_pipeline_config_cls_from_name(str(path)) is LTX2T2VConfig


def test_unknown_bundle_transformer_class_raises_naming_table_and_override() -> None:
    """No wildcard fallback: an unmapped class fails loud and says how to fix it."""
    import pytest

    from fastvideo.registry import get_pipeline_config_cls_from_name

    with tempfile.TemporaryDirectory() as tmp:
        # Name it after a model the detectors DO know, to prove the file name
        # is not consulted -- only the class the checkpoint declares.
        path = Path(tmp) / "ltx2-distilled.safetensors"
        _write_bundle(path, with_gemma_source=True, transformer_cls="NotAModel")
        with pytest.raises(ValueError) as excinfo:
            get_pipeline_config_cls_from_name(str(path))
        message = str(excinfo.value)
        assert "_BUNDLE_TRANSFORMER_TO_CONFIG" in message
        assert "override_pipeline_cls_name" in message


if __name__ == "__main__":
    test_read_metadata_and_routing()
    test_bundle_model_index_declares_what_the_metadata_declares()
    test_missing_gemma_source_is_none()
    test_dit_config_takes_ff_bias_from_metadata()
    test_defaults_unchanged_without_metadata()
    test_encoder_root_override_beats_config()
    test_bundle_resolves_its_pipeline_config_from_declared_transformer_class()
    test_unknown_bundle_transformer_class_raises_naming_table_and_override()
    print("ok")


def test_connector_factorization_must_match_its_stream_width() -> None:
    """A connector's heads x head_dim must equal the width of the stream it feeds.

    This is the mistake that loads cleanly and computes wrong, so it fails loud
    and names both numbers. A field the checkpoint does not declare is not
    checked -- the arch default applies and there is nothing to contradict.
    """
    import pytest

    from fastvideo.models.loader.component_loader import _check_connector_widths

    declared = {
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "cross_attention_dim": 4096,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
        "audio_cross_attention_dim": 2048,
    }
    _check_connector_widths(declared)
    _check_connector_widths({})
    _check_connector_widths({"connector_num_attention_heads": 32})

    # The audio connector inheriting the video head_dim is the exact failure
    # this exists to stop: 32 x 128 = 4096, but the audio stream is 2048.
    with pytest.raises(ValueError, match="audio"):
        _check_connector_widths(dict(declared, audio_connector_attention_head_dim=128))
    with pytest.raises(ValueError, match="video"):
        _check_connector_widths(dict(declared, connector_num_attention_heads=30))
