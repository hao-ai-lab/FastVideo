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
    resolve_text_encoder_root,
    build_dit_config,
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


def _write_bundle(path: Path, *, with_gemma_source: bool) -> None:
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
    metadata = {
        "config": json.dumps({"transformer": TRANSFORMER_SECTION, "vae": {}}),
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
        assert set(meta.config) == {"transformer", "vae"}
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


if __name__ == "__main__":
    test_read_metadata_and_routing()
    test_missing_gemma_source_is_none()
    test_dit_config_takes_ff_bias_from_metadata()
    test_defaults_unchanged_without_metadata()
    print("ok")


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
