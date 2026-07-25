# SPDX-License-Identifier: Apache-2.0
"""Lightweight conversion coverage for official MMAudio variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def _converter_module():
    path = Path(__file__).resolve().parents[3] / (
        "scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py"
    )
    spec = importlib.util.spec_from_file_location("mmaudio_variant_converter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_16k_transformer_conversion_writes_variant_contract(
    tmp_path: Path,
) -> None:
    converter = _converter_module()
    checkpoint = tmp_path / "mmaudio_small_16k.pth"
    output = tmp_path / "converted"
    torch.save({"empty_string_feat": torch.zeros(77, 1024)}, checkpoint)

    converter.convert(
        argparse.Namespace(
            variant="small_16k",
            transformer_checkpoint=checkpoint,
            audio_vae_checkpoint=None,
            synchformer_checkpoint=None,
            dfn5b_dir=None,
            bigvgan_dir=None,
            bigvgan_checkpoint=None,
            transformer_only=True,
            transformer_config_only=False,
            preprocessor_only=False,
            output=output,
        )
    )

    transformer_config = json.loads(
        (output / "transformer/config.json").read_text(encoding="utf-8")
    )
    model_index = json.loads(
        (output / "model_index.json").read_text(encoding="utf-8")
    )
    assert transformer_config["latent_dim"] == 20
    assert transformer_config["latent_seq_len"] == 250
    assert transformer_config["hidden_dim"] == 448
    assert transformer_config["v2"] is False
    assert model_index["_fastvideo_mmaudio_variant"] == "small_16k"
    assert model_index["_fastvideo_audio_sample_rate"] == 16000


def test_converter_lists_all_five_official_variants() -> None:
    converter = _converter_module()
    assert tuple(converter.TRANSFORMER_VARIANTS) == (
        "small_16k",
        "small_44k",
        "medium_44k",
        "large_44k",
        "large_44k_v2",
    )
    assert converter.BIGVGAN_16K_CONFIG["num_mels"] == 80
    assert converter.BIGVGAN_16K_CONFIG["upsample_rates"] == [4, 4, 2, 2, 2, 2]


def test_weight_free_transformer_export_skeleton(tmp_path: Path) -> None:
    converter = _converter_module()
    output = tmp_path / "skeleton"

    converter.convert(
        argparse.Namespace(
            variant="medium_44k",
            transformer_checkpoint=None,
            audio_vae_checkpoint=None,
            synchformer_checkpoint=None,
            dfn5b_dir=None,
            bigvgan_dir=None,
            bigvgan_checkpoint=None,
            transformer_only=False,
            transformer_config_only=True,
            preprocessor_only=False,
            output=output,
        )
    )

    config = json.loads(
        (output / "transformer/config.json").read_text(encoding="utf-8")
    )
    assert config["hidden_dim"] == 896
    assert config["depth"] == 12
    assert not list((output / "transformer").glob("*.safetensors"))
