# SPDX-License-Identifier: Apache-2.0
"""Resolve the per-task preprocessing config that ``fastvideo/pipelines/preprocess/v1_preprocess.py`` runs on.

Cosmos-Predict2.5 declares a bf16 VAE, so it shows whether a video task switches the VAE to fp32.
"""
from pathlib import Path

from fastvideo.api.training_schema import PreprocessRunConfig, load_resolved_run_config

COSMOS_MODEL = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"


def _resolve(tmp_path: Path, preprocess_task: str):
    config_path = tmp_path / "preprocess.yaml"
    config_path.write_text(f"preprocess:\n  preprocess_task: {preprocess_task}\n  max_height: 704\n")
    return load_resolved_run_config(PreprocessRunConfig, [
        "--config",
        str(config_path),
        "--model_path",
        COSMOS_MODEL,
        "--preprocess.data_merge_path",
        "data/merge.txt",
        "--preprocess.dataset_output_dir=data/out",
    ])


def test_video_task_encodes_with_fp32_vae(tmp_path: Path) -> None:
    resolved_config = _resolve(tmp_path, "t2v")

    assert resolved_config.engine.precision.vae == "fp32"
    assert resolved_config.provenance("engine.precision.vae").source == "derive_video_preprocess_vae_precision"
    assert resolved_config.pipeline_config.vae_precision == "fp32"
    assert resolved_config.pipeline_config.vae_config.load_encoder
    assert resolved_config.preprocess.max_height == 704
    assert resolved_config.preprocess.data_merge_path == "data/merge.txt"
    assert resolved_config.preprocess.dataset_output_dir == "data/out"
    assert resolved_config.preprocess.model_path == COSMOS_MODEL


def test_text_only_task_keeps_the_model_vae_precision(tmp_path: Path) -> None:
    resolved_config = _resolve(tmp_path, "text_only")

    assert resolved_config.engine.precision.vae == "bf16"
