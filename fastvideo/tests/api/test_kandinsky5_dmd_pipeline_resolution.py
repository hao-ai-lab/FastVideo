# SPDX-License-Identifier: Apache-2.0
"""Regression tests for self-identifying Kandinsky5 DMD exports."""

import json
import os

from fastvideo.configs.pipelines.kandinsky5 import (
    Kandinsky5DMDConfig,
    Kandinsky5T2VConfig,
)
from fastvideo.pipelines.basic.kandinsky5.kandinsky5_dmd_pipeline import Kandinsky5DMDPipeline
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.registry import get_model_info
from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _rewrite_model_index_pipeline_class,
)


def _write_fake_export(tmp_path, name: str, pipeline_cls_name: str):
    """Minimal on-disk stand-in for a dcp_to_diffusers export: just enough
    for verify_model_config_and_directory to accept it."""
    model_dir = tmp_path / name
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": pipeline_cls_name,
            "_diffusers_version": "0.30.0",
        }))
    return model_dir


def test_kandinsky5_dmd_config_sets_denoising_steps():
    assert Kandinsky5T2VConfig().dmd_denoising_steps is None
    assert Kandinsky5DMDConfig().dmd_denoising_steps == [1000, 750, 500, 250]


def test_exported_dmd_checkpoint_resolves_without_override(tmp_path):
    model_dir = _write_fake_export(
        tmp_path,
        "kandinsky5_t2v_dmd2_4steps_qat",
        "Kandinsky5DMDPipeline",
    )
    info = get_model_info(model_path=str(model_dir), pipeline_type=PipelineType.BASIC)
    assert info.pipeline_cls is Kandinsky5DMDPipeline
    assert info.pipeline_config_cls is Kandinsky5DMDConfig


def test_pipeline_class_rewrite_does_not_mutate_hardlinked_base(tmp_path):
    base_dir = _write_fake_export(
        tmp_path,
        "base",
        "Kandinsky5T2VPipeline",
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    os.link(
        base_dir / "model_index.json",
        export_dir / "model_index.json",
    )

    _rewrite_model_index_pipeline_class(
        str(export_dir),
        "Kandinsky5DMDPipeline",
    )

    base_index = json.loads((base_dir / "model_index.json").read_text())
    export_index = json.loads((export_dir / "model_index.json").read_text())
    assert base_index["_class_name"] == "Kandinsky5T2VPipeline"
    assert export_index["_class_name"] == "Kandinsky5DMDPipeline"
