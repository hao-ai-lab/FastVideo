# SPDX-License-Identifier: Apache-2.0
"""Regression test for exported Kandinsky5 DMD checkpoints resolving to the
wrong pipeline.

``fastvideo.train.entrypoint.dcp_to_diffusers`` copies the base T2V
checkpoint's ``model_index.json`` unchanged into every export (see
``examples/train/configs/fine_tuning/kandinsky5/README.md``), so
``_class_name`` still says the base T2V pipeline. Without an explicit
override, ``fastvideo.registry.get_model_info`` resolves such a directory to
``Kandinsky5T2VPipeline`` (via the path-based "t2v" fallback detector in
``fastvideo/registry.py``) instead of ``Kandinsky5DMDPipeline`` -- silently
running the full-length multi-step sampler on DMD-distilled weights instead
of the four-step re-noise sampler. ``override_pipeline_cls_name`` bypasses
that resolution entirely and must be paired with ``Kandinsky5DMDConfig`` (not
the default ``Kandinsky5T2VConfig``, whose ``dmd_denoising_steps=None`` makes
``Kandinsky5DmdDenoisingStage`` raise immediately).
"""

import json

from fastvideo.configs.pipelines.kandinsky5 import (
    Kandinsky5DMDConfig,
    Kandinsky5T2VConfig,
)
from fastvideo.pipelines.basic.kandinsky5.kandinsky5_dmd_pipeline import Kandinsky5DMDPipeline
from fastvideo.pipelines.basic.kandinsky5.kandinsky5_pipeline import Kandinsky5T2VPipeline
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.registry import get_model_info


def _write_fake_export(tmp_path, name: str):
    """Minimal on-disk stand-in for a dcp_to_diffusers export: just enough
    for verify_model_config_and_directory to accept it."""
    model_dir = tmp_path / name
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": "Kandinsky5T2VPipeline",
            "_diffusers_version": "0.30.0",
        }))
    return model_dir


def test_kandinsky5_dmd_config_sets_denoising_steps():
    assert Kandinsky5T2VConfig().dmd_denoising_steps is None
    assert Kandinsky5DMDConfig().dmd_denoising_steps == [1000, 750, 500, 250]


def test_exported_dmd_checkpoint_resolves_to_t2v_pipeline_without_override(tmp_path):
    """Documents the bug: an unmodified export resolves to the T2V pipeline."""
    model_dir = _write_fake_export(tmp_path, "kandinsky5_t2v_dmd2_4steps_qat")
    info = get_model_info(model_path=str(model_dir), pipeline_type=PipelineType.BASIC)
    assert info.pipeline_cls is Kandinsky5T2VPipeline


def test_override_pipeline_cls_name_resolves_dmd_pipeline(tmp_path):
    model_dir = _write_fake_export(tmp_path, "kandinsky5_t2v_dmd2_4steps_qat_override")
    info = get_model_info(
        model_path=str(model_dir),
        pipeline_type=PipelineType.BASIC,
        override_pipeline_cls_name="Kandinsky5DMDPipeline",
    )
    assert info.pipeline_cls is Kandinsky5DMDPipeline
