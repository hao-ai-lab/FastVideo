# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LTX-2 refine upsampler path resolution.

Pure tmp_path tests: no GPU, no model downloads.
"""

from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import _resolve_refine_upsampler_path


def test_model_index_key_wins_over_directory_probe(tmp_path):
    (tmp_path / "custom_upsampler").mkdir()
    (tmp_path / "spatial_upscaler").mkdir()
    model_index = {"fastvideo_refine_upsampler_path": "custom_upsampler"}
    assert _resolve_refine_upsampler_path(str(tmp_path), model_index) == str(tmp_path / "custom_upsampler")


def test_model_index_spatial_upsampler_module_entry(tmp_path):
    (tmp_path / "spatial_upsampler").mkdir()
    model_index = {"spatial_upsampler": ["diffusers", "LTX2SpatialUpsampler"]}
    assert _resolve_refine_upsampler_path(str(tmp_path), model_index) == str(tmp_path / "spatial_upsampler")


def test_directory_fallback_finds_spatial_upscaler(tmp_path):
    # The published LTX-2 distilled checkpoints ship the weights on disk as
    # "spatial_upscaler" without any model_index.json declaration.
    (tmp_path / "spatial_upscaler").mkdir()
    assert _resolve_refine_upsampler_path(str(tmp_path), {}) == str(tmp_path / "spatial_upscaler")


def test_directory_fallback_finds_spatial_upsampler(tmp_path):
    (tmp_path / "spatial_upsampler").mkdir()
    assert _resolve_refine_upsampler_path(str(tmp_path), {}) == str(tmp_path / "spatial_upsampler")


def test_returns_none_when_nothing_resolves(tmp_path):
    assert _resolve_refine_upsampler_path(str(tmp_path), {}) is None
