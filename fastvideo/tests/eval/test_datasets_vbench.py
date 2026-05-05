"""Smoke tests for the VBench prompt dataset and dataset registry."""
from __future__ import annotations

import pytest

from fastvideo.eval.datasets import (PromptDataset, VBenchPromptDataset,
                                     get_dataset, list_datasets)


def test_registry_contains_vbench():
    assert "vbench" in list_datasets()


def test_get_dataset_returns_typed_instance():
    ds = get_dataset("vbench", dimensions=["color"])
    assert isinstance(ds, PromptDataset)
    assert isinstance(ds, VBenchPromptDataset)
    assert ds.name == "vbench"
    assert ds.supports_dimensions is True


def test_full_corpus_size():
    ds = VBenchPromptDataset()
    assert len(ds) == 946
    assert len(ds.dimensions) == 16


def test_rows_are_dicts_with_required_keys():
    ds = VBenchPromptDataset(dimensions=["color"])
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "prompt" in sample
    assert "n_samples" in sample
    assert "dimensions" in sample


def test_temporal_flickering_n_samples():
    ds = VBenchPromptDataset(dimensions=["temporal_flickering"])
    assert all(s["n_samples"] == 25 for s in ds)


def test_default_n_samples():
    ds = VBenchPromptDataset(dimensions=["subject_consistency"])
    assert all(s["n_samples"] == 5 for s in ds)


def test_color_aux_info_is_flat():
    ds = VBenchPromptDataset(dimensions=["color"])
    sample = ds[0]
    aux = sample["auxiliary_info"]
    # Flat: {"color": "<color name>"}, not nested under the dimension.
    assert "color" in aux
    assert isinstance(aux["color"], str)


def test_unknown_dimension_raises():
    with pytest.raises(ValueError, match="Unknown VBench dimensions"):
        VBenchPromptDataset(dimensions=["bogus"])


def test_by_dimension_groups_correctly():
    ds = VBenchPromptDataset(dimensions=["subject_consistency", "color"])
    groups = ds.by_dimension()
    assert set(groups) == {"subject_consistency", "color"}
    assert all(isinstance(s["auxiliary_info"].get("color"), str)
               for s in groups["color"])
