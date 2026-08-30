"""Coverage for merging the non-factorized half of an adapter into base weights."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "lora_extraction"))

import merge_lora  # noqa: E402


def _base_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "blocks.0.linear.weight": torch.randn(9, 7, generator=generator),
        "audio_proj_in.weight": torch.randn(8, 6, generator=generator),
        "time_embedder.linear.bias": torch.randn(8, generator=generator),
        "scale_param": torch.randn(4, generator=generator),
    }


def _adapter(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    return {
        "blocks.0.linear.lora_A.weight": torch.randn(2, 7, generator=generator),
        "blocks.0.linear.lora_B.weight": torch.randn(9, 2, generator=generator),
        "audio_proj_in.diff": torch.randn(8, 6, generator=generator),
        "time_embedder.linear.diff_b": torch.randn(8, generator=generator),
        "scale_param.diff_param": torch.randn(4, generator=generator),
        "new_proj.set_weight": torch.randn(3, 3, generator=generator),
    }


def test_group_dense_keys_splits_additive_and_replacement() -> None:
    additive, replacement, unrecognized = merge_lora.group_dense_keys(_adapter(_base_state()))

    assert set(additive) == {"audio_proj_in.weight", "time_embedder.linear.bias", "scale_param"}
    assert set(replacement) == {"new_proj.weight"}
    assert unrecognized == []


def test_group_dense_keys_reports_unrecognized_suffixes() -> None:
    _, _, unrecognized = merge_lora.group_dense_keys({"mystery.tensor": torch.zeros(2)})
    assert unrecognized == ["mystery.tensor"]


def test_merge_applies_dense_tensors_alongside_lora() -> None:
    """--exact-tensor-pattern keeps tensors as dense deltas; the merge must still apply them."""
    base = _base_state()
    adapter = _adapter(base)

    merged = merge_lora.merge_lora_into_base(base, adapter)

    expected_lora = base["blocks.0.linear.weight"] + adapter["blocks.0.linear.lora_B.weight"] @ adapter[
        "blocks.0.linear.lora_A.weight"]
    assert torch.allclose(merged["blocks.0.linear.weight"], expected_lora, atol=1e-6)
    assert torch.allclose(merged["audio_proj_in.weight"],
                          base["audio_proj_in.weight"] + adapter["audio_proj_in.diff"],
                          atol=1e-6)
    assert torch.allclose(merged["time_embedder.linear.bias"],
                          base["time_embedder.linear.bias"] + adapter["time_embedder.linear.diff_b"],
                          atol=1e-6)
    assert torch.allclose(merged["scale_param"], base["scale_param"] + adapter["scale_param.diff_param"], atol=1e-6)
    assert torch.equal(merged["new_proj.weight"], adapter["new_proj.set_weight"])
    # the caller's state dict must not be mutated
    assert torch.equal(base["audio_proj_in.weight"], _base_state()["audio_proj_in.weight"])


def test_merge_warns_instead_of_silently_dropping_unknown_keys(caplog) -> None:
    base = _base_state()
    adapter = {"mystery.tensor": torch.zeros(2), "shape_mismatch.diff": torch.zeros(1, 1)}

    with caplog.at_level(logging.WARNING, logger=merge_lora.LOG.name):
        merge_lora.merge_lora_into_base(base, adapter)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "unrecognized suffix" in messages
    assert "mystery.tensor" in messages
    assert "shape_mismatch.weight" in messages
