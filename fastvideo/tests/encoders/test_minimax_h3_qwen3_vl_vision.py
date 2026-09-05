# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for MiniMax-H3's Qwen3-VL vision position embeddings."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from fastvideo.models.encoders import minimax_h3_qwen3_vl
from fastvideo.models.encoders.minimax_h3_qwen3_vl import (
    _VISION_INTERPOLATION_WORKSPACE_BYTES,
    _interpolate_vision_position_embeddings,
)


_GRID_CASES = (
    torch.tensor([[1, 4, 6]], dtype=torch.long),
    torch.tensor([[2, 4, 6]], dtype=torch.long),
    torch.tensor([[1, 4, 6], [2, 8, 10]], dtype=torch.long),
    torch.tensor([[1, 48, 84]], dtype=torch.long),
    torch.tensor([[1, 128, 224]], dtype=torch.long),
    torch.tensor([[15, 42, 74]], dtype=torch.long),
    torch.tensor([[1, 128, 224], [15, 42, 74]], dtype=torch.long),
)


def _torch_interpolation_reference(
    position_embedding: torch.Tensor,
    grid_thw: torch.Tensor,
    side: int,
    merge: int,
) -> torch.Tensor:
    """Use PyTorch's bilinear kernel as an independent, always-available oracle."""
    channels = position_embedding.shape[1]
    source = position_embedding.float().T.reshape(1, channels, side, side)
    outputs = []
    for frames, height, width in grid_thw.tolist():
        raster = F.interpolate(source, size=(height, width), mode="bilinear", align_corners=True)
        raster = raster.squeeze(0).permute(1, 2, 0)
        merged = raster.view(height // merge, merge, width // merge, merge, channels)
        merged = merged.permute(0, 2, 1, 3, 4).flatten(0, 3)
        outputs.append(merged.repeat(frames, 1))
    return torch.cat(outputs)


def _transformers_5_15_contract_reference(
    position_embedding: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the Transformers 5.15 contract without requiring its newer public helper."""
    side = 48
    merge = 2
    channels = position_embedding.shape[1]
    outputs = []
    for frames, height, width in grid_thw.tolist():
        row_positions = (torch.arange(height, device=position_embedding.device, dtype=torch.float32) *
                         (side - 1) / max(height - 1, 1))
        column_positions = (torch.arange(width, device=position_embedding.device, dtype=torch.float32) *
                            (side - 1) / max(width - 1, 1))
        row_floors = row_positions.floor()
        column_floors = column_positions.floor()
        offsets = torch.arange(2, device=position_embedding.device)
        row_taps = (row_floors.long()[:, None] + offsets).clamp(0, side - 1)
        column_taps = (column_floors.long()[:, None] + offsets).clamp(0, side - 1)
        row_weights = (1 - (row_positions[:, None] - row_floors[:, None] - offsets).abs()).clamp(min=0)
        column_distances = (column_positions[:, None] - column_floors[:, None] - offsets).abs()
        column_weights = (1 - column_distances).clamp(min=0)

        indices = (row_taps[:, None, :, None] * side + column_taps[None, :, None, :]).reshape(height, width, 4)
        weights = (row_weights[:, None, :, None] * column_weights[None, :, None, :]).reshape(height, width, 4)
        raster = (position_embedding[indices] * weights[:, :, :, None]).sum(2)
        merged = raster.view(height // merge, merge, width // merge, merge, channels)
        merged = merged.permute(0, 2, 1, 3, 4).flatten(0, 3)
        outputs.append(merged.repeat(frames, 1))
    return torch.cat(outputs)


def _installed_transformers_interpolation_reference(
    position_embedding: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    from transformers import vision_utils

    get_reference = getattr(vision_utils, "get_vision_interpolation_indices_and_weights", None)
    if get_reference is None:
        pytest.skip("the public interpolation helper requires Transformers 5.15 or newer")
    indices, weights = get_reference(grid_thw,
                                     num_grid_per_side=48,
                                     mode="bilinear",
                                     align_corners=True,
                                     spatial_merge_size=2)
    return (position_embedding[indices] * weights[:, :, None]).sum(1)


@pytest.mark.parametrize("grid_thw", _GRID_CASES)
def test_position_interpolation_matches_transformers_5_15_contract(grid_thw: torch.Tensor) -> None:
    """Pin visual-token ordering and float32 accumulation for every supported dependency version."""
    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 32, dtype=torch.bfloat16)
    expected = _transformers_5_15_contract_reference(position_embedding, grid_thw)

    actual = _interpolate_vision_position_embeddings(position_embedding, grid_thw, 48, 2)

    assert actual.dtype == torch.float32
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("grid_thw", _GRID_CASES)
def test_transformers_5_15_contract_matches_installed_helper(grid_thw: torch.Tensor) -> None:
    """Cross-check the self-contained contract when the newer public helper is installed."""
    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 32, dtype=torch.bfloat16)

    expected = _installed_transformers_interpolation_reference(position_embedding, grid_thw)
    actual = _transformers_5_15_contract_reference(position_embedding, grid_thw)

    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("grid_thw", _GRID_CASES)
def test_position_interpolation_uses_float32_bilinear_accumulation(grid_thw: torch.Tensor) -> None:
    """Prevent bf16 interpolation weights from reintroducing visual-token drift."""
    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 32, dtype=torch.bfloat16)

    expected = _torch_interpolation_reference(position_embedding, grid_thw, 48, 2)
    actual = _interpolate_vision_position_embeddings(position_embedding, grid_thw, 48, 2)

    assert actual.dtype == torch.float32
    assert_close(actual, expected, atol=5e-5, rtol=0.0)


def test_position_interpolation_bounds_four_tap_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep production video interpolation from materializing every wide gather at once."""
    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 128, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[15, 42, 74]], dtype=torch.long)
    temporary_sizes = []
    original_embedding = minimax_h3_qwen3_vl.F.embedding

    def recording_embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output_dtype = torch.promote_types(weight.dtype, torch.float32)
        temporary_sizes.append(indices.numel() * weight.shape[1] *
                               (weight.element_size() + torch.empty((), dtype=output_dtype).element_size()))
        return original_embedding(indices, weight)

    monkeypatch.setattr(minimax_h3_qwen3_vl.F, "embedding", recording_embedding)
    actual = _interpolate_vision_position_embeddings(position_embedding, grid_thw, 48, 2)
    expected = _transformers_5_15_contract_reference(position_embedding, grid_thw)

    assert len(temporary_sizes) > 1
    assert max(temporary_sizes) <= _VISION_INTERPOLATION_WORKSPACE_BYTES
    assert_close(actual, expected, atol=0.0, rtol=0.0)
