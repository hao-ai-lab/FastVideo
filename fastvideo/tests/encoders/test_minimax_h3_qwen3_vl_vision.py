# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for MiniMax-H3's Qwen3-VL vision position embeddings."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from fastvideo.models.encoders.minimax_h3_qwen3_vl import _interpolate_vision_position_embeddings


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


@pytest.mark.parametrize(
    "grid_thw",
    (
        torch.tensor([[1, 4, 6]], dtype=torch.long),
        torch.tensor([[2, 4, 6]], dtype=torch.long),
        torch.tensor([[1, 4, 6], [2, 8, 10]], dtype=torch.long),
    ),
)
def test_position_interpolation_matches_transformers_5_15(grid_thw: torch.Tensor) -> None:
    """Pin the visual-token ordering and float32 accumulation used upstream."""
    vision_utils = pytest.importorskip("transformers.vision_utils")
    get_reference = getattr(vision_utils, "get_vision_interpolation_indices_and_weights", None)
    if get_reference is None:
        pytest.skip("requires the Transformers 5.15 Qwen3-VL interpolation reference")

    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 32, dtype=torch.bfloat16)
    indices, weights = get_reference(
        grid_thw,
        num_grid_per_side=48,
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=2,
    )
    expected = (position_embedding[indices] * weights[:, :, None]).sum(1)

    actual = _interpolate_vision_position_embeddings(position_embedding, grid_thw, 48, 2)

    assert actual.dtype == torch.float32
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "grid_thw",
    (
        torch.tensor([[1, 4, 6]], dtype=torch.long),
        torch.tensor([[2, 4, 6]], dtype=torch.long),
        torch.tensor([[1, 4, 6], [2, 8, 10]], dtype=torch.long),
    ),
)
def test_position_interpolation_uses_float32_bilinear_accumulation(grid_thw: torch.Tensor) -> None:
    """Prevent bf16 interpolation weights from reintroducing visual-token drift."""
    torch.manual_seed(1733)
    position_embedding = torch.randn(48 * 48, 32, dtype=torch.bfloat16)

    expected = _torch_interpolation_reference(position_embedding, grid_thw, 48, 2)
    actual = _interpolate_vision_position_embeddings(position_embedding, grid_thw, 48, 2)

    assert actual.dtype == torch.float32
    assert_close(actual, expected, atol=5e-5, rtol=0.0)
