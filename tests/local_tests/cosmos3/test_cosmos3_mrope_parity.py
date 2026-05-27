# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 unified 3D mRoPE position-ID parity (Tier A scaffold).

Reference: ``vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py``
lines 113-177 (``compute_mrope_position_ids_text`` /
``compute_mrope_position_ids_vision``). The reference test asserting
these invariants lives at
``tests/diffusion/models/cosmos3/test_cosmos3_transformer.py:32-57``.

The three invariants under test:

  1. Text tokens broadcast the same monotonically-increasing positions
     across all three (t, h, w) axes. With ``num_tokens=3`` and
     ``temporal_offset=5`` the result is ``[[5,6,7], [5,6,7], [5,6,7]]``
     and the next-offset is ``8``.

  2. Vision tokens (no FPS modulation) flatten a ``(grid_t, grid_h, grid_w)``
     position grid in t-major order. With ``(2, 2, 3)`` and offset ``10``
     the resulting shape is ``(3, 12)`` and the temporal row begins
     ``[10]*6 + [11]*6``; next-offset is ``12``.

  3. FPS-modulated vision tokens scale the temporal axis by
     ``base_fps / temporal_compression_factor / (fps / tcf)``. With
     ``fps=12``, ``base_fps=24``, ``tcf=4``, ``grid_t=2`` the first row is
     ``[10.0, 12.0]``.

The FastVideo side currently does NOT exist; the test is wrapped in
``try/except ImportError`` and skips. Phase 2b replaces the skip with
the real import + assertion path.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.local]


def test_compute_mrope_position_ids_text_and_vision() -> None:
    """Asserts the 3 invariants of unified 3D mRoPE position-ID generation.

    Once FastVideo's ``fastvideo.models.dits.cosmos3`` exports
    ``compute_mrope_position_ids_text`` and
    ``compute_mrope_position_ids_vision``, this test verifies they produce
    output tensors identical to the vllm-omni reference at
    transformer_cosmos3.py:113-177.
    """
    try:
        from fastvideo.models.dits.cosmos3 import (  # type: ignore
            compute_mrope_position_ids_text,
            compute_mrope_position_ids_vision,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 not yet implemented (Phase 2b)")

    text_ids, text_offset = compute_mrope_position_ids_text(num_tokens=3, temporal_offset=5)
    assert text_ids.tolist() == [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    assert text_offset == 8

    vision_ids, vision_offset = compute_mrope_position_ids_vision(
        2, 2, 3, temporal_offset=10, fps=None
    )
    assert tuple(vision_ids.shape) == (3, 12)
    assert vision_ids[0].tolist() == [10] * 6 + [11] * 6
    assert vision_offset == 12

    modulated_ids, modulated_offset = compute_mrope_position_ids_vision(
        2,
        1,
        1,
        temporal_offset=10,
        fps=12.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )
    torch.testing.assert_close(modulated_ids[0], torch.tensor([10.0, 12.0]))
    assert modulated_offset == 13
