# SPDX-License-Identifier: Apache-2.0
"""Tests for SVI multi-clip stitch index math and the frame-budget guard."""
from __future__ import annotations

import pytest
import torch

from fastvideo.pipelines.basic.wan.wan_svi_i2v_pipeline import (_stitch_clip_outputs, _validate_multiclip_frames)


def _clip(num_frames: int, fill: float) -> torch.Tensor:
    # (B=1, C=3, T=num_frames, H=2, W=2)
    return torch.full((1, 3, num_frames, 2, 2), fill, dtype=torch.float32)


def test_stitch_drops_motion_overlap_from_followups():
    num_motion = 2
    clips = [_clip(9, 0.0), _clip(9, 1.0), _clip(9, 2.0)]

    out = _stitch_clip_outputs(clips, num_motion)

    expected_t = 9 + (9 - num_motion) + (9 - num_motion)
    assert out.shape == (1, 3, expected_t, 2, 2)

    # Provenance: 9 from clip0, then 7 from clip1, then 7 from clip2.
    assert torch.all(out[:, :, :9] == 0.0)
    assert torch.all(out[:, :, 9:9 + 7] == 1.0)
    assert torch.all(out[:, :, 9 + 7:] == 2.0)


def test_stitch_single_clip_is_identity():
    clip = _clip(5, 3.0)
    out = _stitch_clip_outputs([clip], num_motion=2)
    assert torch.equal(out, clip)


def test_stitch_num_motion_one_drops_single_frame():
    clips = [_clip(4, 0.0), _clip(4, 1.0)]
    out = _stitch_clip_outputs(clips, num_motion=1)
    assert out.shape[2] == 4 + 3


def test_validate_rejects_motion_ge_frames():
    with pytest.raises(ValueError, match="must be smaller than num_frames"):
        _validate_multiclip_frames(num_motion=5, num_frames=5)
    with pytest.raises(ValueError, match="must be smaller than num_frames"):
        _validate_multiclip_frames(num_motion=8, num_frames=5)


def test_validate_accepts_motion_lt_frames():
    # Should not raise.
    _validate_multiclip_frames(num_motion=1, num_frames=81)
    _validate_multiclip_frames(num_motion=5, num_frames=6)
