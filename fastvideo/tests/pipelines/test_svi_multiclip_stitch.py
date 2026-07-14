# SPDX-License-Identifier: Apache-2.0
"""Tests for SVI multi-clip stitch index math and the frame-budget guard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from fastvideo.pipelines.basic.wan.wan_svi_i2v_pipeline import (
    WanSVIImageToVideoPipeline,
    _clip_seed,
    _resolve_clip_prompts,
    _stitch_clip_outputs,
    _validate_multiclip_frames,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


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


def test_clip_seed_offsets_each_chunk():
    assert [_clip_seed(42, idx) for idx in range(3)] == [42, 43, 44]


def test_resolve_clip_prompts_reuses_primary_prompt_by_default():
    assert _resolve_clip_prompts("one prompt", None, 3) == ["one prompt"] * 3


def test_resolve_clip_prompts_accepts_exact_per_clip_prompts():
    prompts = ["first", "second"]
    assert _resolve_clip_prompts("fallback", prompts, 2) == prompts


def test_resolve_clip_prompts_rejects_wrong_length_or_empty_entries():
    with pytest.raises(ValueError, match="exactly svi_num_clips"):
        _resolve_clip_prompts("fallback", ["only one"], 2)
    with pytest.raises(ValueError, match="non-empty strings"):
        _resolve_clip_prompts("fallback", ["first", ""], 2)


def test_multiclip_forward_uses_distinct_prompts_seeds_and_last_tail_frame():
    pipeline = object.__new__(WanSVIImageToVideoPipeline)
    pipeline.post_init_called = True
    pipeline.input_validation_stage = lambda batch, _args: batch

    observed: list[tuple[str | list[str] | None, int | None, tuple[int, int, int]]] = []

    def fake_stage(batch, _args):
        assert isinstance(batch.pil_image, Image.Image)
        observed.append((batch.prompt, batch.seed, batch.pil_image.getpixel((0, 0))))
        fill = 0.5 if len(observed) == 1 else 0.75
        batch.output = torch.full((1, 3, 4, 2, 2), fill)
        return batch

    pipeline._stages = [fake_stage]
    batch = ForwardBatch(
        data_type="i2v",
        prompt="fallback",
        seed=10,
        num_frames=4,
        num_inference_steps=2,
        svi_num_clips=2,
        svi_num_motion_frames=1,
        svi_clip_prompts=["first", "second"],
        pil_image=Image.new("RGB", (2, 2), color=(7, 7, 7)),
    )

    output = pipeline.forward(batch, SimpleNamespace())

    assert output.output is not None
    assert output.output.shape == (1, 3, 7, 2, 2)
    assert [(prompt, seed) for prompt, seed, _pixel in observed] == [
        ("first", 10),
        ("second", 11),
    ]
    assert observed[0][2] == (7, 7, 7)
    assert observed[1][2] == (128, 128, 128)
