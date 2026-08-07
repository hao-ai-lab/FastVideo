# SPDX-License-Identifier: Apache-2.0
"""Tests for SVI multi-clip inference."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import fastvideo.pipelines.basic.wan.wan_svi_i2v_pipeline as svi_pipeline
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


def test_stitch_drops_motion_tail_from_non_final_clips():
    num_motion = 2
    clips = [_clip(9, 0.0), _clip(9, 1.0), _clip(9, 2.0)]

    out = _stitch_clip_outputs(clips, num_motion)

    expected_t = 9 + (9 - num_motion) + (9 - num_motion)
    assert out.shape == (1, 3, expected_t, 2, 2)

    assert torch.all(out[:, :, :7] == 0.0)
    assert torch.all(out[:, :, 7:7 + 7] == 1.0)
    assert torch.all(out[:, :, 7 + 7:] == 2.0)


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
    assert [_clip_seed(0, idx, 42) for idx in range(3)] == [0, 42, 84]
    assert [_clip_seed(7, idx, 0) for idx in range(3)] == [7, 7, 7]


def test_svi_scheduler_uses_official_shift(monkeypatch):
    monkeypatch.setattr(svi_pipeline, "DenoisingStage", lambda **_kwargs: SimpleNamespace())
    pipeline = object.__new__(WanSVIImageToVideoPipeline)
    pipeline.modules = {}
    pipeline._stages = []
    pipeline._stage_name_mapping = {}

    args = SimpleNamespace(pipeline_config=SimpleNamespace(flow_shift=5.0))
    pipeline.create_pipeline_stages(args)

    assert pipeline.modules["scheduler"].shift == 5.0


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


def test_single_clip_uses_resized_svi_reference():
    pipeline = object.__new__(WanSVIImageToVideoPipeline)
    pipeline.post_init_called = True
    pipeline.input_validation_stage = lambda batch, _args: batch
    observed = []

    def fake_stage(batch, _args):
        observed.append((batch.pil_image.size, batch.svi_first_frames[0].size,
                         batch.svi_random_ref_frame.size))
        batch.output = _clip(4, 0.5)
        return batch

    pipeline._stages = [fake_stage]
    batch = ForwardBatch(
        data_type="i2v",
        prompt="prompt",
        seed=0,
        height=4,
        width=6,
        num_frames=4,
        num_inference_steps=2,
        svi_num_clips=1,
        pil_image=Image.new("RGB", (2, 2)),
    )

    output = pipeline.forward(batch, SimpleNamespace())

    assert output.output.shape == (1, 3, 4, 2, 2)
    assert observed == [((6, 4), (6, 4), (6, 4))]


def test_multiclip_forward_uses_motion_head_and_keeps_padding_reference():
    pipeline = object.__new__(WanSVIImageToVideoPipeline)
    pipeline.post_init_called = True
    pipeline.input_validation_stage = lambda batch, _args: batch

    observed = []

    def fake_stage(batch, _args):
        assert isinstance(batch.pil_image, Image.Image)
        assert batch.svi_first_frames
        assert isinstance(batch.svi_random_ref_frame, Image.Image)
        observed.append((
            batch.prompt,
            batch.seed,
            batch.pil_image.getpixel((0, 0)),
            batch.svi_first_frames[0].getpixel((0, 0)),
            batch.svi_random_ref_frame.getpixel((0, 0)),
        ))
        values = [0.0, 0.25, 0.5, 0.75] if len(observed) == 1 else [0.1, 0.2, 0.3, 0.4]
        batch.output = torch.tensor(values).view(1, 1, 4, 1, 1).expand(1, 3, 4, 2, 2)
        return batch

    pipeline._stages = [fake_stage]
    batch = ForwardBatch(
        data_type="i2v",
        prompt="fallback",
        seed=10,
        height=2,
        width=2,
        num_frames=4,
        num_inference_steps=2,
        svi_num_clips=2,
        svi_num_motion_frames=2,
        svi_seed_stride=42,
        svi_clip_prompts=["first", "second"],
        pil_image=Image.new("RGB", (2, 2), color=(7, 7, 7)),
    )

    output = pipeline.forward(batch, SimpleNamespace())

    assert output.output is not None
    assert output.output.shape == (1, 3, 6, 2, 2)
    assert observed == [
        ("first", 10, (7, 7, 7), (7, 7, 7), (7, 7, 7)),
        ("second", 52, (127, 127, 127), (127, 127, 127), (7, 7, 7)),
    ]
