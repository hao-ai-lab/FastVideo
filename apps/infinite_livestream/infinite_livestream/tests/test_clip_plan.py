"""Clip geometry must match the checkpoint FastVideo actually ships.

`clip_plan` duplicates MiniMax-H3's packing constants instead of importing
them, because the upstream module pulls in torch and -- through
fastvideo-kernel's triton autotuning -- needs a live CUDA driver merely to
import, which would put a GPU in the path of every config test.

Duplication is only safe if something checks it, so that check is here. It
needs the driver, hence the `gpu` marker: run it whenever the pinned FastVideo
version moves, not in CI.

The arithmetic tests below need none of that and run anywhere.
"""

from __future__ import annotations

import pytest

from infinite_livestream import clip_plan


@pytest.mark.gpu
def test_constants_match_upstream() -> None:
    from fastvideo.pipelines.basic.minimax_h3 import packing

    assert clip_plan.FPS == packing.MINIMAX_H3_FPS
    assert clip_plan._SHORT_EDGE == packing.MINIMAX_H3_SHORT_EDGE
    assert clip_plan._MAX_PIXELS == packing.MINIMAX_H3_MAX_PIXELS
    assert clip_plan._CANVAS_MULTIPLE == packing.MINIMAX_H3_CANVAS_MULTIPLE
    assert clip_plan._MIN_DURATION == packing.MINIMAX_H3_MIN_DURATION
    assert clip_plan._MAX_DURATION == packing.MINIMAX_H3_MAX_DURATION
    assert clip_plan._FRAMES_PER_CHUNK == packing.MINIMAX_H3_FRAMES_PER_CHUNK
    assert clip_plan._LATENTS_PER_CHUNK == packing.MINIMAX_H3_LATENTS_PER_CHUNK
    assert clip_plan._MIN_ASPECT == packing.MINIMAX_H3_MIN_ASPECT_RATIO
    assert clip_plan._MAX_ASPECT == packing.MINIMAX_H3_MAX_ASPECT_RATIO


def test_every_legal_length_round_trips() -> None:
    """`frames_for_seconds` must land on something the checkpoint can build."""
    legal = set(clip_plan.legal_frame_counts())
    assert legal, "the checkpoint must admit at least one clip length"
    for frames in legal:
        seconds = clip_plan.seconds_for_frames(frames)
        assert clip_plan.frames_for_seconds(seconds) == frames


def test_published_range_is_generatable() -> None:
    """Every value a client may legally ask for must snap into range.

    The published bounds are rounded inward precisely so this holds; rounding
    outward would advertise a length the model then refuses.
    """
    for seconds in (
        clip_plan.MIN_SECONDS_PUBLISHED,
        clip_plan.MAX_SECONDS_PUBLISHED,
        (clip_plan.MIN_SECONDS_PUBLISHED + clip_plan.MAX_SECONDS_PUBLISHED) / 2,
    ):
        frames = clip_plan.frames_for_seconds(seconds)
        assert frames in clip_plan.legal_frame_counts()


def test_max_frames_respects_the_duration_cap() -> None:
    """The ceiling is the subtle one: 15.0s aligns up to 362f, which is illegal."""
    assert clip_plan.MAX_FRAMES == 345
    assert clip_plan.seconds_for_frames(clip_plan.MAX_FRAMES) <= clip_plan._MAX_DURATION
    assert clip_plan.align_frames(clip_plan.MAX_FRAMES + 1) / clip_plan.FPS > clip_plan._MAX_DURATION


def test_canvases_land_on_the_multiple_and_under_the_area_cap() -> None:
    for aspect in clip_plan.ASPECT_CHOICES:
        height, width = clip_plan.canvas_for_choice(aspect)
        assert height % clip_plan._CANVAS_MULTIPLE == 0
        assert width % clip_plan._CANVAS_MULTIPLE == 0
        assert height * width <= clip_plan._MAX_PIXELS


def test_illegal_aspects_are_refused() -> None:
    with pytest.raises(ValueError):
        clip_plan.canvas_for_aspect(5, 1)      # past the 4:1 cap
    with pytest.raises(ValueError):
        clip_plan.canvas_for_aspect(0, 1)
    with pytest.raises(ValueError):
        clip_plan.canvas_for_choice("21:9")    # not an offered choice
