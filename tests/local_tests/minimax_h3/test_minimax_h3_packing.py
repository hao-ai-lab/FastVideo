# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference, assert_reference_source

assert_pinned_reference(
    "src/diffusers/modular_pipelines/minimax_h3/packing.py",
    "969d667d1c0316ed931d1675d474220045390e9a566cc885e3bd9cc6147b3e5b",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers.modular_pipelines.minimax_h3 import packing as reference  # noqa: E402

assert_reference_source(reference, "src/diffusers/modular_pipelines/minimax_h3/packing.py")

from fastvideo.pipelines.basic.minimax_h3 import packing as actual  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout  # noqa: E402


@pytest.mark.parametrize("ratio", [(16, 9), (9, 16), (1, 1), (4, 1), (1, 4)])
def test_canvas_and_frame_geometry_match_reference(ratio: tuple[int, int]) -> None:
    assert actual.resolve_canvas_size(*ratio) == reference.resolve_canvas_size(*ratio)
    for requested_frames in (1, 5, 6, 22, 120, 360):
        aligned = actual.align_num_frames(requested_frames)
        assert aligned == reference.align_num_frames(requested_frames)
        assert actual.video_latent_num_frames(aligned) == reference.video_latent_num_frames(aligned)
        assert actual.audio_latent_num_frames(aligned) == reference.audio_latent_num_frames(aligned)


def test_channel_major_patch_order_and_round_trip_match_reference() -> None:
    latents = torch.arange(2 * 3 * 4 * 6 * 8, dtype=torch.float32).reshape(2, 3, 4, 6, 8)
    patch_size = (2, 2, 4)
    expected = reference.patchify_video_latents(latents, patch_size)
    result = actual.patchify_video_latents(latents, patch_size)
    assert_close(result, expected, rtol=0, atol=0)
    assert result[0].tolist() == expected[0].tolist()
    restored = actual.unpatchify_video_tokens(result, 4, 6, 8, 3, patch_size)
    assert_close(restored, latents, rtol=0, atol=0)


@pytest.mark.parametrize("anchors", [(), ("first", ), ("first", "last")])
def test_layout_position_tags_indices_and_timesteps_match_reference(anchors: tuple[str, ...]) -> None:
    kwargs = {
        "text_token_tags": torch.tensor([1, 1, 0, 0, 1]),
        "num_latent_frames": 7,
        "latent_height": 6,
        "latent_width": 8,
        "num_audio_latents": 11,
        "patch_size": (1, 2, 2),
        "keyframe_anchors": anchors,
    }
    expected = reference.build_packed_sequence(**kwargs)
    result = actual.build_packed_sequence(**kwargs)
    assert isinstance(result, MiniMaxH3PackedLayout)
    assert result.sequence_length == expected.sequence_length
    for field in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert_close(getattr(result, field), getattr(expected, field), rtol=0, atol=0)
    assert result.num_condition_video_rows == expected.num_condition_video_rows
    assert result.num_condition_audio_rows == expected.num_condition_audio_rows

    expected_steps = reference.build_row_timesteps(expected, 0.25, 0.5, 0.999, 0.75)
    result_steps = actual.build_row_timesteps(result, 0.25, 0.5, 0.999, 0.75)
    assert_close(result_steps[0], expected_steps[0], rtol=0, atol=0)
    assert_close(result_steps[1], expected_steps[1], rtol=0, atol=0)


def test_condition_noise_preserves_draw_order() -> None:
    shapes = ((1, 4, 6), (2, 4, 6))
    kwargs = {"condition_latent_shapes": shapes, "patch_size": (1, 2, 2), "latent_channels": 3}
    reference_generator = torch.Generator().manual_seed(77)
    actual_generator = torch.Generator().manual_seed(77)
    expected = reference.keyframe_condition_noise(**kwargs, generator=reference_generator)
    result = actual.keyframe_condition_noise(**kwargs, generator=actual_generator)
    assert_close(result, expected, rtol=0, atol=0)
    assert_close(torch.randn(5, generator=actual_generator),
                 torch.randn(5, generator=reference_generator),
                 rtol=0,
                 atol=0)


def test_stereo_audio_rows_unpack_channel_major() -> None:
    rows = torch.arange(2 * 5 * 3).reshape(10, 3)
    assert_close(actual.unpack_audio_tokens(rows, 5), reference.unpack_audio_tokens(rows, 5), rtol=0, atol=0)
