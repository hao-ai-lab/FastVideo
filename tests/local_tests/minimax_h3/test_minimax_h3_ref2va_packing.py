# SPDX-License-Identifier: Apache-2.0
"""Implementation-subcomponent parity for MiniMax H3 Ref2VA packing."""

import sys

import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference

PARITY_SCOPE = "implementation_subcomponent"

assert_pinned_reference(
    "src/diffusers/modular_pipelines/minimax_h3/packing.py",
    "969d667d1c0316ed931d1675d474220045390e9a566cc885e3bd9cc6147b3e5b",
)
assert_pinned_reference(
    "src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py",
    "1f025b68af3f5c4316e2b89b026d4976c5b85286dafcc8b5c1d8ceb186c9bc01",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers.modular_pipelines.minimax_h3 import packing_ref2va as reference  # noqa: E402

from fastvideo.pipelines.basic.minimax_h3 import packing as base_packing  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3 import packing_ref2va as actual  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3PreparedReference  # noqa: E402


def _actual_mixed_references() -> list[MiniMaxH3PreparedReference]:
    return [
        MiniMaxH3PreparedReference(
            media_type="image",
            num_latent_frames=1,
            latent_height=4,
            latent_width=2,
        ),
        MiniMaxH3PreparedReference(
            media_type="video",
            has_audio=True,
            num_latent_frames=2,
            latent_height=2,
            latent_width=4,
            num_audio_latents=2,
        ),
        MiniMaxH3PreparedReference(media_type="audio", has_audio=True, num_audio_latents=1),
    ]


def _reference_mixed_references() -> list[reference.MiniMaxH3PreparedReference]:
    return [
        reference.MiniMaxH3PreparedReference(
            kind="image",
            num_latent_frames=1,
            latent_height=4,
            latent_width=2,
        ),
        reference.MiniMaxH3PreparedReference(
            kind="video",
            has_audio=True,
            num_latent_frames=2,
            latent_height=2,
            latent_width=4,
            num_audio_latents=2,
        ),
        reference.MiniMaxH3PreparedReference(kind="audio", has_audio=True, num_audio_latents=1),
    ]


def _mixed_layout_kwargs() -> dict:
    return {
        "text_token_tags": torch.tensor([1, 0, 1]),
        "num_latent_frames": 2,
        "latent_height": 4,
        "latent_width": 4,
        "num_audio_latents": 2,
        "patch_size": (1, 2, 2),
    }


def test_mixed_reference_layout_matches_handwritten_27_row_oracle() -> None:
    layout = actual.build_ref2va_packed_sequence(
        references=_actual_mixed_references(),
        **_mixed_layout_kwargs(),
    )

    narrow_grid = -6.627416997969519
    wide_grid = 4.686291501015241
    expected_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, narrow_grid, wide_grid],
            [3.0, 16.0, wide_grid],
            [4.0, 0.0, narrow_grid],
            [5.0, 0.0, narrow_grid],
            [4.0, 0.0, 16.0],
            [5.0, 0.0, 16.0],
            [4.0, wide_grid, narrow_grid],
            [4.0, wide_grid, 16.0],
            [17.0 / 3.0, wide_grid, narrow_grid],
            [17.0 / 3.0, wide_grid, 16.0],
            [37.0 / 3.0, 0.0, 0.0],
            [37.0 / 3.0, 0.0, 16.0],
            [40.0 / 3.0, 0.0, 0.0],
            [43.0 / 3.0, 0.0, 0.0],
            [40.0 / 3.0, 0.0, 16.0],
            [43.0 / 3.0, 0.0, 16.0],
            [40.0 / 3.0, 0.0, 0.0],
            [40.0 / 3.0, 0.0, 16.0],
            [40.0 / 3.0, 16.0, 0.0],
            [40.0 / 3.0, 16.0, 16.0],
            [15.0, 0.0, 0.0],
            [15.0, 0.0, 16.0],
            [15.0, 16.0, 0.0],
            [15.0, 16.0, 16.0],
        ],
        dtype=torch.float64,
    )
    expected_tags = torch.tensor([
        1,
        0,
        1,
        0,
        0,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
    expected_video_indices = torch.tensor([3, 4, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26])
    expected_audio_indices = torch.tensor([5, 6, 7, 8, 13, 14, 15, 16, 17, 18])

    assert layout.sequence_length == 27
    assert_close(layout.position_ids, expected_positions, rtol=0, atol=0)
    assert_close(layout.token_tags, expected_tags, rtol=0, atol=0)
    assert_close(layout.video_indices, expected_video_indices, rtol=0, atol=0)
    assert_close(layout.audio_indices, expected_audio_indices, rtol=0, atol=0)
    assert_close(layout.text_indices, torch.tensor([0, 1, 2]), rtol=0, atol=0)
    assert layout.num_condition_video_rows == 6
    assert layout.num_condition_audio_rows == 6
    assert (layout.num_video_latent_frames, layout.latent_height, layout.latent_width,
            layout.num_audio_latents) == (2, 4, 4, 2)

    unique_timesteps, timestep_indices = base_packing.build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    expected_timestep_indices = torch.tensor([
        0,
        0,
        0,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        3,
        3,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
    assert_close(unique_timesteps, torch.tensor([0.25, 0.5, 0.999, 1.0]), rtol=0, atol=0)
    assert_close(timestep_indices, expected_timestep_indices, rtol=0, atol=0)


def test_ref2va_layout_matches_pinned_diffusers_exactly() -> None:
    kwargs = _mixed_layout_kwargs()
    expected = reference.build_ref2va_packed_sequence(
        references=_reference_mixed_references(),
        **kwargs,
    )
    result = actual.build_ref2va_packed_sequence(
        references=_actual_mixed_references(),
        **kwargs,
    )

    assert result.sequence_length == expected.sequence_length
    for field in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert_close(getattr(result, field), getattr(expected, field), rtol=0, atol=0)
    assert result.num_condition_video_rows == expected.num_condition_video_rows
    assert result.num_condition_audio_rows == expected.num_condition_audio_rows


def test_video_reference_clock_uses_sequential_span() -> None:
    reference_video = MiniMaxH3PreparedReference(
        media_type="video",
        num_latent_frames=15,
        latent_height=2,
        latent_width=2,
    )
    layout = actual.build_ref2va_packed_sequence(
        text_token_tags=torch.tensor([1]),
        references=[reference_video],
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=(1, 2, 2),
    )

    assert actual._reference_temporal_span(15) == 85.0
    assert base_packing._temporal_position_span(15) == 85.00000000000001
    assert_close(layout.position_ids[16:, 0], torch.full((3, ), 86.0, dtype=torch.float64), rtol=0, atol=0)


def test_paired_soundtrack_differs_from_separate_audio_reference() -> None:
    paired_references = [
        MiniMaxH3PreparedReference(
            media_type="video",
            has_audio=True,
            num_latent_frames=1,
            latent_height=2,
            latent_width=4,
            num_audio_latents=2,
        )
    ]
    separate_references = [
        MiniMaxH3PreparedReference(
            media_type="video",
            num_latent_frames=1,
            latent_height=2,
            latent_width=4,
        ),
        MiniMaxH3PreparedReference(media_type="audio", has_audio=True, num_audio_latents=2),
    ]
    kwargs = {
        "text_token_tags": torch.tensor([1]),
        "num_latent_frames": 1,
        "latent_height": 4,
        "latent_width": 4,
        "num_audio_latents": 1,
        "patch_size": (1, 2, 2),
    }

    paired = actual.build_ref2va_packed_sequence(references=paired_references, **kwargs)
    separate = actual.build_ref2va_packed_sequence(references=separate_references, **kwargs)

    assert paired.sequence_length == separate.sequence_length == 13
    assert paired.video_indices.tolist() == [5, 6, 9, 10, 11, 12]
    assert paired.audio_indices.tolist() == [1, 2, 3, 4, 7, 8]
    assert separate.video_indices.tolist() == [1, 2, 9, 10, 11, 12]
    assert separate.audio_indices.tolist() == [3, 4, 5, 6, 7, 8]
    assert paired.position_ids[1, 2].item() == -6.627416997969519
    assert separate.position_ids[3, 2].item() == 0.0
    assert_close(paired.position_ids[7:, 0], torch.full((6, ), 3.0, dtype=torch.float64), rtol=0, atol=0)
    assert_close(
        separate.position_ids[7:, 0],
        torch.full((6, ), 14.0 / 3.0, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


class _PresentationTokenizer:
    text_ids = {
        "<Picture 1>: ": 10,
        "<Audio 1>: ": 11,
        "<Video 1>: ": 12,
        "<0.2 seconds>": 13,
        "<1.0 seconds>": 14,
        "<Audio 2>: ": 15,
        "dance": 16,
    }
    special_ids = {
        "<|vision_start|>": 100,
        "<|image_pad|>": 101,
        "<|video_pad|>": 102,
        "<|vision_end|>": 103,
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, value: str, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        self.calls.append(value)
        return {"input_ids": [self.text_ids[value]]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_ids[token]


def test_presentation_preserves_reference_and_soundtrack_order() -> None:
    actual_references = [
        MiniMaxH3PreparedReference(media_type="image"),
        MiniMaxH3PreparedReference(
            media_type="video",
            has_audio=True,
            block_timestamps=[0.25, 1.0],
        ),
        MiniMaxH3PreparedReference(media_type="audio", has_audio=True),
    ]
    reference_references = [
        reference.MiniMaxH3PreparedReference(kind="image"),
        reference.MiniMaxH3PreparedReference(
            kind="video",
            has_audio=True,
            block_timestamps=[0.25, 1.0],
        ),
        reference.MiniMaxH3PreparedReference(kind="audio", has_audio=True),
    ]
    actual_tokenizer = _PresentationTokenizer()
    reference_tokenizer = _PresentationTokenizer()

    result = actual.build_ref2va_presentation(
        actual_tokenizer,
        "dance",
        actual_references,
        image_token_counts=[2],
        video_block_token_counts=[1],
    )
    expected_from_reference = reference.build_ref2va_presentation(
        reference_tokenizer,
        "dance",
        reference_references,
        image_token_counts=[2],
        video_block_token_counts=[1],
    )
    expected_calls = [
        "<Picture 1>: ",
        "<Audio 1>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "<1.0 seconds>",
        "<Audio 2>: ",
        "dance",
    ]
    expected_ids = [10, 100, 101, 101, 103, 11, 12, 13, 100, 102, 103, 14, 100, 102, 103, 15, 16]
    expected_tags = [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]

    assert actual_tokenizer.calls == reference_tokenizer.calls == expected_calls
    assert result == expected_from_reference == (expected_ids, expected_tags)
