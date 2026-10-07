# SPDX-License-Identifier: Apache-2.0
"""Pinned Diffusers parity for MiniMax H3 Ref2VA packing."""

import sys

import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference, assert_reference_source

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

assert_reference_source(reference, "src/diffusers/modular_pipelines/minimax_h3/packing_ref2va.py")

from fastvideo.pipelines.basic.minimax_h3 import packing as actual  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference  # noqa: E402
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (  # noqa: E402
    build_ref2va_presentation, )


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

    result = build_ref2va_presentation(
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
