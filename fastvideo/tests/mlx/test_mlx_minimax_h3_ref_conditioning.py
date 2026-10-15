# SPDX-License-Identifier: Apache-2.0
"""Ref2VA conditioning: references become presentation rows and vision inputs.

This is the seam between the reference preparation, the Qwen3-VL processor
equivalent and the conditioner. A recording fake conditioner captures what the
conditioner would actually receive, so the placeholder counts, the vision
grids and the block timestamps are all pinned without weights.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for Ref2VA conditioning tests")

from PIL import Image  # noqa: E402

from fastvideo.mlx_runtime.minimax_h3_ref_conditioning import (  # noqa: E402
    encode_reference_prompt,
    load_vision_limits,
)
from fastvideo.mlx_runtime.minimax_h3_reference import (  # noqa: E402
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_VIDEO_PAD_TOKEN,
    MiniMaxH3PreparedReference,
)
from fastvideo.mlx_runtime.minimax_h3_vision import Qwen3VLVisionConfig  # noqa: E402

VISION_CONFIG = Qwen3VLVisionConfig(
    hidden_size=32,
    depth=2,
    num_heads=4,
    intermediate_size=48,
    in_channels=3,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=32,
    num_position_embeddings=64,
    deepstack_visual_indexes=(0, ),
    hidden_act="gelu_pytorch_tanh",
)
HIDDEN = 12


class _FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|vision_start|>": 1000, MINIMAX_H3_IMAGE_PAD_TOKEN: 1001, MINIMAX_H3_VIDEO_PAD_TOKEN: 1002,
                "<|vision_end|>": 1003}[token]


class _RecordingConditioner:
    tokenizer = _FakeTokenizer()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode_tokens_with_vision(self, token_ids, token_tags, **vision):
        self.calls.append({"token_ids": list(token_ids), "token_tags": np.asarray(token_tags), **vision})
        return np.zeros((len(token_ids), HIDDEN), dtype=np.float32), np.asarray(token_tags, dtype=np.int64)


def _image(width: int = 64, height: int = 64) -> MiniMaxH3PreparedReference:
    pixels = np.random.default_rng(5).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return MiniMaxH3PreparedReference(media_type="image", image=Image.fromarray(pixels))


def _video(frames: int = 24) -> MiniMaxH3PreparedReference:
    data = np.random.default_rng(7).integers(0, 256, size=(frames, 64, 64, 3), dtype=np.uint8)
    return MiniMaxH3PreparedReference(media_type="video", frames=data)


def test_encode_reference_prompt_pairs_placeholders_with_vision_rows() -> None:
    """Every vision placeholder gets exactly one feature row, in order."""
    conditioner = _RecordingConditioner()
    hidden, tags = encode_reference_prompt(
        conditioner,
        "a prompt",
        [_image()],
        vision_config=VISION_CONFIG,
        min_pixels=1024,
        max_pixels=65536,
    )

    assert hidden.shape == (len(tags), HIDDEN)
    assert tags.ndim == 1
    call = conditioner.calls[0]
    image_rows = sum(1 for token in call["token_ids"] if token == 1001)
    grid = call["image_grid_thw"][0]
    # One placeholder per merged token; pixel_values keeps one row per raw patch.
    assert image_rows == int(np.prod(grid)) // VISION_CONFIG.spatial_merge_size**2
    assert call["pixel_values"].shape[0] == int(np.prod(grid))
    assert call["image_grid_thw"].shape == (1, 3)
    assert call["pixel_values_videos"] is None


def test_encode_reference_prompt_assigns_video_block_timestamps() -> None:
    """The 2-fps presentation sampling is what stamps a video reference's blocks."""
    conditioner = _RecordingConditioner()
    reference = _video()
    assert reference.block_timestamps == []

    encode_reference_prompt(
        conditioner,
        "p",
        [reference],
        vision_config=VISION_CONFIG,
        min_pixels=1024,
        max_pixels=65536,
    )

    assert reference.block_timestamps
    call = conditioner.calls[0]
    # One video block per 2-fps sample, each labelled and padded in the presentation.
    assert call["video_grid_thw"].shape[0] == len(reference.block_timestamps)
    video_rows = sum(1 for token in call["token_ids"] if token == 1002)
    pad = VISION_CONFIG.spatial_merge_size**2
    assert video_rows == int(np.prod(call["video_grid_thw"][:, 1:])) // pad


def test_encode_reference_prompt_orders_references_as_declared() -> None:
    """A video reference before an image one presents in that order."""
    conditioner = _RecordingConditioner()
    encode_reference_prompt(
        conditioner,
        "p",
        [_video(), _image()],
        vision_config=VISION_CONFIG,
        min_pixels=1024,
        max_pixels=65536,
    )
    token_ids = conditioner.calls[0]["token_ids"]
    first_video = next(index for index, token in enumerate(token_ids) if token == 1002)
    first_image = next(index for index, token in enumerate(token_ids) if token == 1001)
    assert first_video < first_image


def test_vision_token_counts_use_the_label_presentation() -> None:
    """The picture label leads its vision block, and only the block is tagged video."""
    conditioner = _RecordingConditioner()
    _, tags = encode_reference_prompt(
        conditioner,
        "p",
        [_image()],
        vision_config=VISION_CONFIG,
        min_pixels=1024,
        max_pixels=65536,
    )
    token_ids = conditioner.calls[0]["token_ids"]
    label = [ord(character) for character in "<Picture 1>: "]
    assert token_ids[:len(label)] == label
    assert list(tags[:len(label)]) == [1] * len(label)
    # The vision block that follows is tagged video, up to the prompt tail.
    vision_end = next(index for index, token in enumerate(token_ids) if token == 1003)
    assert set(tags[len(label):vision_end + 1]) == {0}
    assert set(tags[vision_end + 1:]) == {1}


def test_load_vision_limits_reads_the_processor_config(tmp_path) -> None:
    assert load_vision_limits(tmp_path) == (65536, 16777216)
    (tmp_path / "preprocessor_config.json").write_text('{"min_pixels": 1024, "max_pixels": 4096}')
    assert load_vision_limits(tmp_path) == (1024, 4096)
