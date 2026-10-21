# SPDX-License-Identifier: Apache-2.0
"""Reference preparation parity: the MLX port vs the upstream PyTorch helpers.

``fastvideo.mlx_runtime.minimax_h3_reference`` re-implements
``fastvideo.pipelines.basic.minimax_h3.reference`` without torch. The reference
canvas it resolves feeds the video VAE and the Qwen presentation on both
runtimes, so a drift here shows up as a different clip rather than an error.
Both implementations are driven with the same inputs and compared.

Audio resampling is the one deliberate divergence: the MLX side uses SciPy's
polyphase filter because torchaudio is not part of an MLX-only install, so that
case is compared by correlation and band energy rather than bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastvideo.mlx_runtime import minimax_h3_reference as mlx_reference
from fastvideo.pipelines.basic.minimax_h3 import packing as torch_packing
from fastvideo.pipelines.basic.minimax_h3 import reference as torch_reference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
    build_ref2va_presentation as torch_presentation,
)

NUM_FRAMES = 40
TARGET_SAMPLE_RATE = 32000


def _frame_bytes(seed: int, frames: int, height: int, width: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(frames, height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("size", [(64, 48), (48, 64), (96, 96), (200, 64)])
def test_image_reference_matches_upstream(size: tuple[int, int]) -> None:
    width, height = size
    generator = np.random.default_rng(7)
    pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    expected = torch_reference.prepare_reference(
        torch_reference.MiniMaxH3Reference(source=pixels), NUM_FRAMES, TARGET_SAMPLE_RATE)
    actual = mlx_reference.prepare_reference(
        mlx_reference.MiniMaxH3Reference(source=pixels), NUM_FRAMES, TARGET_SAMPLE_RATE)

    assert actual.image.size == expected.image.size
    assert actual.media_type == expected.media_type
    np.testing.assert_array_equal(np.asarray(actual.image), np.asarray(expected.image))


def test_reference_image_size_resolution_matches_upstream() -> None:
    for width, height in [(64, 48), (2048, 512), (512, 2048), (1000, 999), (33, 32)]:
        assert mlx_reference.resolve_reference_image_size(width, height) == \
            torch_reference.resolve_reference_image_size(width, height)
    with pytest.raises(ValueError, match="1:4 and 4:1"):
        mlx_reference.resolve_reference_image_size(500, 100)
    with pytest.raises(ValueError, match="1:4 and 4:1"):
        torch_reference.resolve_reference_image_size(500, 100)


def test_canvas_resolution_matches_upstream() -> None:
    for width, height in [(64, 48), (32, 32), (128, 32), (32, 128), (1000, 300)]:
        assert mlx_reference.resolve_canvas_size(width, height) == torch_packing.resolve_canvas_size(width, height)
    # Beyond 4:1 both reject rather than resolving a canvas.
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        mlx_reference.resolve_canvas_size(2000, 300)
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        torch_packing.resolve_canvas_size(2000, 300)


def test_video_reference_matches_upstream() -> None:
    frames = _frame_bytes(11, 50, 24, 32)

    expected = torch_reference.prepare_reference(
        torch_reference.MiniMaxH3Reference(source=frames, media_type="video"), NUM_FRAMES, TARGET_SAMPLE_RATE)
    actual = mlx_reference.prepare_reference(
        mlx_reference.MiniMaxH3Reference(source=frames, media_type="video"), NUM_FRAMES, TARGET_SAMPLE_RATE)

    assert actual.frames.shape == expected.frames.shape
    assert actual.has_audio == expected.has_audio is False
    np.testing.assert_array_equal(actual.frames, expected.frames)


def test_video_reference_fps_resample_matches_upstream() -> None:
    """A 12 fps source is nearest-resampled onto the 24 fps timeline identically."""
    frames = _frame_bytes(13, 40, 16, 16)
    expected = torch_reference.prepare_reference(
        torch_reference.MiniMaxH3Reference(source=frames, media_type="video", fps=12.0), NUM_FRAMES, TARGET_SAMPLE_RATE)
    actual = mlx_reference.prepare_reference(
        mlx_reference.MiniMaxH3Reference(source=frames, media_type="video", fps=12.0), NUM_FRAMES, TARGET_SAMPLE_RATE)
    assert actual.frames.shape == expected.frames.shape
    np.testing.assert_array_equal(actual.frames, expected.frames)


def test_video_reference_trim_matches_upstream() -> None:
    frames = _frame_bytes(17, 50, 16, 16)
    for num_frames in (1, 20, 40, 50):
        expected = torch_reference.prepare_reference(
            torch_reference.MiniMaxH3Reference(source=frames, media_type="video"), num_frames, TARGET_SAMPLE_RATE)
        actual = mlx_reference.prepare_reference(
            mlx_reference.MiniMaxH3Reference(source=frames, media_type="video"), num_frames, TARGET_SAMPLE_RATE)
        assert actual.frames.shape == expected.frames.shape


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_reference_waveform_matches_upstream(channels: int) -> None:
    """Mono is expanded to stereo and trimmed to the request, identically."""
    samples = np.random.default_rng(23).standard_normal((channels, TARGET_SAMPLE_RATE)).astype(np.float32)

    expected = torch_reference.prepare_reference(
        torch_reference.MiniMaxH3Reference(source=samples, media_type="audio"), NUM_FRAMES, TARGET_SAMPLE_RATE)
    actual = mlx_reference.prepare_reference(
        mlx_reference.MiniMaxH3Reference(source=samples, media_type="audio"), NUM_FRAMES, TARGET_SAMPLE_RATE)

    assert actual.waveform.shape == tuple(expected.waveform.shape)
    assert actual.has_audio is True
    np.testing.assert_allclose(actual.waveform, expected.waveform.numpy(), atol=1e-6)


def test_audio_reference_resample_tracks_upstream() -> None:
    """Different resamplers: the envelope and length must still agree."""
    samples = np.random.default_rng(29).standard_normal((1, 44100)).astype(np.float32)
    expected = torch_reference.prepare_reference(
        torch_reference.MiniMaxH3Reference(source=samples, media_type="audio", sample_rate=44100), NUM_FRAMES,
        TARGET_SAMPLE_RATE)
    actual = mlx_reference.prepare_reference(
        mlx_reference.MiniMaxH3Reference(source=samples, media_type="audio", sample_rate=44100), NUM_FRAMES,
        TARGET_SAMPLE_RATE)

    reference_waveform = expected.waveform.numpy()
    assert actual.waveform.shape == tuple(reference_waveform.shape)
    assert abs(np.mean(np.abs(actual.waveform)) - np.mean(np.abs(reference_waveform))) < 0.05
    assert np.corrcoef(actual.waveform.ravel(), reference_waveform.ravel())[0, 1] > 0.2


def test_sample_reference_video_frames_matches_upstream() -> None:
    frames = _frame_bytes(31, 30, 8, 8)
    expected_frames, expected_timestamps = torch_reference.sample_reference_video_frames(frames)
    actual_frames, actual_timestamps = mlx_reference.sample_reference_video_frames(frames)
    assert actual_timestamps == expected_timestamps
    assert len(actual_frames) == len(expected_frames)
    for actual, expected in zip(actual_frames, expected_frames, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_trim_reference_num_frames_matches_upstream() -> None:
    for num_frames in (1, 5, 21, 22, 40, 360):
        assert mlx_reference.trim_reference_num_frames(num_frames) == \
            torch_reference.trim_reference_num_frames(num_frames)


def test_reference_validation_matches_upstream() -> None:
    for module in (mlx_reference, torch_reference):
        images = [module.MiniMaxH3Reference(source=np.zeros((4, 4, 3), dtype=np.uint8))] * 3
        assert len(module.validate_references(images)) == 3

        audio_only = module.MiniMaxH3Reference(source=np.zeros((2, 8), dtype=np.float32), media_type="audio")
        with pytest.raises(ValueError, match="at least one image or video"):
            module.validate_references([audio_only])

        too_many = [module.MiniMaxH3Reference(source=np.zeros((4, 4, 3), dtype=np.uint8)) for _ in range(10)]
        with pytest.raises(ValueError, match="at most 9 image references"):
            module.validate_references(too_many)


def test_presentation_matches_upstream() -> None:
    """Labels, vision blocks and tags are emitted in the same order."""
    tokenizer = _FakeTokenizer()
    torch_references = [_prepared_image()[0], _prepared_video()[0]]
    mlx_references = [_prepared_image()[1], _prepared_video()[1]]

    expected_ids, expected_tags = torch_presentation(tokenizer, "a prompt", torch_references, [3], [2])
    actual_ids, actual_tags = mlx_reference.build_ref2va_presentation(
        tokenizer, "a prompt", mlx_references, [3], [2])

    assert actual_ids == expected_ids
    assert actual_tags == expected_tags


class _FakeTokenizer:
    """Character-level stand-in; both builders only need ids and pad lookups."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|vision_start|>": 1000, "<|image_pad|>": 1001, "<|video_pad|>": 1002, "<|vision_end|>": 1003}[token]


def _prepared_image():
    prepared = mlx_reference.MiniMaxH3PreparedReference(media_type="image")
    torch_prepared = torch_reference.MiniMaxH3PreparedReference(media_type="image")
    return torch_prepared, prepared


def _prepared_video():
    prepared = mlx_reference.MiniMaxH3PreparedReference(media_type="video", block_timestamps=[0.5, 1.5])
    torch_prepared = torch_reference.MiniMaxH3PreparedReference(media_type="video", block_timestamps=[0.5, 1.5])
    return torch_prepared, prepared


def test_presentation_uses_upstream_vision_tokens() -> None:
    """The picture label is text; only the vision block carries the video tag."""
    tokenizer = _FakeTokenizer()
    prepared = mlx_reference.MiniMaxH3PreparedReference(media_type="image")
    token_ids, token_tags = mlx_reference.build_ref2va_presentation(tokenizer, "hi", [prepared], [2], [])

    label = "<Picture 1>: "
    vision = [1000, 1001, 1001, 1003]
    prompt = [ord(character) for character in "hi"]
    assert token_ids == [ord(character) for character in label] + vision + prompt
    assert token_tags == [1] * len(label) + [0] * len(vision) + [1] * len(prompt)


@pytest.mark.parametrize("size", [(3000, 2000), (1080, 1920), (832, 480), (400, 300)])
@pytest.mark.parametrize("max_pixels", [832 * 480, 4 * 832 * 480])
def test_reference_image_cap_scales_down_only(size: tuple[int, int], max_pixels: int) -> None:
    """An opt-in pixel cap shrinks the trained canvas, keeps its aspect, and never upscales."""
    trained = mlx_reference.resolve_reference_image_size(*size)
    capped = mlx_reference.resolve_reference_image_size(*size, max_pixels=max_pixels)
    multiple = mlx_reference.MINIMAX_H3_CANVAS_MULTIPLE
    assert capped[0] % multiple == 0 and capped[1] % multiple == 0
    assert capped[0] <= trained[0] and capped[1] <= trained[1]
    if trained[0] * trained[1] <= max_pixels:
        assert capped == trained
    else:
        # Within one snapping step of the cap, aspect within one multiple.
        assert capped[0] * capped[1] <= max_pixels * 1.15
        assert abs(capped[1] / capped[0] - trained[1] / trained[0]) < multiple / min(capped)


def test_reference_video_canvas_cap_scales_down_only() -> None:
    trained = mlx_reference.resolve_canvas_size(1920, 1080)
    assert mlx_reference.resolve_canvas_size(1920, 1080, max_pixels=None) == trained
    capped = mlx_reference.resolve_canvas_size(1920, 1080, max_pixels=416 * 736)
    assert capped[0] * capped[1] < trained[0] * trained[1]
    assert mlx_reference.resolve_canvas_size(1920, 1080, max_pixels=10**9) == trained


def test_reference_cap_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="max_pixels"):
        mlx_reference.resolve_reference_image_size(800, 600, max_pixels=0)
