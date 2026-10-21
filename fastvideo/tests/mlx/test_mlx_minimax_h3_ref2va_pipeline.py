# SPDX-License-Identifier: Apache-2.0
"""Ref2VA plumbing in the MLX H3 pipeline, against a recording fake DiT.

The reference path changes what the denoiser is handed and what it returns:
the encoder rows ride in every forward while only the target rows are stepped.
A fake DiT records what it saw and returns a constant velocity, which is enough
to pin all of that without any weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 Ref2VA tests")

from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3SchedulerState,
    build_ref2va_packed_layout,
)
from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_reference import (  # noqa: E402
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
)

LATENT_CHANNELS = 4
PATCH_SIZE = (1, 2, 2)
PATCH_DIM = LATENT_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
AUDIO_IN_CHANNELS = 2
NUM_TEXT_TOKENS = 5
NUM_FRAMES = 22  # a valid H3 bucket: 17 * 1 + 5, below the 5 s floor on its own
HEIGHT, WIDTH = 64, 96
LATENT_HEIGHT, LATENT_WIDTH = HEIGHT // 16, WIDTH // 16


def _geometry() -> tuple[int, int, int, int]:
    """The latent geometry the pipeline resolves for the test's request."""
    from fastvideo.mlx_runtime.minimax_h3 import align_num_frames, audio_latent_num_frames, video_latent_num_frames

    aligned = align_num_frames(NUM_FRAMES)
    return video_latent_num_frames(aligned), LATENT_HEIGHT, LATENT_WIDTH, audio_latent_num_frames(aligned)


def _expected_layout():
    num_latent_frames, latent_height, latent_width, audio_latents = _geometry()
    return build_ref2va_packed_layout(
        np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
        _references(),
        num_latent_frames,
        latent_height,
        latent_width,
        audio_latents,
        PATCH_SIZE,
    )


class _RecordingDiT:
    """Records every forward and returns a constant, distinguishable velocity."""

    patch_size = PATCH_SIZE
    patch_dim = PATCH_DIM
    audio_in_channels = AUDIO_IN_CHANNELS
    vsa_config = None
    _adaln_cache = None
    last_vsa_stats = None

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def precompute_adaln(self, timesteps, *, drop_weights: bool = True):
        self.adaln_timesteps = np.asarray(timesteps)
        return None

    def forward_with_cache(self, x_v, x_a, text, *, layout, step_timesteps, row_timestep_inverse, step_index=0):
        import mlx.core as mx

        self.calls.append({
            "video_rows": np.asarray(x_v),
            "audio_rows": np.asarray(x_a),
            "step_timesteps": np.asarray(step_timesteps),
            "text_rows": int(np.asarray(text).shape[0]),
        })
        return x_v * 0.0 + 1.0, x_a * 0.0 + 1.0


def _pipeline(tmp_path: Path) -> MiniMaxH3MLXPipeline:
    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    pipeline.model_root = tmp_path
    pipeline.dit_checkpoint = tmp_path
    pipeline.resident = False
    pipeline.video_decode_backend = "h3-vae"
    pipeline.last_dit_forward_s = 0.0
    pipeline.last_vsa_stats = None
    pipeline.vae_dtype = "fp32"
    return pipeline


def _references() -> list[MiniMaxH3PreparedReference]:
    from PIL import Image

    image = MiniMaxH3PreparedReference(media_type="image",
                                       image=Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)))
    image.num_latent_frames, image.latent_height, image.latent_width = 1, 4, 4
    video = MiniMaxH3PreparedReference(media_type="video",
                                       has_audio=True,
                                       frames=np.zeros((NUM_FRAMES, 4, 4, 3), dtype=np.uint8),
                                       waveform=np.zeros((2, 8), dtype=np.float32),
                                       block_timestamps=[0.5, 1.5])
    video.num_latent_frames, video.latent_height, video.latent_width = 2, 4, 4
    video.num_audio_latents = 3
    return [image, video]


class _FakeVideoVAE:
    """Encodes to the latent geometry the fixtures' references already declare."""

    latent_channels = LATENT_CHANNELS

    @staticmethod
    def normalize_pixels(pixels):
        return pixels

    @staticmethod
    def normalize_latents(latents):
        return latents

    @staticmethod
    def sample_posterior(mean, logvar, noise):
        return mean

    def _moments(self, pixels):
        import mlx.core as mx

        frames = 2 if pixels.shape[2] > 1 else 1
        shape = (1, self.latent_channels, frames, 4, 4)
        return mx.zeros(shape), mx.zeros(shape)

    def encode_keyframe(self, pixels):
        return self._moments(pixels)

    def encode(self, pixels):
        return self._moments(pixels)


class _FakeAudioVAE:
    latent_channels = MINIMAX_H3_AUDIO_CHANNELS

    @staticmethod
    def normalize_latents(latents):
        return latents

    @staticmethod
    def encode(waveform):
        # One posterior per stereo channel, as the CUDA path's audio VAE does.
        import mlx.core as mx

        shape = (waveform.shape[0], _FakeAudioVAE.latent_channels, 3)
        return mx.zeros(shape), mx.zeros(shape)


@pytest.fixture
def fake_vaes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_video_vae.mlx_h3_video_vae_from_dir",
                        lambda *args, **kwargs: _FakeVideoVAE())
    monkeypatch.setattr("fastvideo.mlx_runtime.minimax_h3_audio_vae.mlx_h3_audio_vae_from_dir",
                        lambda *args, **kwargs: _FakeAudioVAE())


def test_denoise_hands_the_encoder_rows_to_every_forward(tmp_path: Path) -> None:
    """The forward sees condition+target rows; only target rows are returned."""
    pipeline = _pipeline(tmp_path)
    references = _references()
    patch_size = PATCH_SIZE
    layout = _expected_layout()
    video_conditions = np.full((layout.num_condition_video_rows, PATCH_DIM), 3.0, dtype=np.float32)
    audio_conditions = np.full((layout.num_condition_audio_rows, AUDIO_IN_CHANNELS), 4.0, dtype=np.float32)
    dit = _RecordingDiT()

    video_rows, audio_rows = pipeline.denoise(
        np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
        np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        audio_num_frames=NUM_FRAMES,
        seed=0,
        num_steps=2,
        dit=dit,
        references=references,
        condition_video_rows=video_conditions,
        condition_audio_rows=audio_conditions,
    )

    assert len(dit.calls) == 2
    for call in dit.calls:
        assert call["video_rows"].shape == (layout.video_indices.shape[0], PATCH_DIM)
        assert call["audio_rows"].shape == (layout.audio_indices.shape[0], AUDIO_IN_CHANNELS)
        # The encoder rows are passed through untouched on every step.
        np.testing.assert_allclose(call["video_rows"][:layout.num_condition_video_rows], 3.0)
        np.testing.assert_allclose(call["audio_rows"][:layout.num_condition_audio_rows], 4.0)
        assert call["text_rows"] == NUM_TEXT_TOKENS

    # Only the target rows come back, at their own geometry.
    assert video_rows.shape == (layout.video_indices.shape[0] - layout.num_condition_video_rows, PATCH_DIM)
    assert audio_rows.shape == (layout.audio_indices.shape[0] - layout.num_condition_audio_rows, AUDIO_IN_CHANNELS)


def test_denoise_schedules_the_condition_timestep(tmp_path: Path, fake_vaes: None) -> None:
    """The 0.999 condition timestep is in the AdaLN ladder the cache is built for."""
    pipeline = _pipeline(tmp_path)
    dit = _RecordingDiT()
    pipeline.denoise(
        np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
        np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        audio_num_frames=NUM_FRAMES,
        seed=0,
        num_steps=2,
        dit=dit,
        references=_references(),
        condition_video_rows=np.zeros((_condition_rows(), PATCH_DIM), dtype=np.float32),
        condition_audio_rows=np.zeros((_expected_layout().num_condition_audio_rows, AUDIO_IN_CHANNELS),
                                      dtype=np.float32),
    )
    assert np.isclose(dit.adaln_timesteps, MINIMAX_H3_KEYFRAME_NOISE_AUG).any()
    # Every forward's timestep set must be resolvable in that ladder.
    for call in dit.calls:
        assert set(np.round(call["step_timesteps"], 6)) <= set(np.round(dit.adaln_timesteps, 6))


def test_denoise_rejects_fast_mode_temporal_stretch(tmp_path: Path) -> None:
    """A stretched ladder would leave the target rows on the unstretched clock."""
    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    with pytest.raises(ValueError, match="no temporal-scale stretch"):
        pipeline.denoise(
            np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
            np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            audio_num_frames=NUM_FRAMES,
            video_temporal_scale=1.7,
            seed=0,
            num_steps=2,
            dit=_RecordingDiT(),
            references=_references(),
            condition_video_rows=np.zeros((layout.num_condition_video_rows, PATCH_DIM), dtype=np.float32),
            condition_audio_rows=np.zeros((layout.num_condition_audio_rows, AUDIO_IN_CHANNELS), dtype=np.float32),
        )


def test_denoise_encodes_condition_rows_when_absent(tmp_path: Path, fake_vaes: None) -> None:
    """With half the rows given, the pipeline encodes the missing half itself."""
    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    dit = _RecordingDiT()
    pipeline.denoise(
        np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
        np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        audio_num_frames=NUM_FRAMES,
        seed=0,
        num_steps=2,
        dit=dit,
        references=_references(),
        condition_video_rows=np.full((layout.num_condition_video_rows, PATCH_DIM), 3.0, dtype=np.float32),
    )
    assert dit.calls
    for call in dit.calls:
        assert call["audio_rows"].shape[0] == layout.audio_indices.shape[0]


def test_denoise_rejects_mismatched_video_condition_rows(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    with pytest.raises(ValueError, match="Video condition rows"):
        pipeline.denoise(
            np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
            np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            audio_num_frames=NUM_FRAMES,
            seed=0,
            num_steps=2,
            dit=_RecordingDiT(),
            references=_references(),
            condition_video_rows=np.zeros((layout.num_condition_video_rows + 1, PATCH_DIM), dtype=np.float32),
            condition_audio_rows=np.zeros((layout.num_condition_audio_rows, AUDIO_IN_CHANNELS), dtype=np.float32),
        )


def test_denoise_rejects_mismatched_audio_condition_rows(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    with pytest.raises(ValueError, match="Audio condition rows"):
        pipeline.denoise(
            np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
            np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            audio_num_frames=NUM_FRAMES,
            seed=0,
            num_steps=2,
            dit=_RecordingDiT(),
            references=_references(),
            condition_video_rows=np.zeros((layout.num_condition_video_rows, PATCH_DIM), dtype=np.float32),
            condition_audio_rows=np.zeros((layout.num_condition_audio_rows + 1, AUDIO_IN_CHANNELS),
                                          dtype=np.float32),
        )


def test_prepare_references_decodes_and_validates(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    (tmp_path / "audio_vae").mkdir()
    (tmp_path / "audio_vae" / "config.json").write_text('{"sampling_rate": 32000}')
    pixels = np.random.default_rng(3).integers(0, 256, size=(40, 28, 3), dtype=np.uint8)

    prepared = pipeline.prepare_references([MiniMaxH3Reference(source=pixels, media_type="image")], NUM_FRAMES)

    assert len(prepared) == 1
    assert prepared[0].media_type == "image"
    assert prepared[0].image is not None
    with pytest.raises(ValueError, match="at least one reference"):
        pipeline.prepare_references([], NUM_FRAMES)


def test_audio_sample_rate_reads_the_audio_vae(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline.audio_sample_rate() == 32000
    (tmp_path / "audio_vae").mkdir()
    (tmp_path / "audio_vae" / "config.json").write_text('{"sampling_rate": 44100}')
    assert pipeline.audio_sample_rate() == 44100


def test_target_rows_step_alongside_pinned_encoder_rows(tmp_path: Path) -> None:
    """A constant velocity moves target rows and leaves encoder rows where they are."""
    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    dit = _RecordingDiT()
    pipeline.denoise(
        np.zeros((NUM_TEXT_TOKENS, 16), dtype=np.float32),
        np.full(NUM_TEXT_TOKENS, 1, dtype=np.int64),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        audio_num_frames=NUM_FRAMES,
        seed=0,
        num_steps=3,
        dit=dit,
        references=_references(),
        condition_video_rows=np.full((layout.num_condition_video_rows, PATCH_DIM), 3.0, dtype=np.float32),
        condition_audio_rows=np.full((layout.num_condition_audio_rows, AUDIO_IN_CHANNELS), 4.0, dtype=np.float32),
    )
    # A constant unit velocity is an Euler move toward the target each step, so
    # the encoder rows stay exactly at their conditioning values while the
    # target rows change.
    for call in dit.calls:
        np.testing.assert_allclose(call["video_rows"][:layout.num_condition_video_rows], 3.0, atol=1e-6)
        np.testing.assert_allclose(call["audio_rows"][:layout.num_condition_audio_rows], 4.0, atol=1e-6)
    first, last = dit.calls[0]["video_rows"], dit.calls[-1]["video_rows"]
    assert not np.allclose(first[layout.num_condition_video_rows:], last[layout.num_condition_video_rows:])


def _condition_rows() -> int:
    return _expected_layout().num_condition_video_rows


def test_reference_encode_skips_the_base_decoder(tmp_path, monkeypatch) -> None:
    from fastvideo.mlx_runtime import minimax_h3_video_vae as video_vae
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    seen = {}
    monkeypatch.setattr(video_vae, "mlx_h3_video_vae_from_dir", lambda _dir, **kwargs: seen.update(kwargs) or "vae")
    pipeline = object.__new__(MiniMaxH3MLXPipeline)
    pipeline.model_root = tmp_path
    pipeline.vae_dtype = "bf16"
    assert pipeline._load_video_vae(include_encoder=True, include_decoder=False) == "vae"
    assert seen["include_decoder"] is False and seen["include_encoder"] is True
