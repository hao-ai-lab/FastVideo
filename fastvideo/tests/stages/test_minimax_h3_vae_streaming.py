# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MiniMaxH3PackedLayout,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_decoding
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding import MiniMaxH3VideoDecodingStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
    MINIMAX_H3_LAYOUT_KEY,
    MiniMaxH3LatentPreparationStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def _layout(rows: int, latent_shape: tuple[int, ...]) -> MiniMaxH3PackedLayout:
    empty = torch.empty(0, dtype=torch.long)
    return MiniMaxH3PackedLayout(
        sequence_length=rows,
        position_ids=empty,
        token_tags=empty,
        video_indices=empty,
        audio_indices=empty,
        text_indices=empty,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_video_latent_frames=latent_shape[2],
        latent_height=latent_shape[3],
        latent_width=latent_shape[4],
        num_audio_latents=0,
    )


def test_reference_video_encode_keeps_pixels_on_cpu() -> None:
    observed = {}

    class VAE:

        def encode_pixels(self, pixels):
            observed["pixels"] = pixels
            posterior = SimpleNamespace(sample=lambda generator=None: torch.zeros(1, 4, 7, 4, 4))
            return SimpleNamespace(latent_dist=posterior)

        def normalize_latents(self, latents):
            return latents

    stage = MiniMaxH3LatentPreparationStage(
        vae=VAE(),
        audio_vae=None,
        scheduler=None,
        ref2va=True,
    )
    reference = MiniMaxH3PreparedReference(
        media_type="video",
        frames=np.zeros((22, 16, 16, 3), dtype=np.uint8),
    )
    args = SimpleNamespace(
        vae_parallel_encode=False,
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(patch_size=(1, 1, 1))),
    )
    rows = stage._encode_visual_rows([reference], torch.device("cpu"), args)

    assert observed["pixels"].dtype == torch.uint8
    assert observed["pixels"].device.type == "cpu"
    assert rows[0].shape == (7 * 4 * 4, 4)


def _decode_args(**overrides):
    args = dict(
        output_type="pil",
        pin_cpu_memory=False,
        vae_cpu_offload=False,
        vae_parallel_decode=False,
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(patch_size=(1, 1, 1))),
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class _FillingVAE:
    """Stands in for the video VAE: records its inputs and fills the pixel buffer."""

    def __init__(self, latent_shape: tuple[int, ...], pixel_shape: tuple[int, ...], fill: float) -> None:
        self.latent_shape = latent_shape
        self.pixel_shape = pixel_shape
        self.fill = fill
        self.observed: dict[str, torch.Tensor] = {}

    def to(self, device):
        return self

    def denormalize_latents(self, decoded_latents):
        return decoded_latents

    def decoded_pixel_shape(self, shape):
        assert tuple(shape) == self.latent_shape
        return self.pixel_shape

    def decode_to_pixels(self, decoded_latents, output):
        self.observed["latents"] = decoded_latents
        self.observed["output"] = output
        output.fill_(self.fill)


def _video_batch(rows: torch.Tensor, latent_shape: tuple[int, ...], **fields) -> ForwardBatch:
    batch = ForwardBatch(data_type="video", latents=rows, raw_latent_shape=latent_shape, **fields)
    batch.extra[MINIMAX_H3_LAYOUT_KEY] = _layout(rows.shape[0], latent_shape)
    return batch


def test_decode_stage_returns_uint8_frames_from_reused_pixel_buffer(monkeypatch) -> None:
    """The VAE streams fp32 pixels into a stage-owned buffer that is reused
    across requests; the caller receives quantized uint8 frames in a fresh
    shared-memory tensor (``save_video=False`` keeps the RGB frames)."""
    latent_shape = (1, 4, 2, 4, 4)
    pixel_shape = (1, 3, 5, 16, 16)
    latents = torch.randn(latent_shape)
    rows = patchify_video_latents(latents, (1, 1, 1))
    vae = _FillingVAE(latent_shape, pixel_shape, fill=0.2)  # 0.2 * 255 = 51
    stage = MiniMaxH3VideoDecodingStage(vae)
    monkeypatch.setattr(minimax_h3_decoding, "get_local_torch_device", lambda: torch.device("cpu"))

    result = stage.forward(_video_batch(rows, latent_shape, save_video=False), _decode_args())

    torch.testing.assert_close(vae.observed["latents"], latents)
    pixel_buffer = vae.observed["output"]
    assert pixel_buffer.dtype == torch.float32
    assert tuple(pixel_buffer.shape) == pixel_shape
    assert stage._pixel_buffer is pixel_buffer
    assert result.output.dtype == torch.uint8
    assert result.output.device.type == "cpu"
    assert tuple(result.output.shape) == pixel_shape
    assert result.output.is_shared()
    assert torch.all(result.output == 51)
    assert "frames_yuv420p" not in result.extra

    # A second request of the same geometry decodes into the same buffer.
    stage.forward(_video_batch(rows.clone(), latent_shape, save_video=False), _decode_args())
    assert vae.observed["output"] is pixel_buffer


def test_decode_stage_ships_yuv420p_frames_for_video_saving(monkeypatch) -> None:
    """With the default ``save_video`` the stage hands the driver planar
    yuv420p frames and only a shape-only stand-in for ``output``."""
    latent_shape = (1, 4, 2, 4, 4)
    pixel_shape = (1, 3, 5, 16, 16)
    rows = patchify_video_latents(torch.randn(latent_shape), (1, 1, 1))
    vae = _FillingVAE(latent_shape, pixel_shape, fill=0.2)
    monkeypatch.setattr(minimax_h3_decoding, "get_local_torch_device", lambda: torch.device("cpu"))

    result = MiniMaxH3VideoDecodingStage(vae).forward(_video_batch(rows, latent_shape), _decode_args())

    assert result.output.dtype == torch.uint8
    assert tuple(result.output.shape) == pixel_shape
    frames_yuv = result.extra["frames_yuv420p"]
    assert frames_yuv.dtype == torch.uint8
    assert frames_yuv.device.type == "cpu"
    assert frames_yuv.is_shared()
    assert tuple(frames_yuv.shape) == (5, 16 * 3 // 2, 16)
    expected = minimax_h3_decoding._rgb_u8_to_yuv420p(torch.full(pixel_shape[1:], 51, dtype=torch.uint8))
    assert torch.equal(frames_yuv, expected)


def test_decode_stages_skip_vae_on_non_output_rank(monkeypatch) -> None:
    class VAE:

        sampling_rate = 32000

        def to(self, device):
            raise AssertionError("non-output ranks must not execute a VAE")

    monkeypatch.setattr(minimax_h3_decoding, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(minimax_h3_decoding, "get_sp_group",
                        lambda: SimpleNamespace(is_first_rank=True, world_size=4, rank_in_group=0))
    monkeypatch.setattr(minimax_h3_decoding, "get_world_group", lambda: SimpleNamespace(is_first_rank=False))
    args = SimpleNamespace(output_type="pil", pin_cpu_memory=False, vae_cpu_offload=True, vae_parallel_decode=False)

    video = MiniMaxH3VideoDecodingStage(VAE()).forward(ForwardBatch(data_type="video"), args)
    assert video.output.shape == (0, 3, 0, 0, 0)

    audio_batch = ForwardBatch(data_type="audio", latents=torch.zeros(1), audio_latents=torch.zeros(1))
    audio_batch.extra[MINIMAX_H3_LAYOUT_KEY] = object()
    audio = minimax_h3_decoding.MiniMaxH3AudioDecodingStage(VAE()).forward(audio_batch, args)
    assert audio.extra["audio"].shape == (0, 2)
    assert audio.extra["audio_sample_rate"] == 32000
    assert audio.latents is None
    assert audio.audio_latents is None
    assert MINIMAX_H3_LAYOUT_KEY not in audio.extra


def test_parallel_decode_runs_on_every_rank(monkeypatch) -> None:
    """With vae_parallel_decode, non-leader ranks must enter the decode body
    (the collectives inside require uniform participation) and only the
    leader owns the CPU output buffer."""
    latent_shape = (1, 4, 2, 4, 4)
    rows = patchify_video_latents(torch.randn(latent_shape), (1, 1, 1))
    calls = []

    class VAE:

        def to(self, device):
            return self

        def denormalize_latents(self, decoded_latents):
            return decoded_latents

        def decoded_pixel_shape(self, shape):
            return (1, 3, 5, 16, 16)

    def fake_parallel(vae, latents, output, group, strategy):
        calls.append((group.rank_in_group, output, strategy))
        if output is not None:
            output.fill_(0.4)  # 0.4 * 255 = 102 once quantized to uint8
        return output

    monkeypatch.setattr(minimax_h3_decoding, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(minimax_h3_decoding, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(minimax_h3_decoding, "decode_to_pixels_parallel", fake_parallel)
    args = SimpleNamespace(
        output_type="pil",
        pin_cpu_memory=False,
        vae_cpu_offload=False,
        vae_parallel_decode=True,
        vae_parallel_decode_strategy="gather",
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(patch_size=(1, 1, 1))),
    )

    for rank, is_first in ((0, True), (2, False)):
        monkeypatch.setattr(
            minimax_h3_decoding, "get_sp_group",
            lambda rank=rank, is_first=is_first: SimpleNamespace(is_first_rank=is_first,
                                                                 world_size=4,
                                                                 rank_in_group=rank))
        batch = _video_batch(rows.clone(), latent_shape, save_video=False)
        result = MiniMaxH3VideoDecodingStage(VAE()).forward(batch, args)
        if is_first:
            assert result.output.dtype == torch.uint8
            assert result.output.shape == (1, 3, 5, 16, 16)
            assert torch.all(result.output == 102)
        else:
            assert result.output.shape == (0, 3, 0, 0, 0)

    assert [(rank, output is not None) for rank, output, _ in calls] == [(0, True), (2, False)]
    assert all(strategy == "gather" for _, _, strategy in calls)
