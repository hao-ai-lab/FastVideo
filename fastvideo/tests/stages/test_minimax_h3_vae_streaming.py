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
        transformer=SimpleNamespace(patch_size=(1, 1, 1)),
        vae=VAE(),
        audio_vae=None,
        scheduler=None,
        ref2va=True,
    )
    reference = MiniMaxH3PreparedReference(
        media_type="video",
        frames=np.zeros((22, 16, 16, 3), dtype=np.uint8),
    )
    rows = stage._encode_visual_rows([reference], torch.device("cpu"))

    assert observed["pixels"].dtype == torch.uint8
    assert observed["pixels"].device.type == "cpu"
    assert rows[0].shape == (7 * 4 * 4, 4)


def test_decode_stage_uses_cpu_output_buffer(monkeypatch) -> None:
    latent_shape = (1, 4, 2, 4, 4)
    latents = torch.randn(latent_shape)
    rows = patchify_video_latents(latents, (1, 1, 1))
    batch = ForwardBatch(data_type="video", latents=rows, raw_latent_shape=latent_shape)
    batch.extra[MINIMAX_H3_LAYOUT_KEY] = _layout(rows.shape[0], latent_shape)
    observed = {}

    class VAE:

        def to(self, device):
            return self

        def denormalize_latents(self, decoded_latents):
            return decoded_latents

        def decoded_pixel_shape(self, shape):
            assert tuple(shape) == latent_shape
            return (1, 3, 5, 16, 16)

        def decode_to_pixels(self, decoded_latents, output):
            observed["latents"] = decoded_latents
            observed["output"] = output
            output.fill_(0.25)

    monkeypatch.setattr(minimax_h3_decoding, "get_local_torch_device", lambda: torch.device("cpu"))
    result = MiniMaxH3VideoDecodingStage(VAE(), SimpleNamespace(patch_size=(1, 1, 1))).forward(
        batch,
        SimpleNamespace(output_type="pil", pin_cpu_memory=False, vae_cpu_offload=False),
    )

    torch.testing.assert_close(observed["latents"], latents)
    assert observed["output"] is result.output
    assert result.output.device.type == "cpu"
    assert torch.all(result.output == 0.25)


def test_decode_stages_skip_vae_on_non_output_rank(monkeypatch) -> None:
    class VAE:

        sampling_rate = 32000

        def to(self, device):
            raise AssertionError("non-output ranks must not execute a VAE")

    monkeypatch.setattr(minimax_h3_decoding, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(minimax_h3_decoding, "get_world_group", lambda: SimpleNamespace(is_first_rank=False))
    args = SimpleNamespace(output_type="pil", pin_cpu_memory=False, vae_cpu_offload=True)

    video = MiniMaxH3VideoDecodingStage(VAE(), SimpleNamespace()).forward(ForwardBatch(data_type="video"), args)
    assert video.output.shape == (0, 3, 0, 0, 0)

    audio_batch = ForwardBatch(data_type="audio", latents=torch.zeros(1), audio_latents=torch.zeros(1))
    audio_batch.extra[MINIMAX_H3_LAYOUT_KEY] = object()
    audio = minimax_h3_decoding.MiniMaxH3AudioDecodingStage(VAE()).forward(audio_batch, args)
    assert audio.extra["audio"].shape == (0, 2)
    assert audio.extra["audio_sample_rate"] == 32000
    assert audio.latents is None
    assert audio.audio_latents is None
    assert MINIMAX_H3_LAYOUT_KEY not in audio.extra
