# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 video and stereo-audio decoding."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device, get_sp_group, get_world_group, model_parallel_is_initialized
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_parallel import DEFAULT_DECODE_GATHER_STRATEGY, decode_to_pixels_parallel
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.profiler import nvtx_range
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MiniMaxH3PackedLayout,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import is_pin_memory_available

logger = init_logger(__name__)


def _layout(batch: ForwardBatch) -> MiniMaxH3PackedLayout:
    layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
    if not isinstance(layout, MiniMaxH3PackedLayout):
        raise ValueError("MiniMax-H3 packed layout is missing at decode.")
    return layout


def _decode_participation(fastvideo_args: FastVideoArgs, want_parallel: bool) -> tuple[Any, bool, bool]:
    """Resolve (sp_group, is_output_rank, parallel) for the VAE decode stages.

    The existing serial path keeps its global-rank-zero output ownership.
    Parallel decode assembles once per sequence-parallel group, on that
    group's first rank. ``parallel`` is only true when every group rank will
    run the decode body — the collectives inside require uniform
    participation, so no rank-dependent branch may guard them.
    """
    if not model_parallel_is_initialized():
        return None, True, False
    sp_group = get_sp_group()
    if bool(want_parallel) and sp_group.world_size > 1:
        return sp_group, sp_group.is_first_rank, True
    return sp_group, get_world_group().is_first_rank, False


class MiniMaxH3VideoDecodingStage(PipelineStage):
    """Drop visual condition rows, unpatchify, and decode the target video."""

    performance_component_metric = "vae_decode_time_s"

    def __init__(self, vae: AutoencoderKLMiniMaxH3, transformer: Any) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.not_none)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.with_dims(5))
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Decode H3 video latents into normalized CPU pixels."""
        placeholder = torch.empty((0, 3, 0, 0, 0), device="cpu", dtype=torch.float32)
        sp_group, is_output_rank, parallel = _decode_participation(fastvideo_args, fastvideo_args.vae_parallel_decode)
        if not is_output_rank and not parallel:
            # Consumers read the output rank's ForwardBatch. Keep a
            # verifier-compatible placeholder on other ranks and avoid
            # duplicating the full VAE decode and CPU output buffer.
            batch.output = placeholder
            return batch

        layout = _layout(batch)
        if batch.latents is None or batch.raw_latent_shape is None or len(batch.raw_latent_shape) != 5:
            raise ValueError("MiniMax-H3 video latents or raw geometry are missing at decode.")
        _, channels, num_frames, latent_height, latent_width = batch.raw_latent_shape
        latents = unpatchify_video_tokens(
            batch.latents[layout.num_condition_video_rows:],
            num_frames,
            latent_height,
            latent_width,
            channels,
            self.transformer.patch_size,
        )
        device = get_local_torch_device()
        self.vae.to(device)
        try:
            latents = self.vae.denormalize_latents(latents.to(device=device, dtype=torch.float32))
            if fastvideo_args.output_type == "latent":
                # No collectives on this path, so uniform participation is
                # trivial: every rank returns here.
                batch.output = latents.detach().float().cpu() if is_output_rank else placeholder
                return batch

            output = None
            if is_output_rank:
                output = torch.empty(
                    self.vae.decoded_pixel_shape(latents.shape),
                    device="cpu",
                    dtype=torch.float32,
                    pin_memory=fastvideo_args.pin_cpu_memory and is_pin_memory_available(),
                )
            # Attribute the streamed decoder computation while retaining
            # per-chunk device-to-host transfer and pinned-buffer reuse.
            with (
                    nvtx_range("minimax_h3.vae"),
                    torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                if parallel:
                    strategy = fastvideo_args.vae_parallel_decode_strategy or DEFAULT_DECODE_GATHER_STRATEGY
                    logger.info("MiniMax-H3 VAE decode: sequence-parallel chunks across %d ranks (%s)",
                                sp_group.world_size, strategy)
                    decode_to_pixels_parallel(self.vae, latents, output, sp_group, strategy=strategy)
                else:
                    self.vae.decode_to_pixels(latents, output)
            batch.output = output if is_output_rank else placeholder
            return batch
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")


class MiniMaxH3AudioDecodingStage(PipelineStage):
    """Drop audio condition rows and decode the target stereo waveform."""

    performance_component_metric = "audio_decode_time_s"

    def __init__(self, audio_vae: MiniMaxH3AudioVAE) -> None:
        super().__init__()
        self.audio_vae = audio_vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio", batch.extra.get("audio"), V.is_tensor)
        result.add_check("audio_sample_rate", batch.extra.get("audio_sample_rate"), V.positive_int)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Decode H3 audio latents into a stereo CPU waveform."""
        # Audio decode is sub-second, so preserve the serial path's global
        # rank-zero ownership.
        if model_parallel_is_initialized() and not get_world_group().is_first_rank:
            batch.extra["audio"] = torch.empty((0, 2), device="cpu", dtype=torch.float32)
            batch.extra["audio_sample_rate"] = self.audio_vae.sampling_rate
            self._clear_runtime(batch)
            return batch

        layout = _layout(batch)
        if batch.audio_latents is None:
            raise ValueError("MiniMax-H3 audio latents are missing at decode.")
        latents = unpack_audio_tokens(
            batch.audio_latents[layout.num_condition_audio_rows:],
            layout.num_audio_latents,
        )
        device = get_local_torch_device()
        self.audio_vae.to(device)
        try:
            latents = self.audio_vae.denormalize_latents(latents.to(device=device, dtype=torch.float32))
            if fastvideo_args.output_type == "latent":
                batch.extra["audio"] = latents.detach().float().cpu()
                batch.extra["audio_sample_rate"] = self.audio_vae.sampling_rate
                self._clear_runtime(batch)
                return batch

            # The range isolates waveform synthesis from packing and runtime
            # cleanup so the audio decoder has one stable timeline boundary.
            with nvtx_range("minimax_h3.audio_vae"):
                decoded = self.audio_vae.decode(latents).sample.float()
            if decoded.ndim != 3 or decoded.shape[0] != 2 or decoded.shape[1] != 1:
                raise ValueError("MiniMax-H3 audio VAE must decode stereo channels as two mono batch items; "
                                 f"got {tuple(decoded.shape)}.")
            batch.extra["audio"] = decoded[:, 0].transpose(0, 1).contiguous().cpu()
            batch.extra["audio_sample_rate"] = self.audio_vae.sampling_rate
            self._clear_runtime(batch)
            return batch
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.audio_vae.to("cpu")

    @staticmethod
    def _clear_runtime(batch: ForwardBatch) -> None:
        batch.prompt_embeds = []
        batch.latents = None
        batch.audio_latents = None
        batch.references = None
        for key in tuple(batch.extra):
            if key.startswith("minimax_h3_"):
                batch.extra.pop(key)


__all__ = ["MiniMaxH3AudioDecodingStage", "MiniMaxH3VideoDecodingStage"]
