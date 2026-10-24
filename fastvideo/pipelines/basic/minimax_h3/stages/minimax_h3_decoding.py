# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 video and stereo-audio decoding."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import torch

from fastvideo.distributed import get_local_torch_device, get_sp_group, model_parallel_is_initialized
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models import pinned_offload
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_parallel import DEFAULT_DECODE_GATHER_STRATEGY, decode_to_pixels_parallel
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.pipelines.basic.minimax_h3.encoder_split import h3_is_output_worker, h3_output_worker_rank
from fastvideo.profiler import nvtx_range
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MiniMaxH3PackedLayout,
    h3_dit_patch_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import current_platform
from fastvideo.utils import PRECISION_TO_TYPE, allocate_cpu_tensor_with_pin_fallback

logger = init_logger(__name__)

_IDLE_AUDIO_KEY = "minimax_h3_idle_audio"


def _layout(batch: ForwardBatch) -> MiniMaxH3PackedLayout:
    layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
    if not isinstance(layout, MiniMaxH3PackedLayout):
        raise ValueError("MiniMax-H3 packed layout is missing at decode.")
    return layout


def _decode_participation(fastvideo_args: FastVideoArgs, want_parallel: bool) -> tuple[Any, bool, bool]:
    """Resolve (sp_group, is_output_rank, parallel) for the VAE decode stages.

    The output rank is ``h3_output_worker_rank``: global rank zero, or the
    first denoise rank under the MiniMax-H3 encoder split. The serial path
    decodes on that rank alone. Parallel decode runs only in the
    sequence-parallel group containing the output rank and assembles on that
    rank. Every rank in any other SP group skips the decode uniformly.
    ``parallel`` is only true when every rank in the active group will run the
    decode body, because its collectives require group-uniform participation.
    """
    if not model_parallel_is_initialized():
        return None, True, False
    sp_group = get_sp_group()
    is_output_rank = h3_is_output_worker(fastvideo_args)
    if bool(want_parallel) and sp_group.world_size > 1:
        return sp_group, is_output_rank, h3_output_worker_rank(fastvideo_args) in sp_group.ranks
    return sp_group, is_output_rank, False


class MiniMaxH3VideoDecodingStage(PipelineStage):
    """Drop visual condition rows, unpatchify, and decode the target video."""

    performance_component_metric = "vae_decode_time_s"

    def __init__(self,
                 vae: AutoencoderKLMiniMaxH3 | None,
                 audio_stage: MiniMaxH3AudioDecodingStage | None = None) -> None:
        super().__init__()
        self.vae = vae
        self.audio_stage = audio_stage
        self._output_buffer: tuple[tuple[tuple[int, ...], bool], torch.Tensor] | None = None
        self._unified_memory: bool | None = None

    def _pixel_buffer(self, shape: tuple[int, ...], pin_memory: bool, device: torch.device) -> torch.Tensor:
        """The CPU FP32 pixel buffer, reused across requests of the same geometry on unified memory.

        The stage returns a uint8 copy, so the buffer never escapes. Reuse matters on
        unified-memory GPUs (DGX Spark): device-to-host copies into never-touched pageable
        pages fault through the driver at ~150 MB/s (+3-7 s per 480p-768p clip whenever the
        allocator hands back fresh pages), while populated pages copy at full speed.
        ``zero_`` populates a new buffer from the CPU once. Discrete GPUs gain nothing from
        holding up to 4.27 GB of host memory between requests, so they allocate per request.
        """
        if self._unified_memory is None:
            self._unified_memory = current_platform.has_unified_memory(device.index or 0)
        if not self._unified_memory:
            return allocate_cpu_tensor_with_pin_fallback(shape, dtype=torch.float32, pin_memory=pin_memory)
        key = (tuple(shape), bool(pin_memory))
        if self._output_buffer is None or self._output_buffer[0] != key:
            self._output_buffer = None
            buffer = allocate_cpu_tensor_with_pin_fallback(shape, dtype=torch.float32, pin_memory=pin_memory)
            if not buffer.is_pinned():
                buffer.zero_()
            self._output_buffer = (key, buffer)
        return self._output_buffer[1]

    def _idle_audio_callback(self, batch: ForwardBatch, args: FastVideoArgs, group: Any,
                             device: torch.device) -> Callable[[], None] | None:
        """Use the last idle slot of an odd two-rank chunk plan for audio.

        The result lives only on this request. The audio stage transfers it
        after video assembly, keeping its normal output ownership and cleanup.
        """
        audio_stage, vae = self.audio_stage, self.vae
        if audio_stage is None or vae is None or group.world_size != 2 or batch.raw_latent_shape[0] != 1:
            return None
        _, chunks, _ = vae._temporal_decode_plan(batch.raw_latent_shape[2])
        if chunks <= 1 or chunks % 2 == 0:
            return None
        batch.extra[_IDLE_AUDIO_KEY] = None
        if group.rank_in_group != 1:
            return None
        # Capture the outer stage context before video enables its own autocast.
        enabled, dtype = torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type)

        def decode_audio() -> None:
            with torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
                batch.extra[_IDLE_AUDIO_KEY] = audio_stage.decode_audio(batch, args)

        return decode_audio

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
        batch.extra.pop(_IDLE_AUDIO_KEY, None)
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
            h3_dit_patch_size(fastvideo_args),
        )
        device = get_local_torch_device()
        backend = getattr(fastvideo_args, "video_decode_backend", "h3-vae")
        if backend == "taeh3":
            from fastvideo.models.vaes.minimax_h3_taeh3 import decode_ncthw_latents_taeh3, taeh3_decoded_pixel_shape

            if fastvideo_args.output_type == "latent":
                batch.output = latents.detach().float().cpu() if is_output_rank else placeholder
                return batch
            expected = taeh3_decoded_pixel_shape(tuple(latents.shape))
            logger.info("MiniMax-H3 video decode: TAEH3 preview (%s -> %s)", tuple(latents.shape), expected)
            decode_precision = (fastvideo_args.pipeline_config.vae_decode_precision
                                or fastvideo_args.pipeline_config.vae_precision)
            with nvtx_range("minimax_h3.taeh3"):
                pixels = decode_ncthw_latents_taeh3(
                    latents,
                    device=device,
                    checkpoint_path=getattr(fastvideo_args, "taeh3_checkpoint", None),
                    chunk_size=int(getattr(fastvideo_args, "taeh3_chunk_size", 5) or 5),
                    dtype=PRECISION_TO_TYPE[decode_precision],
                )
            batch.output = pixels.float().cpu() if is_output_rank else placeholder
            if is_output_rank and tuple(batch.output.shape) != expected:
                raise RuntimeError(f"TAEH3 wrote {tuple(batch.output.shape)}, expected {expected}.")
            return batch

        if self.vae is None:
            raise RuntimeError("MiniMax-H3 full VAE decode requires a loaded video VAE.")
        pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            # An NVFP4 decoder swapped in on the CPU could not pack its linears there;
            # freeze them now that the VAE is on the device, on the first decode.
            from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import freeze_nvfp4_linears_on_device
            if freeze_nvfp4_linears_on_device(self.vae):
                pinned_offload.refresh_host_copies(self.vae, pin=fastvideo_args.pin_cpu_memory)
            latents = self.vae.denormalize_latents(latents.to(device=device, dtype=torch.float32))
            if fastvideo_args.output_type == "latent":
                # No collectives on this path, so uniform participation is
                # trivial: every rank returns here.
                batch.output = latents.detach().float().cpu() if is_output_rank else placeholder
                return batch

            output = None
            if is_output_rank:
                output = self._pixel_buffer(tuple(self.vae.decoded_pixel_shape(latents.shape)),
                                            fastvideo_args.pin_cpu_memory, device)
            # Attribute the streamed decoder computation while retaining
            # per-chunk device-to-host transfer and pinned-buffer reuse.
            # fp16 for the release decoders; NVFP4 decoders were trained under bf16.
            idle_callback = self._idle_audio_callback(batch, fastvideo_args, sp_group, device) if parallel else None
            autocast_dtype = getattr(self.vae, "decode_autocast_dtype", torch.float16)
            with (
                    nvtx_range("minimax_h3.vae"),
                    torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"),
            ):
                if parallel:
                    strategy = fastvideo_args.vae_parallel_decode_strategy or DEFAULT_DECODE_GATHER_STRATEGY
                    logger.info("MiniMax-H3 VAE decode: sequence-parallel chunks across %d ranks (%s)",
                                sp_group.world_size, strategy)
                    if idle_callback is None:
                        decode_to_pixels_parallel(self.vae, latents, output, sp_group, strategy=strategy)
                    else:
                        decode_to_pixels_parallel(self.vae,
                                                  latents,
                                                  output,
                                                  sp_group,
                                                  strategy=strategy,
                                                  idle_callback=idle_callback)
                else:
                    self.vae.decode_to_pixels(latents, output)
            if is_output_rank and output is not None and output.dtype == torch.float32:
                # Quantize to uint8 here, before the frames cross the
                # multiprocess-executor boundary. A 345-frame 1344x768 clip is
                # 4.27 GB as fp32 and 1.07 GB as uint8, and the main process
                # cast it to uint8 immediately anyway; the clamp matches the
                # post-decode handling of VAE output slightly outside [0, 1].
                output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
            batch.output = output if is_output_rank else placeholder
            return batch
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.vae)


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
        produced = None
        if _IDLE_AUDIO_KEY in batch.extra:
            group = get_sp_group()
            produced = group.broadcast_object(batch.extra.pop(_IDLE_AUDIO_KEY), src=1)
        is_output = not model_parallel_is_initialized() or h3_is_output_worker(fastvideo_args)
        if not is_output:
            batch.extra["audio"] = torch.empty((0, 2), device="cpu", dtype=torch.float32)
            batch.extra["audio_sample_rate"] = self.audio_vae.sampling_rate
        else:
            # A tile-parallel decoder may not use a temporal idle slot. All
            # ranks observe None in that case and retain the serial audio path.
            waveform, sample_rate = produced if produced is not None else self.decode_audio(batch, fastvideo_args)
            batch.extra["audio"] = waveform
            batch.extra["audio_sample_rate"] = sample_rate
        self._clear_runtime(batch)
        return batch

    @torch.no_grad()
    def decode_audio(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> tuple[torch.Tensor, int]:
        """Synthesize this request without changing its latents or stage state."""
        layout = _layout(batch)
        if batch.audio_latents is None:
            raise ValueError("MiniMax-H3 audio latents are missing at decode.")
        latents = unpack_audio_tokens(
            batch.audio_latents[layout.num_condition_audio_rows:],
            layout.num_audio_latents,
        )
        device = get_local_torch_device()
        pinned_offload.load(self.audio_vae, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            latents = self.audio_vae.denormalize_latents(latents.to(device=device, dtype=torch.float32))
            if fastvideo_args.output_type == "latent":
                return latents.detach().float().cpu(), self.audio_vae.sampling_rate

            # The range isolates waveform synthesis from packing and runtime
            # cleanup so the audio decoder has one stable timeline boundary.
            with nvtx_range("minimax_h3.audio_vae"):
                decoded = self.audio_vae.decode(latents).sample.float()
            if decoded.ndim != 3 or decoded.shape[0] != 2 or decoded.shape[1] != 1:
                raise ValueError("MiniMax-H3 audio VAE must decode stereo channels as two mono batch items; "
                                 f"got {tuple(decoded.shape)}.")
            return decoded[:, 0].transpose(0, 1).contiguous().cpu(), self.audio_vae.sampling_rate
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.audio_vae)

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
