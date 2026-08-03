# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 video and stereo-audio decoding stages."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.pipelines.basic.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.types import MINIMAX_H3_STATE_KEY, get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _arch_value(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None:
        config = getattr(module, "config", None)
        arch = getattr(config, "arch_config", config)
        value = getattr(arch, name, None)
    if value is None:
        raise ValueError(f"MiniMax-H3 component {type(module).__name__} does not expose `{name}`.")
    return value


class MiniMaxH3VideoDecodingStage(PipelineStage):
    """Drop condition rows, unpatchify, and reverse H3 latent/pixel normalization."""

    performance_component_metric = "vae_decode_time_s"

    def __init__(self, vae: AutoencoderKLMiniMaxH3, transformer: Any) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        result.add_check("video_latents", state.video_latents, V.with_dims(2))
        result.add_check("num_latent_frames", state.num_latent_frames, V.positive_int)
        result.add_check("latent_height", state.latent_height, V.positive_int)
        result.add_check("latent_width", state.latent_width, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.with_dims(5))
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        if state.layout is None or state.video_latents is None:
            raise ValueError("MiniMax-H3 video latents are missing at decode.")
        geometry = (state.num_latent_frames, state.latent_height, state.latent_width)
        if any(value is None for value in geometry):
            raise ValueError("MiniMax-H3 video geometry is incomplete at decode.")
        num_frames, latent_height, latent_width = (int(value) for value in geometry if value is not None)
        patch_size = tuple(int(value) for value in _arch_value(self.transformer, "patch_size"))
        latent_channels = int(_arch_value(self.vae, "latent_channels"))

        latents = unpatchify_video_tokens(
            state.video_latents[state.layout.num_condition_video_rows:],
            num_frames,
            latent_height,
            latent_width,
            latent_channels,
            patch_size,
        )
        self.vae, device, _ = move_module_to_local_device(self.vae)
        try:
            latents = self.vae.denormalize_latents(latents.to(device=device, dtype=torch.float32))
            if fastvideo_args.output_type == "latent":
                batch.output = latents.detach().float().cpu()
                return batch

            video = self.vae.decode(latents).sample
            batch.output = self.vae.denormalize_pixels(video.float()).clamp_(0, 1).cpu()
            return batch
        finally:
            self.vae = maybe_offload_module(
                self.vae,
                enabled=bool(getattr(fastvideo_args, "vae_cpu_offload", False)),
            )


class MiniMaxH3AudioDecodingStage(PipelineStage):
    """Decode channel-major rows and expose FastVideo's joint-AV output keys."""

    performance_component_metric = "audio_decode_time_s"

    def __init__(self, audio_vae: MiniMaxH3AudioVAE) -> None:
        super().__init__()
        self.audio_vae = audio_vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        result.add_check("audio_latents", state.audio_latents, V.with_dims(2))
        result.add_check("num_audio_latents", state.num_audio_latents, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio", batch.extra.get("audio"), V.is_tensor)
        result.add_check("audio_sample_rate", batch.extra.get("audio_sample_rate"), V.positive_int)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        if state.layout is None or state.audio_latents is None or state.num_audio_latents is None:
            raise ValueError("MiniMax-H3 audio latents are missing at decode.")
        latent_channels = int(_arch_value(self.audio_vae, "latent_channels"))
        latents = unpack_audio_tokens(
            state.audio_latents[state.layout.num_condition_audio_rows:],
            state.num_audio_latents,
        )
        mean = torch.as_tensor(_arch_value(self.audio_vae, "latents_mean"), dtype=torch.float32).view(1, -1, 1)
        std = torch.as_tensor(_arch_value(self.audio_vae, "latents_std"), dtype=torch.float32).view(1, -1, 1)
        if mean.shape[1] != latent_channels or std.shape[1] != latent_channels:
            raise ValueError("MiniMax-H3 audio latent statistics do not match the audio VAE channels.")

        self.audio_vae, device, _ = move_module_to_local_device(self.audio_vae)
        try:
            latents = latents.to(device=device, dtype=torch.float32) * std.to(device) + mean.to(device)
            if fastvideo_args.output_type == "latent":
                batch.extra["audio"] = latents.detach().float().cpu()
                batch.extra["audio_sample_rate"] = int(_arch_value(self.audio_vae, "sampling_rate"))
                batch.prompt_embeds = []
                batch.extra.pop(MINIMAX_H3_STATE_KEY, None)
                return batch

            decoded = self.audio_vae.decode(latents).sample.float()
            if decoded.ndim != 3 or decoded.shape[0] != 2 or decoded.shape[1] != 1:
                raise ValueError("MiniMax-H3 audio VAE must decode stereo channels as two mono batch items; "
                                 f"got {tuple(decoded.shape)}.")
            batch.extra["audio"] = decoded[:, 0].transpose(0, 1).contiguous().cpu()
            batch.extra["audio_sample_rate"] = int(_arch_value(self.audio_vae, "sampling_rate"))
            batch.prompt_embeds = []
            batch.extra.pop(MINIMAX_H3_STATE_KEY, None)
            return batch
        finally:
            self.audio_vae = maybe_offload_module(
                self.audio_vae,
                enabled=bool(getattr(fastvideo_args, "vae_cpu_offload", False)),
            )


__all__ = ["MiniMaxH3AudioDecodingStage", "MiniMaxH3VideoDecodingStage"]
