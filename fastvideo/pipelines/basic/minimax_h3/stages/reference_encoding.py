# SPDX-License-Identifier: Apache-2.0
"""Video/audio VAE conditioning for MiniMax H3 Ref2VA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    keyframe_condition_noise,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.packing_ref2va import trim_reference_num_frames
from fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle import (
    maybe_offload_module,
    module_device,
    move_module_to_local_device,
)
from fastvideo.pipelines.basic.minimax_h3.stages._vae_conditioning import arch_value, latent_stats, sample_posterior
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3PreparedReference, get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class MiniMaxH3ReferenceEncodingStage(PipelineStage):
    """Encode ordered reference media and create fixed conditioning rows."""

    def __init__(self, vae: Any, audio_vae: Any, transformer: Any, scheduler: Any) -> None:
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae
        self.transformer = transformer
        self.scheduler = scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("prepared_references", state.prepared_references, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("condition_video_latents", state.condition_video_latents, V.with_dims(2))
        result.add_check("condition_audio_latents", state.condition_audio_latents, V.none_or_tensor)
        return result

    def _encode_visuals(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        patch_size = tuple(int(value) for value in arch_value(self.transformer, "patch_size"))
        latents_mean, latents_std = latent_stats(self.vae, (1, -1, 1, 1, 1))
        rows = []
        for reference in references:
            if reference.media_type == "audio":
                continue
            if reference.media_type == "image":
                if reference.image is None:
                    raise ValueError("MiniMax-H3 reference image pixels are missing.")
                pixels = torch.from_numpy(np.asarray(reference.image).copy()).permute(2, 0, 1)[None, :, None]
                encode = self.vae.encode_keyframe
            else:
                if reference.frames is None:
                    raise ValueError("MiniMax-H3 reference video frames are missing.")
                frames = reference.frames[:trim_reference_num_frames(reference.frames.shape[0])]
                pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
                encode = self.vae.encode

            pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
            posterior = encode(self.vae.normalize_pixels(pixels)).latent_dist
            latents = sample_posterior(posterior, MINIMAX_H3_KEYFRAME_ENCODE_SEED)
            latents = latents.to(torch.float16).float().cpu()
            reference.num_latent_frames = int(latents.shape[2])
            reference.latent_height = int(latents.shape[3])
            reference.latent_width = int(latents.shape[4])
            rows.append(patchify_video_latents((latents - latents_mean) / latents_std, patch_size))
        return rows

    def _encode_audio(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        latent_channels = int(arch_value(self.audio_vae, "latent_channels"))
        latents_mean, latents_std = latent_stats(self.audio_vae, (1, 1, -1))
        rows = []
        for reference in references:
            if not reference.has_audio:
                continue
            if reference.waveform is None:
                raise ValueError("MiniMax-H3 reference waveform is missing.")
            posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None]).latent_dist
            latents = posterior.mode().float().cpu().transpose(1, 2)
            reference.num_audio_latents = int(latents.shape[1])
            rows.append(((latents - latents_mean) / latents_std).reshape(-1, latent_channels))
        return rows

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = get_minimax_h3_state(batch)
        references = state.prepared_references
        if not references:
            raise ValueError("MiniMax-H3 Ref2VA media must be prepared before reference encoding.")

        self.vae, vae_device, _ = move_module_to_local_device(self.vae)
        try:
            video_rows = self._encode_visuals(references, vae_device)
        finally:
            self.vae = maybe_offload_module(
                self.vae,
                enabled=bool(getattr(fastvideo_args, "vae_cpu_offload", False)),
            )

        audio_rows = []
        if any(reference.has_audio for reference in references):
            self.audio_vae, audio_vae_device, _ = move_module_to_local_device(self.audio_vae)
            try:
                audio_rows = self._encode_audio(references, audio_vae_device)
            finally:
                self.audio_vae = maybe_offload_module(
                    self.audio_vae,
                    enabled=bool(getattr(fastvideo_args, "vae_cpu_offload", False)),
                )

        if not video_rows:
            raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference.")
        transformer_device = module_device(self.transformer)
        patch_size = tuple(int(value) for value in arch_value(self.transformer, "patch_size"))
        latent_channels = int(arch_value(self.vae, "latent_channels"))
        noise = keyframe_condition_noise(
            tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                  for reference in references if reference.media_type != "audio"),
            patch_size,
            latent_channels,
            generator=batch.generator,
            device=transformer_device,
        )
        state.condition_video_latents = self.scheduler.scale_noise(
            torch.cat(video_rows).to(transformer_device),
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        )
        state.condition_audio_latents = torch.cat(audio_rows).to(transformer_device) if audio_rows else None

        # Qwen and both VAEs have consumed the prepared media by this point; only
        # geometry and timestamps remain live for layout construction.
        for reference in references:
            reference.image = None
            reference.frames = None
            reference.waveform = None
        return batch


__all__ = ["MiniMaxH3ReferenceEncodingStage"]
