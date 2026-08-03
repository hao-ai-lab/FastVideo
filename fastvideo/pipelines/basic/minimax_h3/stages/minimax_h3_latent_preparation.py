# SPDX-License-Identifier: Apache-2.0
"""Condition encoding, packed layout, and target noise for MiniMax H3."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedLayout,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    keyframe_condition_noise,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
    trim_reference_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MINIMAX_H3_KEYFRAME_ANCHORS_KEY,
    MINIMAX_H3_KEYFRAMES_KEY,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

MINIMAX_H3_LAYOUT_KEY = "minimax_h3_layout"


def _video_geometry(batch: ForwardBatch) -> tuple[int, int, int, int]:
    shape = batch.raw_latent_shape
    if shape is None or len(shape) != 5:
        raise ValueError("MiniMax-H3 input preparation must set a five-dimensional raw latent shape.")
    _, channels, num_frames, height, width = shape
    if min(channels, num_frames, height, width) <= 0:
        raise ValueError(f"MiniMax-H3 raw latent geometry must be positive, got {shape}.")
    return channels, num_frames, height, width


def _sample_visual_posterior(posterior: Any) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
    return posterior.sample(generator=generator)


class MiniMaxH3LatentPreparationStage(PipelineStage):
    """Encode fixed conditions, build the row layout, then draw target noise."""

    performance_component_metric = "vae_encode_time_s"

    def __init__(
        self,
        transformer: Any,
        vae: Any,
        audio_vae: Any,
        scheduler: Any,
        *,
        ref2va: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.scheduler = scheduler
        self.ref2va = ref2va

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("text_token_tags", batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY), V.with_dims(1))
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.not_none)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        if self.ref2va:
            result.add_check("references", batch.references, V.list_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        return result

    def _encode_visual_rows(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        patch_size = self.transformer.patch_size
        mean = self.vae.latents_mean.detach().float().cpu()
        std = self.vae.latents_std.detach().float().cpu()
        rows: list[torch.Tensor] = []
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
            latents = _sample_visual_posterior(posterior).to(torch.float16).float().cpu()
            reference.num_latent_frames = int(latents.shape[2])
            reference.latent_height = int(latents.shape[3])
            reference.latent_width = int(latents.shape[4])
            rows.append(patchify_video_latents((latents - mean) / std, patch_size))
        return rows

    def _encode_audio_rows(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        mean = self.audio_vae.latents_mean.detach().float().cpu().transpose(1, 2)
        std = self.audio_vae.latents_std.detach().float().cpu().transpose(1, 2)
        rows: list[torch.Tensor] = []
        for reference in references:
            if not reference.has_audio:
                continue
            if reference.waveform is None:
                raise ValueError("MiniMax-H3 reference waveform is missing.")
            posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None]).latent_dist
            latents = posterior.mode().float().cpu().transpose(1, 2)
            reference.num_audio_latents = int(latents.shape[1])
            rows.append(((latents - mean) / std).reshape(-1, self.audio_vae.latent_channels))
        return rows

    def _encode_fl2va_conditions(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        keyframes = batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY, [])
        if not isinstance(keyframes, list):
            raise TypeError("MiniMax-H3 keyframes must be a list.")
        if not keyframes:
            return None, None

        vae_device = get_local_torch_device()
        self.vae.to(vae_device)
        try:
            mean = self.vae.latents_mean.detach().float().cpu()
            std = self.vae.latents_std.detach().float().cpu()
            clean_rows: list[torch.Tensor] = []
            for image in keyframes:
                pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
                pixels = pixels.to(device=vae_device, dtype=torch.float32).div_(255.0)
                posterior = self.vae.encode_keyframe(self.vae.normalize_pixels(pixels)).latent_dist
                latents = _sample_visual_posterior(posterior).to(torch.float16).float().cpu()
                clean_rows.append(patchify_video_latents((latents - mean) / std, self.transformer.patch_size))
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")

        _, _, latent_height, latent_width = _video_geometry(batch)
        shapes = ((1, latent_height, latent_width), ) * len(keyframes)
        noise = keyframe_condition_noise(
            shapes,
            self.transformer.patch_size,
            self.vae.latent_channels,
            generator=batch.generator,
            device=device,
        )
        return (
            self.scheduler.scale_noise(
                torch.cat(clean_rows).to(device),
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                noise,
            ),
            None,
        )

    def _encode_ref2va_conditions(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        references = list(batch.references or [])
        if not references or not all(isinstance(item, MiniMaxH3PreparedReference) for item in references):
            raise TypeError("MiniMax-H3 Ref2VA latent preparation requires prepared references.")

        vae_device = get_local_torch_device()
        self.vae.to(vae_device)
        try:
            video_rows = self._encode_visual_rows(references, vae_device)
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")

        audio_rows: list[torch.Tensor] = []
        if any(reference.has_audio for reference in references):
            audio_device = get_local_torch_device()
            self.audio_vae.to(audio_device)
            try:
                audio_rows = self._encode_audio_rows(references, audio_device)
            finally:
                if fastvideo_args.vae_cpu_offload:
                    self.audio_vae.to("cpu")

        if not video_rows:
            raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference.")
        shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                       for reference in references if reference.media_type != "audio")
        noise = keyframe_condition_noise(
            shapes,
            self.transformer.patch_size,
            self.vae.latent_channels,
            generator=batch.generator,
            device=device,
        )
        video_conditions = self.scheduler.scale_noise(
            torch.cat(video_rows).to(device),
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        )
        audio_conditions = torch.cat(audio_rows).to(device) if audio_rows else None

        for reference in references:
            reference.image = None
            reference.frames = None
            reference.waveform = None
        return video_conditions, audio_conditions

    def _build_layout(self, batch: ForwardBatch) -> MiniMaxH3PackedLayout:
        text_token_tags = batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)
        if not isinstance(text_token_tags, torch.Tensor):
            raise ValueError("MiniMax-H3 conditioning must produce text token tags.")
        _, num_frames, height, width = _video_geometry(batch)
        if not isinstance(batch.num_frames, int):
            raise TypeError("MiniMax-H3 num_frames must be an integer.")
        num_audio_latents = audio_latent_num_frames(batch.num_frames)
        if self.ref2va:
            references = list(batch.references or [])
            return build_ref2va_packed_sequence(
                text_token_tags,
                references,
                num_frames,
                height,
                width,
                num_audio_latents,
                self.transformer.patch_size,
            )
        anchors = batch.extra.get(MINIMAX_H3_KEYFRAME_ANCHORS_KEY, ())
        if not isinstance(anchors, tuple):
            raise TypeError("MiniMax-H3 keyframe anchors must be a tuple.")
        return build_packed_sequence(
            text_token_tags,
            num_frames,
            height,
            width,
            num_audio_latents,
            self.transformer.patch_size,
            anchors,
        )

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        video_noise = batch.latents
        audio_noise = batch.audio_latents
        device = get_local_torch_device()
        if self.ref2va:
            condition_video, condition_audio = self._encode_ref2va_conditions(batch, fastvideo_args, device)
        else:
            condition_video, condition_audio = self._encode_fl2va_conditions(batch, fastvideo_args, device)

        layout = self._build_layout(batch)
        video_channels, num_frames, height, width = _video_geometry(batch)
        expected_video_shape = (1, video_channels, num_frames, height, width)
        if video_noise is None:
            video_noise = randn_tensor(
                expected_video_shape,
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        elif tuple(video_noise.shape) != expected_video_shape:
            raise ValueError(f"MiniMax-H3 injected video latents must have shape {expected_video_shape}, "
                             f"got {tuple(video_noise.shape)}.")
        video_rows = patchify_video_latents(video_noise.to(device=device, dtype=torch.float32),
                                            self.transformer.patch_size)

        num_audio_latents = layout.num_audio_latents
        expected_audio_shape = (MINIMAX_H3_AUDIO_CHANNELS, self.audio_vae.latent_channels, num_audio_latents)
        if audio_noise is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_vae.latent_channels),
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            if tuple(audio_noise.shape) != expected_audio_shape:
                raise ValueError(f"MiniMax-H3 injected audio latents must have shape {expected_audio_shape}, "
                                 f"got {tuple(audio_noise.shape)}.")
            audio_rows = audio_noise.to(device=device,
                                        dtype=torch.float32).permute(0, 2,
                                                                     1).reshape(-1, self.audio_vae.latent_channels)

        if condition_video is not None:
            video_rows = torch.cat((condition_video.to(device), video_rows))
        if condition_audio is not None:
            audio_rows = torch.cat((condition_audio.to(device), audio_rows))
        if video_rows.shape[0] != layout.video_indices.numel():
            raise ValueError("MiniMax-H3 packed video row count does not match its layout.")
        if audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError("MiniMax-H3 packed audio row count does not match its layout.")

        batch.latents = video_rows
        batch.audio_latents = audio_rows
        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
        batch.extra.pop(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY, None)
        batch.extra.pop(MINIMAX_H3_KEYFRAMES_KEY, None)
        batch.extra.pop(MINIMAX_H3_KEYFRAME_ANCHORS_KEY, None)
        batch.references = None
        return batch


__all__ = ["MINIMAX_H3_LAYOUT_KEY", "MiniMaxH3LatentPreparationStage"]
