# SPDX-License-Identifier: Apache-2.0
"""Draft video -> upscaled, re-noised H3 latent plus the fixed silent-audio rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models import pinned_offload
from fastvideo.models.upsamplers.minimax_h3_latent_upscaler import MiniMaxH3LatentUpscaler, upscale_latent
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    h3_dit_patch_size,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
    MINIMAX_H3_LAYOUT_KEY,
    MiniMaxH3LatentPreparationStage,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k.draft_latent_cache import (
    DraftLatentCache,
    draft_latent_cache_key,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k.draft_video import load_draft_pixels
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sampling import mix_initial_state, refine_sigmas
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)

DRAFT_FRAME_COUNT = 124


class Regenerate2KLatentPreparationStage(MiniMaxH3LatentPreparationStage):
    """Encode the draft with the H3 VAE, upscale its latent to the target grid and re-noise it to the first sigma."""

    def __init__(self,
                 vae: Any,
                 audio_vae: Any,
                 scheduler: Any,
                 *,
                 load_upscaler: Callable[[], MiniMaxH3LatentUpscaler],
                 draft_size: str,
                 start_sigma: float | None,
                 draft_latent_path: str | None = None) -> None:
        super().__init__(vae, audio_vae, scheduler, ref2va=False)
        self.load_upscaler = load_upscaler
        self.draft_size = draft_size
        self.start_sigma = start_sigma
        self.draft_latent_path = draft_latent_path
        self._silence: tuple[tuple[str, int], torch.Tensor] | None = None
        self._draft_cache = DraftLatentCache()

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_input(batch, fastvideo_args)
        if self.draft_latent_path is None:
            result.add_check("video_path", batch.video_path, lambda value: isinstance(value, str) and bool(value))
        return result

    def _encode_draft(self, video_path: str, device: torch.device, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """Normalized native latent [C, T, h, w] of the draft, with the VAE posterior sampled under a fixed seed."""
        pixels = load_draft_pixels(video_path, self.draft_size)
        pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
        # Encoder spatial tile override (GB10): the reference encodes the draft at
        # 256x256 tiles / 64 overlap, which on sm_121 is dominated by launch and
        # copy overhead rather than FLOPs. `FASTVIDEO_H3_VAE_ENCODE_TILE="H,W,OH,OW"`
        # (e.g. "768,768,128,128") makes the encoder do fewer, larger passes.
        tile_spec = envs.FASTVIDEO_H3_VAE_ENCODE_TILE.get()
        if tile_spec:
            tile_height, tile_width, overlap_height, overlap_width = (int(x) for x in tile_spec.split(","))
            self.vae.enable_tiling(tile_height, tile_width, overlap_height, overlap_width)
        try:
            posterior = self.vae.encode_pixels(pixels).latent_dist
            sample = posterior.sample(generator=torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            return self.vae.normalize_latents(sample.float())[0].to(device)
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.vae)

    def _encode_draft_cached(self, video_path: str, device: torch.device,
                             fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """``_encode_draft`` behind an exact content-keyed cache (draft_latent_cache).

        The encode is a pure function of the draft file, tiling, fixed posterior
        seed and model bundle, so an identical presentation reuses the encoder's
        own output bit-for-bit instead of re-encoding (~82 s on the Spark).
        """
        cache = self._draft_cache
        if not cache.enabled:
            return self._encode_draft(video_path, device, fastvideo_args)
        key = draft_latent_cache_key(video_path, self.draft_size, envs.FASTVIDEO_H3_VAE_ENCODE_TILE.get(),
                                     getattr(fastvideo_args, "model_path", None), MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        cached = cache.get(key)
        if cached is not None:
            logger.info("Open-H3 draft-latent cache hit (%s); skipping the VAE encode", key)
            return cached.to(device=device, dtype=torch.float32)
        draft = self._encode_draft(video_path, device, fastvideo_args)
        cache.put(key, draft)
        return draft

    def _draft_latent_source(self, video_path: str | None) -> str | None:
        """The saved draft latent for this request: the configured one, or a ``.pt`` passed as the draft video."""
        if self.draft_latent_path is not None:
            return self.draft_latent_path
        return video_path if video_path is not None and video_path.lower().endswith(".pt") else None

    def _load_draft_latent(self, path: str, device: torch.device) -> torch.Tensor:
        """Normalized native latent [C, T, h, w] saved by a draft run, standing in for the encoded draft video."""
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(tensor, dict):
            raise ValueError(f"{path} holds a mapping; save the draft latent tensor itself.")
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"A draft latent must be [C, T, h, w] or [1, C, T, h, w], got {tuple(tensor.shape)}.")
        return tensor.to(device=device, dtype=torch.float32)

    def _silent_audio_rows(self, num_frames: int, device: torch.device, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """The encoded-silence audio stream, cached: it is the same for every request of this length."""
        key = (str(device), num_frames)
        if self._silence is not None and self._silence[0] == key:
            return self._silence[1]
        pinned_offload.load(self.audio_vae, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            samples = round(num_frames / MINIMAX_H3_FPS * int(self.audio_vae.sampling_rate))
            silence = torch.zeros(MINIMAX_H3_AUDIO_CHANNELS, 1, samples, device=device)
            posterior = self.audio_vae.encode(silence).latent_dist
            latents = self.audio_vae.normalize_latents(posterior.mode().float()).transpose(1, 2)
            rows = latents.reshape(-1, self.audio_vae.latent_channels).to(torch.bfloat16)
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.audio_vae)
        self._silence = (key, rows)
        return rows

    @staticmethod
    def _noise_rows(shape: tuple[int, int], device: torch.device, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn((1, *shape), device=device, dtype=torch.float32, generator=generator)[0]

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.num_frames != DRAFT_FRAME_COUNT:
            raise ValueError(f"Open-H3-Regenerate-2K refines {DRAFT_FRAME_COUNT} frames, got {batch.num_frames}.")
        device = get_local_torch_device()
        patch_size = h3_dit_patch_size(fastvideo_args)
        _, channels, num_latent_frames, latent_height, latent_width = batch.raw_latent_shape

        latent_source = self._draft_latent_source(batch.video_path)
        if latent_source is None:
            draft = self._encode_draft_cached(batch.video_path, device, fastvideo_args)
        else:
            draft = self._load_draft_latent(latent_source, device)
        upscaler = self.load_upscaler()
        upscaled = upscale_latent(upscaler, draft, (batch.height, batch.width))
        if tuple(upscaled.shape) != (channels, num_latent_frames, latent_height, latent_width):
            raise ValueError(f"The upscaled latent has shape {tuple(upscaled.shape)}, expected "
                             f"{(channels, num_latent_frames, latent_height, latent_width)}.")
        upscaled_rows = patchify_video_latents(upscaled[None], patch_size)

        if batch.latents is None:
            noise_rows = self._noise_rows(tuple(upscaled_rows.shape), device, 0 if batch.seed is None else batch.seed)
        else:
            expected = (1, channels, num_latent_frames, latent_height, latent_width)
            if tuple(batch.latents.shape) != expected:
                raise ValueError(f"Injected noise must have shape {expected}, got {tuple(batch.latents.shape)}.")
            noise_rows = patchify_video_latents(batch.latents.to(device=device, dtype=torch.float32), patch_size)

        layout = self._build_layout(batch, fastvideo_args)
        audio_rows = self._silent_audio_rows(batch.num_frames, device, fastvideo_args)
        if audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError(f"The silent audio has {audio_rows.shape[0]} rows, the packed layout expects "
                             f"{layout.audio_indices.numel()} ({layout.num_audio_latents} latents).")

        batch.latents = mix_initial_state(upscaled_rows, noise_rows,
                                          refine_sigmas(batch.num_inference_steps, self.start_sigma)[0])
        batch.audio_latents = audio_rows
        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
        for key in tuple(batch.extra):
            if key.startswith("minimax_h3_") and key != MINIMAX_H3_LAYOUT_KEY:
                batch.extra.pop(key)
        return batch


__all__ = ["Regenerate2KLatentPreparationStage"]
