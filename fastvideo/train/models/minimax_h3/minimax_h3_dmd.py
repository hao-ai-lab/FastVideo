# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 distribution-matching adapter (packed dual-modality latents)."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    audio_latent_num_frames,
)
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    _AUDIO_LATENT_CHANNELS,
    _AUDIO_SCHEDULER_SHIFT,
    _VIDEO_LATENT_CHANNELS,
    _VIDEO_SCHEDULER_SHIFT,
    MiniMaxH3Model,
    shift_noise_amount,
)

# DMD2 samples integer score timesteps on the legacy [0, 1000] scale. H3 maps
# them to its shared base noise amount before applying the modality shifts.
_DMD_TIMESTEP_SCALE = 1000


class MiniMaxH3DMDModel(MiniMaxH3Model):
    """Present H3's dual (video, audio) streams to DMD2 as one packed tensor.

    ``DMD2Method``'s rollout and loss math assume one latent tensor per
    sample. This adapter flattens both modality latents into one ``[1, N]``
    tensor (video's ``[B, T, 24, H, W]`` elements first, stereo audio's
    ``[B, 2, 32, Ta]`` elements after) so ``dmd2.py`` stays model-agnostic.
    Integer method timesteps become one shared base noise amount that is
    shifted per modality (video 12.0, audio 3.0), exactly as H3's paired
    schedulers synchronize the two streams during fine-tuning and inference.

    ``modality_slices()`` exposes the packed video/audio column ranges so
    DMD2 computes losses and normalizers per modality instead of one packed
    mean that video's element count would dominate.
    """

    @property
    def num_train_timesteps(self) -> int:
        return _DMD_TIMESTEP_SCALE

    # ------------------------------------------------------------------
    # Packed dual-modality helpers
    # ------------------------------------------------------------------

    def _modality_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the ``[1, T, C, H, W]`` video and ``[1, 2, 32, Ta]`` audio shapes."""
        data = self.training_config.data
        video_shape = (
            1,
            int(data.num_latent_t),
            _VIDEO_LATENT_CHANNELS,
            int(data.num_height) // 16,
            int(data.num_width) // 16,
        )
        audio_shape = (
            1,
            MINIMAX_H3_AUDIO_CHANNELS,
            _AUDIO_LATENT_CHANNELS,
            audio_latent_num_frames(int(data.num_frames)),
        )
        return video_shape, audio_shape

    def pack_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten both modality latents into one ``[B, N]`` tensor."""
        if video_latents.shape[0] != audio_latents.shape[0]:
            raise ValueError("Video and audio latent batches must match, got "
                             f"{video_latents.shape[0]} and {audio_latents.shape[0]}")
        batch_size = video_latents.shape[0]
        return torch.cat(
            (video_latents.reshape(batch_size, -1), audio_latents.reshape(batch_size, -1)),
            dim=1,
        )

    def modality_slices(self) -> tuple[tuple[str, slice], ...]:
        """Named packed-latent column slices for per-modality DMD2 losses.

        Video is ~3.6M packed elements against audio's ~15-30k, so a single
        global mean would give audio <1% of the distillation signal; DMD2
        consumes these slices to normalize and weight each stream separately.
        """
        video_shape, audio_shape = self._modality_shapes()
        split = math.prod(video_shape)
        return (
            ("video", slice(0, split)),
            ("audio", slice(split, split + math.prod(audio_shape))),
        )

    def unpack_latents(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split packed ``[B, N]`` rows back into (video, audio) latents."""
        video_shape, audio_shape = self._modality_shapes()
        split = math.prod(video_shape)
        packed_width = split + math.prod(audio_shape)
        if packed.ndim != 2 or packed.shape[1] != packed_width:
            raise ValueError("Packed latents must have shape "
                             f"[B, {packed_width}], got {tuple(packed.shape)}")
        batch_size = packed.shape[0]
        return (
            packed[:, :split].reshape(batch_size, *video_shape[1:]),
            packed[:, split:].reshape(batch_size, *audio_shape[1:]),
        )

    def _noise_amounts(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map one integer method timestep to both modality noise amounts."""
        base = (timestep.reshape(-1)[:1].to(torch.float32) / _DMD_TIMESTEP_SCALE)
        base = base.clamp(0.0, 1.0)
        return (
            shift_noise_amount(base, _VIDEO_SCHEDULER_SHIFT),
            shift_noise_amount(base, _AUDIO_SCHEDULER_SHIFT),
        )

    # ------------------------------------------------------------------
    # ModelBase overrides (packed convention)
    # ------------------------------------------------------------------

    def set_requires_negative_conditioning(self, requires: bool) -> None:
        """Fail fast: H3 cannot encode negative prompts at training time."""
        if requires:
            raise ValueError("MiniMaxH3DMDModel has no negative-prompt encoder; set "
                             "method.cfg_uncond={'text': 'zero'} for unconditional forwards")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Prepare the T2VA batch, then expose clean latents in packed form."""
        batch = super().prepare_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
        # DMD2 draws its own noise and timesteps per forward; only the packed
        # clean latents matter here (the base prepare_batch already built the
        # VSA metadata view for VSA-H3 roles). The fine-tuning noisy fields
        # are refreshed by predict_noise on every call.
        batch.latents = self.pack_latents(batch.latents, batch.audio_latents)
        return batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Noise packed latents at one shared timestep, shifted per modality."""
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        clean_video, clean_audio = self.unpack_latents(clean_latents)
        noise_video, noise_audio = self.unpack_latents(noise)
        return self.pack_latents(
            self._mix(clean_video, noise_video, sigma_video),
            self._mix(clean_audio, noise_audio, sigma_audio),
        )

    @staticmethod
    def _mix(
        clean: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.to(device=clean.device, dtype=clean.dtype)
        return (1.0 - sigma) * clean + sigma * noise

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Run one packed joint forward at an explicit method timestep.

        Both modality clean-time fields on ``batch`` are rewritten from
        ``timestep`` so the packed-row timestep plan and the backward
        forward-context stay coherent with this call.
        """
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        noisy_video, noisy_audio = self.unpack_latents(noisy_latents)
        batch.timesteps = (1.0 - sigma_video).to(noisy_latents.device)
        batch.audio_timesteps = (1.0 - sigma_audio).to(noisy_latents.device)
        batch.audio_noisy_model_input = noisy_audio
        video_pred, audio_pred = super().predict_noise(
            noisy_video,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        return self.pack_latents(video_pred, audio_pred)

    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Convert packed noise-minus-clean predictions to packed clean latents."""
        pred_noise = self.predict_noise(
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        noisy_video, noisy_audio = self.unpack_latents(noisy_latents)
        pred_video, pred_audio = self.unpack_latents(pred_noise)
        return self.pack_latents(
            self._to_x0(noisy_video, pred_video, sigma_video),
            self._to_x0(noisy_audio, pred_audio, sigma_audio),
        )

    @staticmethod
    def _to_x0(
        noisy: torch.Tensor,
        pred_noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        # noisy = (1 - sigma) * clean + sigma * noise and pred approximates
        # noise - clean, so clean = noisy - sigma * pred.
        sigma = sigma.to(device=noisy.device, dtype=noisy.dtype)
        return noisy - sigma * pred_noise

    # ------------------------------------------------------------------
    # Intermediate-latent visualization (LatentVisCallback)
    # ------------------------------------------------------------------

    def _load_vis_vae(self) -> Any:
        """Lazily load the H3 video VAE for visualization decodes.

        The module stays CPU-resident between decodes; ``decode_vis_latents``
        moves it to the GPU per call. Loading mirrors the H3 preprocess
        scripts: the inference component registry keeps precision policy and
        normalization identical to the published decode recipe.
        """
        vae = getattr(self, "_vis_vae_module", None)
        if vae is not None:
            return vae
        import os

        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.models.loader.component_loader import PipelineComponentLoader
        from fastvideo.utils import verify_model_config_and_directory

        model_index = verify_model_config_and_directory(
            self._init_from,
            required_component_dirs=("vae", ),
        )
        transformers_or_diffusers, _ = model_index["vae"][:2]
        args = FastVideoArgs(
            model_path=self._init_from,
            pipeline_config=self.training_config.pipeline_config,
            num_gpus=1,
            tp_size=1,
            sp_size=1,
            hsdp_shard_dim=1,
            use_fsdp_inference=False,
            vae_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )
        vae = PipelineComponentLoader.load_module(
            module_name="vae",
            component_model_path=os.path.join(self._init_from, "vae"),
            transformers_or_diffusers=transformers_or_diffusers,
            fastvideo_args=args,
        )
        if int(self.training_config.distributed.num_gpus) == 1:
            # A full BF16 H3 student leaves too little headroom for the
            # visualization VAE's default spatial tiles on an 80GB H100.
            from fastvideo.hooks.layerwise_offload import enable_layerwise_offload
            enable_layerwise_offload(vae.decoder)
            vae.enable_tiling(
                tile_sample_min_height=128,
                tile_sample_min_width=128,
            )
        vae.to("cpu")
        self._vis_vae_module = vae
        return vae

    @torch.no_grad()
    def decode_vis_latents(self, packed: torch.Tensor) -> Any:
        """Decode the packed video stream into a uint8 ``[B, T, C, H, W]`` clip.

        Follows ``MiniMaxH3VideoDecodingStage``: denormalize latents, decode
        under the published FP16-autocast-over-FP32 recipe, denormalize
        pixels. The audio stream is dropped — the tracker artifact is a
        silent video.
        """
        video_latents, _ = self.unpack_latents(packed.detach())
        latents = video_latents.permute(0, 2, 1, 3, 4).to(device=self.device, dtype=torch.float32)
        vae = self._load_vis_vae()
        if self.device.type == "cuda":
            # Rollout sampling can leave hundreds of MiB in fragmented cache
            # blocks. Release those unused blocks before materializing the VAE.
            torch.cuda.empty_cache()
        vae.to(self.device)
        try:
            latents = vae.denormalize_latents(latents)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                video = vae.decode(latents).sample
            video = vae.denormalize_pixels(video.float()).clamp_(0.0, 1.0).cpu()
        finally:
            vae.to("cpu")
        video = video.permute(0, 2, 1, 3, 4)
        return (video * 255.0).to(torch.uint8).numpy()


__all__ = ["MiniMaxH3DMDModel"]
