# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 distribution-matching adapter (packed dual-modality latents)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, cast, Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    audio_latent_num_frames,
)
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    _AUDIO_LATENT_CHANNELS,
    _VIDEO_LATENT_CHANNELS,
    MiniMaxH3Model,
    shift_noise_amount,
)

# DMD2 expresses score time in timestep units on [0, 1000]. Strict FastGen
# parity keeps that coordinate continuous and applies shifts on max_t=0.999.
_DMD_TIMESTEP_SCALE = 1000
_FASTGEN_MAX_T = 0.999


@dataclass(frozen=True, slots=True)
class MiniMaxH3DMDLatentLayout:
    """Real scalar rollout columns followed by optional inert scalar padding.

    ``pad_tokens`` here counts flattened scalar columns, not joint DiT rows.
    Fixed transformer capacities are tracked separately by the semantic layout.
    """

    video_shape: tuple[int, int, int, int, int]
    audio_shape: tuple[int, int, int, int]
    # This layout describes scalar rollout columns, NOT joint transformer rows.
    # Capacity padding of joint rows is tracked by MiniMaxH3PackedLayout.
    pad_tokens: int = 0

    @property
    def valid(self) -> torch.Tensor:
        """Boolean validity of every scalar rollout column (CPU metadata)."""
        return torch.arange(self.packed_numel) < self.real_numel

    def __post_init__(self) -> None:
        if type(self.pad_tokens) is not int or self.pad_tokens < 0:
            raise ValueError("pad_tokens must be a non-negative integer")
        if self.video_shape[0] != 1 or self.video_shape[2] != _VIDEO_LATENT_CHANNELS:
            raise ValueError("DMD video latents must have shape [1, T, 24, H, W], got "
                             f"{self.video_shape}")
        if self.audio_shape[:3] != (1, MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_LATENT_CHANNELS):
            raise ValueError("DMD audio latents must have shape [1, 2, 32, Ta], got "
                             f"{self.audio_shape}")
        if any(value <= 0 for value in self.video_shape + self.audio_shape):
            raise ValueError("DMD latent layout dimensions must all be positive")

    @classmethod
    def from_latents(
        cls,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> MiniMaxH3DMDLatentLayout:
        if video_latents.ndim != 5 or audio_latents.ndim != 4:
            raise ValueError("H3 DMD pack expects video [1,T,24,H,W] and audio [1,2,32,Ta], "
                             f"got {tuple(video_latents.shape)} and {tuple(audio_latents.shape)}")
        return cls(
            video_shape=cast(tuple[int, int, int, int, int], tuple(int(value) for value in video_latents.shape)),
            audio_shape=cast(tuple[int, int, int, int], tuple(int(value) for value in audio_latents.shape)),
        )

    @property
    def video_numel(self) -> int:
        return math.prod(self.video_shape)

    @property
    def audio_numel(self) -> int:
        return math.prod(self.audio_shape)

    @property
    def real_numel(self) -> int:
        return self.video_numel + self.audio_numel

    @property
    def packed_numel(self) -> int:
        return self.real_numel + self.pad_tokens

    def modality_slices(self) -> tuple[tuple[str, slice], ...]:
        return (
            ("video", slice(0, self.video_numel)),
            ("audio", slice(self.video_numel, self.real_numel)),
        )


class MiniMaxH3DMDModel(MiniMaxH3Model):
    """Present H3's dual (video, audio) streams to DMD2 as one packed tensor.

    ``DMD2Method``'s rollout and loss math assume one latent tensor per
    sample. This adapter flattens both modality latents into one ``[1, N]``
    tensor (video's ``[1, T, 24, H, W]`` elements first, stereo audio's
    ``[1, 2, 32, Ta]`` elements after) so ``dmd2.py`` stays model-agnostic.
    Method timesteps become one shared base noise amount that is shifted per
    modality (released defaults: video 12.0, audio 3.0). Continuous score
    times use FastGen's ``max_t=0.999`` domain; integer rollout rungs preserve
    the release pipeline's unit-domain grid.

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

    def _modality_shapes(self) -> tuple[tuple[int, int, int, int, int], tuple[int, int, int, int]]:
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

    def _fixed_latent_layout(self) -> MiniMaxH3DMDLatentLayout:
        video_shape, audio_shape = self._modality_shapes()
        return MiniMaxH3DMDLatentLayout(
            video_shape=video_shape,
            audio_shape=audio_shape,
        )

    def _batch_latent_layout(self, batch: TrainingBatch) -> MiniMaxH3DMDLatentLayout:
        layout = batch.minimax_h3_dmd_layout
        if not isinstance(layout, MiniMaxH3DMDLatentLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_dmd_layout")
        return layout

    def pack_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        *,
        layout: MiniMaxH3DMDLatentLayout | None = None,
    ) -> torch.Tensor:
        """Flatten both modality latents into one ``[1, N]`` tensor."""
        actual_layout = MiniMaxH3DMDLatentLayout.from_latents(video_latents, audio_latents)
        if layout is not None and actual_layout != replace(layout, pad_tokens=0):
            raise ValueError(f"Latent tensors do not match their batch layout: {actual_layout} != {layout}")
        packed = torch.cat((video_latents.reshape(1, -1), audio_latents.reshape(1, -1)), dim=1)
        return torch.nn.functional.pad(packed, (0, layout.pad_tokens)) if layout and layout.pad_tokens else packed

    def modality_slices(
        self,
        *,
        layout: MiniMaxH3DMDLatentLayout | None = None,
    ) -> tuple[tuple[str, slice], ...]:
        """Named packed-latent column slices for per-modality DMD2 losses.

        Video is ~3.6M packed elements against audio's ~15-30k, so a single
        global mean would give audio <1% of the distillation signal; DMD2
        consumes these slices to normalize and weight each stream separately.
        """
        return (layout or self._fixed_latent_layout()).modality_slices()

    def modality_slices_for_batch(self, batch: TrainingBatch) -> tuple[tuple[str, slice], ...]:
        """Return per-modality slices for this batch's native shape."""
        return self.modality_slices(layout=self._batch_latent_layout(batch))

    def document_modality_slices_for_batch(self, batch: TrainingBatch) -> tuple[tuple[int, str, slice], ...]:
        """Return document-aware regions for unbiased packed-batch losses."""
        layout = self._batch_latent_layout(batch)
        return tuple((0, name, region) for name, region in layout.modality_slices())

    def unpack_latents(
        self,
        packed: torch.Tensor,
        *,
        layout: MiniMaxH3DMDLatentLayout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split one packed ``[1, N]`` tensor back into (video, audio) latents."""
        resolved = layout or self._fixed_latent_layout()
        if packed.shape != (1, resolved.packed_numel):
            raise ValueError(f"Packed latents must have shape [1, {resolved.packed_numel}], got {tuple(packed.shape)}")
        return (
            packed[:, :resolved.video_numel].reshape(resolved.video_shape),
            packed[:, resolved.video_numel:resolved.real_numel].reshape(resolved.audio_shape),
        )

    def _noise_amounts(
        self,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the document's shared method time to modality noise amounts."""
        # Strict FastGen score times are continuous and live on max_t=0.999.
        # Integer rollout rungs retain the release pipeline's unit-domain
        # schedule so training and offline validation walk identical states.
        warp_max = (_FASTGEN_MAX_T if timestep.is_floating_point() else 1.0)
        flat_timestep = timestep.reshape(-1)
        if flat_timestep.numel() > 1 and not bool(torch.all(flat_timestep == flat_timestep[0])):
            raise ValueError("H3 DMD packs one document; every timestep must agree")
        base = flat_timestep[:1].to(torch.float64) / _DMD_TIMESTEP_SCALE
        base = base.clamp(0.0, warp_max)
        return (
            shift_noise_amount(
                base,
                self.video_scheduler_shift,
                max_noise_amount=warp_max,
            ),
            shift_noise_amount(
                base,
                self.audio_scheduler_shift,
                max_noise_amount=warp_max,
            ),
        )

    # ------------------------------------------------------------------
    # ModelBase overrides (packed convention)
    # ------------------------------------------------------------------

    def set_requires_negative_conditioning(self, requires: bool) -> None:
        """Fail fast: H3 cannot encode negative prompts at training time."""
        if requires:
            raise ValueError("MiniMaxH3DMDModel has no negative-prompt encoder; set "
                             "method.cfg_uncond={'text': 'zero'} for unconditional forwards")

    def _pack_dmd_batch(self, batch: TrainingBatch) -> TrainingBatch:
        """Replace paired clean latents with the shared packed DMD2 view."""
        if batch.latents is None or batch.audio_latents is None:
            raise RuntimeError("MiniMax H3 batch preparation did not produce paired latents")
        single_layout = MiniMaxH3DMDLatentLayout.from_latents(batch.latents, batch.audio_latents)
        batch.minimax_h3_dmd_layout = single_layout
        batch.latents = self.pack_latents(batch.latents, batch.audio_latents, layout=single_layout)
        return batch

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
        return self._pack_dmd_batch(batch)

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Noise legacy fixed-shape packed latents at one shared timestep."""
        return self._add_noise_with_layout(
            clean_latents,
            noise,
            timestep,
            self._fixed_latent_layout(),
        )

    def add_noise_for_batch(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        """Noise packed latents using immutable geometry from ``batch``."""
        layout = self._batch_latent_layout(batch)
        if not layout.pad_tokens and not bool(getattr(self.training_config.data, "native_shape_bucketing", False)):
            # Keep the established fixed-shape call path observable and
            # byte-identical for existing data-free/data-forcing recipes.
            return self.add_noise(clean_latents, noise, timestep)
        return self._add_noise_with_layout(
            clean_latents,
            noise,
            timestep,
            layout,
        )

    def _add_noise_with_layout(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        layout: MiniMaxH3DMDLatentLayout,
    ) -> torch.Tensor:
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        clean_video, clean_audio = self.unpack_latents(clean_latents, layout=layout)
        noise_video, noise_audio = self.unpack_latents(noise, layout=layout)
        return self.pack_latents(
            self._mix(clean_video, noise_video, sigma_video),
            self._mix(clean_audio, noise_audio, sigma_audio),
            layout=layout,
        )

    @staticmethod
    def _mix(
        clean: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = clean.dtype
        clean_fp64 = clean.to(torch.float64)
        noise_fp64 = noise.to(torch.float64)
        sigma_fp64 = sigma.to(device=clean.device, dtype=torch.float64)
        return ((1.0 - sigma_fp64) * clean_fp64 + sigma_fp64 * noise_fp64).to(original_dtype)

    def extract_eps(
        self,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Recover noise for legacy fixed-shape packed latents.

        The inverse of :meth:`add_noise` at the same shared timestep: with
        ``x_t = (1 - sigma_m) x0 + sigma_m eps`` under each modality's
        shifted sigma (video 12.0, audio 3.0), the implied noise is
        ``eps_m = (x_t - (1 - sigma_m) x0) / sigma_m``. DMD2's ODE renoise
        uses this to step a carried trajectory deterministically between
        grid rungs.
        """
        return self._extract_eps_with_layout(
            noisy_latents,
            clean_latents,
            timestep,
            self._fixed_latent_layout(),
        )

    def extract_eps_for_batch(
        self,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        """Recover packed noise using immutable geometry from ``batch``."""
        layout = self._batch_latent_layout(batch)
        if not layout.pad_tokens and not bool(getattr(self.training_config.data, "native_shape_bucketing", False)):
            return self.extract_eps(noisy_latents, clean_latents, timestep)
        return self._extract_eps_with_layout(
            noisy_latents,
            clean_latents,
            timestep,
            layout,
        )

    def _extract_eps_with_layout(
        self,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        layout: MiniMaxH3DMDLatentLayout,
    ) -> torch.Tensor:
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        noisy_video, noisy_audio = self.unpack_latents(noisy_latents, layout=layout)
        clean_video, clean_audio = self.unpack_latents(clean_latents, layout=layout)
        return self.pack_latents(
            self._unmix(noisy_video, clean_video, sigma_video),
            self._unmix(noisy_audio, clean_audio, sigma_audio),
            layout=layout,
        )

    @staticmethod
    def _unmix(
        noisy: torch.Tensor,
        clean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = noisy.dtype
        noisy_fp64 = noisy.to(torch.float64)
        clean_fp64 = clean.to(torch.float64)
        sigma_fp64 = sigma.to(device=noisy.device, dtype=torch.float64)
        # The DMD grid never renoises from t=0, but clamp so a degenerate
        # call cannot divide by zero.
        eps = ((noisy_fp64 - (1.0 - sigma_fp64) * clean_fp64) / sigma_fp64.clamp_min(1e-6))
        return eps.to(original_dtype)

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
        layout = self._batch_latent_layout(batch)

        noisy_video = self._bind_packed_forward_inputs(noisy_latents, timestep, batch)
        video_pred, audio_pred = super().predict_noise(
            noisy_video,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        return self.pack_latents(video_pred, audio_pred, layout=layout)

    def _bind_packed_forward_inputs(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        """Unpack one packed state and rewrite the batch's modality time fields for *timestep*.

        Returns the video input; the audio input lands on ``batch``.
        """
        layout = self._batch_latent_layout(batch)
        sigma_video, sigma_audio = self._noise_amounts(timestep)
        noisy_video, noisy_audio = self.unpack_latents(noisy_latents, layout=layout)
        batch.timesteps = (1.0 - sigma_video).to(noisy_latents.device)
        batch.audio_timesteps = (1.0 - sigma_audio).to(noisy_latents.device)
        batch.audio_noisy_model_input = noisy_audio
        return noisy_video

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
        layout = self._batch_latent_layout(batch)

        sigma_video, sigma_audio = self._noise_amounts(timestep)
        noisy_video, noisy_audio = self.unpack_latents(noisy_latents, layout=layout)
        pred_video, pred_audio = self.unpack_latents(pred_noise, layout=layout)
        return self.pack_latents(
            self._to_x0(noisy_video, pred_video, sigma_video),
            self._to_x0(noisy_audio, pred_audio, sigma_audio),
            layout=layout,
        )

    @staticmethod
    def _to_x0(
        noisy: torch.Tensor,
        pred_noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        # noisy = (1 - sigma) * clean + sigma * noise and pred approximates
        # noise - clean, so clean = noisy - sigma * pred.
        original_dtype = noisy.dtype
        noisy_fp64 = noisy.to(torch.float64)
        pred_noise_fp64 = pred_noise.to(torch.float64)
        sigma_fp64 = sigma.to(device=noisy.device, dtype=torch.float64)
        return (noisy_fp64 - sigma_fp64 * pred_noise_fp64).to(original_dtype)

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

        model_index = verify_model_config_and_directory(self._init_from)
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
        vae.to("cpu")
        self._vis_vae_module = vae
        return vae

    @torch.no_grad()
    def decode_vis_latents(
        self,
        packed: torch.Tensor,
        *,
        layout: MiniMaxH3DMDLatentLayout | None = None,
    ) -> Any:
        """Decode the first packed video into a uint8 ``[B, T, C, H, W]`` clip.

        Follows ``MiniMaxH3VideoDecodingStage``: denormalize latents, decode
        under the published FP16-autocast-over-FP32 recipe, denormalize
        pixels. The audio stream is dropped — the tracker artifact is a
        silent video.
        """
        video_latents, _ = self.unpack_latents(packed.detach(), layout=layout)
        latents = video_latents.permute(0, 2, 1, 3, 4).to(device=self.device, dtype=torch.float32)
        vae = self._load_vis_vae()
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


__all__ = ["MiniMaxH3DMDLatentLayout", "MiniMaxH3DMDModel"]
