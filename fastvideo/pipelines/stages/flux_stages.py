# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage
from fastvideo.utils import PRECISION_TO_TYPE


def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Diffusers ``_pack_latents`` for FLUX (2×2 spatial pack in latent space)."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def _unpack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Inverse of ``_pack_latents``."""
    latents = latents.reshape(batch_size, height // 2, width // 2, num_channels_latents, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, num_channels_latents, height, width)


def _prepare_latent_image_ids(
    patch_height: int,
    patch_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Match Diffusers ``FluxPipeline._prepare_latent_image_ids`` (no batch dim)."""
    latent_image_ids = torch.zeros(patch_height, patch_width, 3, device=device, dtype=torch.float32)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(patch_height, device=device)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(patch_width, device=device)[None, :]
    h, w, c = latent_image_ids.shape
    return latent_image_ids.reshape(h * w, c).to(dtype=dtype)


class FluxInputValidationStage(InputValidationStage):
    """Require height/width divisible by 16 (VAE scale × 2 for FLUX packing)."""

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if (batch.height is not None and batch.width is not None and (batch.height % 16 != 0 or batch.width % 16 != 0)):
            raise ValueError("FLUX expects height and width divisible by 16 "
                             f"(VAE latent grid × 2× packing); got {batch.height}×{batch.width}.")
        return super().forward(batch, fastvideo_args)


class FluxConditioningStage(PipelineStage):
    """Build CLIP pooled + T5 sequence + ``text_ids`` (and optional negative for true CFG)."""

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if len(batch.prompt_embeds) < 2:
            raise ValueError("FluxConditioningStage expects 2 prompt_embeds (CLIP pooled, T5 sequence), "
                             f"got {len(batch.prompt_embeds)}")

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]

        pooled = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        enc = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        seq_len = enc.shape[1]
        text_ids = torch.zeros(seq_len, 3, device=device, dtype=torch.long)

        batch.extra["flux_pooled_projections"] = pooled
        batch.extra["flux_encoder_hidden_states"] = enc
        batch.extra["flux_text_ids"] = text_ids

        if batch.do_classifier_free_guidance:
            if not batch.negative_prompt_embeds or len(batch.negative_prompt_embeds) < 2:
                raise ValueError("True CFG requires two negative_prompt_embeds (CLIP, T5).")
            neg_pooled = batch.negative_prompt_embeds[0].to(device=device, dtype=target_dtype)
            neg_enc = batch.negative_prompt_embeds[1].to(device=device, dtype=target_dtype)
            batch.extra["flux_negative_pooled_projections"] = neg_pooled
            batch.extra["flux_negative_encoder_hidden_states"] = neg_enc

        return batch


class FluxTimestepPreparationStage(TimestepPreparationStage):
    """Flow Match with resolution-dependent ``mu`` from packed image sequence length."""

    @staticmethod
    def _calculate_mu(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return float(image_seq_len) * m + b

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sig = inspect.signature(self.scheduler.set_timesteps)
        if "mu" not in sig.parameters:
            return super().forward(batch, fastvideo_args)

        cfg = getattr(self.scheduler, "config", None)
        use_dynamic = bool(getattr(cfg, "use_dynamic_shifting", False)) if cfg is not None else False
        if not use_dynamic:
            return super().forward(batch, fastvideo_args)

        if batch.height is None or batch.width is None:
            raise ValueError("height/width must be set before FluxTimestepPreparationStage")

        vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
        spatial_ratio = int(getattr(vae_arch, "spatial_compression_ratio", 8))
        h_lat = batch.height // spatial_ratio
        w_lat = batch.width // spatial_ratio
        if h_lat % 2 != 0 or w_lat % 2 != 0:
            raise ValueError(
                f"Latent spatial dims must be even for FLUX packing; got {h_lat}×{w_lat} from {batch.height}×{batch.width}."
            )
        image_seq_len = (h_lat // 2) * (w_lat // 2)

        base_seq_len = int(getattr(cfg, "base_image_seq_len", 256))
        max_seq_len = int(getattr(cfg, "max_image_seq_len", 4096))
        base_shift = float(getattr(cfg, "base_shift", 0.5))
        max_shift = float(getattr(cfg, "max_shift", 1.15))

        device = get_local_torch_device()
        mu = self._calculate_mu(
            image_seq_len=image_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
        )
        self.scheduler.set_timesteps(batch.num_inference_steps, device=device, mu=mu)
        batch.timesteps = self.scheduler.timesteps
        return batch


class FluxLatentPreparationStage(PipelineStage):

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("height/width required for FluxLatentPreparationStage")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            if not batch.prompt_embeds:
                raise ValueError("prompt or prompt_embeds must be provided")
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        if isinstance(batch.generator, list) and len(batch.generator) != batch_size:
            raise ValueError(f"generator list length {len(batch.generator)} does not match batch_size {batch_size}")

        arch = fastvideo_args.pipeline_config.dit_config.arch_config
        in_channels = int(getattr(arch, "in_channels", 64))
        num_channels_latents = in_channels // 4

        vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
        spatial_ratio = int(getattr(vae_arch, "spatial_compression_ratio", 8))

        h_lat = batch.height // spatial_ratio
        w_lat = batch.width // spatial_ratio

        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        device = get_local_torch_device()

        shape = (batch_size, num_channels_latents, h_lat, w_lat)
        latents = batch.latents
        if latents is None:
            latents = randn_tensor(shape, generator=batch.generator, device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device=device, dtype=dtype)
            if latents.shape != shape:
                raise ValueError(f"Expected latents shape {shape}, got {tuple(latents.shape)}")
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma

        packed = _pack_latents(latents, batch_size, num_channels_latents, h_lat, w_lat)

        patch_h, patch_w = h_lat // 2, w_lat // 2
        img_ids = _prepare_latent_image_ids(patch_h, patch_w, device, dtype=torch.long)

        batch.latents = packed
        batch.raw_latent_shape = shape
        batch.extra["flux_h_lat"] = h_lat
        batch.extra["flux_w_lat"] = w_lat
        batch.extra["flux_num_channels_latents"] = num_channels_latents
        batch.extra["flux_latent_image_ids"] = img_ids

        return batch


class FluxDenoisingStage(PipelineStage):

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @staticmethod
    def _step_kwargs(scheduler_step, batch: ForwardBatch) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(scheduler_step)
        if "generator" in sig.parameters:
            gen = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
            kwargs["generator"] = gen
        return kwargs

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None:
            raise ValueError("timesteps must be set before FluxDenoisingStage")
        if batch.latents is None:
            raise ValueError("latents must be set before FluxDenoisingStage")

        packed = batch.latents
        timesteps = batch.timesteps

        pooled = batch.extra["flux_pooled_projections"]
        enc = batch.extra["flux_encoder_hidden_states"]
        txt_ids = batch.extra["flux_text_ids"]
        img_ids = batch.extra["flux_latent_image_ids"]

        neg_pooled = batch.extra.get("flux_negative_pooled_projections")
        neg_enc = batch.extra.get("flux_negative_encoder_hidden_states")

        true_cfg_scale = float(batch.true_cfg_scale)
        use_true_cfg = batch.do_classifier_free_guidance and true_cfg_scale > 1.0

        # Prefer the loaded transformer's arch (HF ``guidance_embeds``), not static pipeline defaults.
        tr_arch = self.transformer.fastvideo_config.arch_config
        guidance_embeds = bool(getattr(tr_arch, "guidance_embeds", False))

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        bs = packed.shape[0]
        if guidance_embeds:
            guidance = torch.full((bs, ), float(batch.guidance_scale), device=device, dtype=torch.float32)
        else:
            guidance = None

        step_extras = self._step_kwargs(self.scheduler.step, batch)

        for t in timesteps:
            t_scalar = t
            if not isinstance(t_scalar, torch.Tensor):
                t_scalar = torch.tensor([t_scalar], device=device, dtype=torch.float32)
            t_scalar = t_scalar.to(device=device, dtype=torch.float32)

            timestep_model = t_scalar.expand(bs).float() / 1000.0
            timestep_model = timestep_model.to(dtype=target_dtype)

            ts_ctx = int(t_scalar.reshape(-1)[0].item())
            with (
                    torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled and device.type == "cuda",
                    ),
                    set_forward_context(
                        current_timestep=ts_ctx,
                        attn_metadata=None,
                        forward_batch=batch,
                    ),
            ):
                if use_true_cfg:
                    assert neg_enc is not None and neg_pooled is not None
                    n_neg = self.transformer(
                        hidden_states=packed,
                        encoder_hidden_states=neg_enc,
                        pooled_projections=neg_pooled,
                        timestep=timestep_model,
                        guidance=guidance,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                    n_pos = self.transformer(
                        hidden_states=packed,
                        encoder_hidden_states=enc,
                        pooled_projections=pooled,
                        timestep=timestep_model,
                        guidance=guidance,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = n_neg + true_cfg_scale * (n_pos - n_neg)
                else:
                    noise_pred = self.transformer(
                        hidden_states=packed,
                        encoder_hidden_states=enc,
                        pooled_projections=pooled,
                        timestep=timestep_model,
                        guidance=guidance,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]

            packed = self.scheduler.step(
                noise_pred,
                t_scalar,
                packed,
                return_dict=False,
                **step_extras,
            )[0]

        batch.latents = packed
        return batch


class FluxDecodingStage(PipelineStage):
    """Unpack latents, apply VAE scaling/shift, decode to pixels (5D output ``B×3×1×H×W``)."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    @staticmethod
    def _denormalize_latents(latents: torch.Tensor, vae: Any) -> torch.Tensor:
        cfg = getattr(vae, "config", None)
        sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
        sh = getattr(cfg, "shift_factor", None) if cfg is not None else None
        if sf is None and hasattr(vae, "scaling_factor"):
            sf = vae.scaling_factor
        if sh is None and hasattr(vae, "shift_factor"):
            sh = vae.shift_factor
        if sf is not None:
            latents = latents / (sf.to(latents.device, latents.dtype) if isinstance(sf, torch.Tensor) else sf)
            if sh is not None:
                latents = latents + (sh.to(latents.device, latents.dtype) if isinstance(sh, torch.Tensor) else sh)
        return latents

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        packed = batch.latents
        if packed is None:
            raise ValueError("latents must be set before FluxDecodingStage")

        h_lat = int(batch.extra["flux_h_lat"])
        w_lat = int(batch.extra["flux_w_lat"])
        num_ch = int(batch.extra["flux_num_channels_latents"])
        raw_shape = batch.raw_latent_shape
        if raw_shape is None:
            raise ValueError("raw_latent_shape missing; FluxLatentPreparationStage must run first.")
        batch_size = int(raw_shape[0])

        infer_device = get_local_torch_device()
        packed = packed.to(infer_device)

        latents_4d = _unpack_latents(packed, batch_size, num_ch, h_lat, w_lat)
        latents_4d = self._denormalize_latents(latents_4d, self.vae)

        vae_device = next(self.vae.parameters()).device
        latents_4d = latents_4d.to(device=vae_device)

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast
        use_cuda_autocast = autocast_enabled and vae_device.type == "cuda"

        with torch.autocast(
                device_type="cuda",
                dtype=vae_dtype,
                enabled=use_cuda_autocast,
        ):
            if not autocast_enabled:
                latents_4d = latents_4d.to(dtype=vae_dtype)
            dec = self.vae.decode(latents_4d)
            image = dec.sample if hasattr(dec, "sample") else dec[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        batch.output = image.unsqueeze(2).detach().float().cpu()
        return batch
