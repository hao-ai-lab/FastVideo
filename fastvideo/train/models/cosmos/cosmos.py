# SPDX-License-Identifier: Apache-2.0
"""Cosmos model plugin (per-role instance).

Subclasses WanModel since Cosmos uses the same
FlowMatchEulerDiscreteScheduler.  Differences:
  - transformer class name: CosmosTransformer3DModel
  - normalize_dit_input("cosmos", ...) instead of ("wan", ...)
  - forward kwargs: no encoder_attention_mask, needs
    condition_mask + padding_mask + fps
  - hidden_states in (B,C,T,H,W) — no permute needed
  - default flow_shift = 1.0
  - single T5 text encoder (not dual like Hunyuan)
"""

from __future__ import annotations

import copy
import gc
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import (
    normalize_dit_input, )

from fastvideo.train.models.wan.wan import WanModel

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


class CosmosModel(WanModel):
    """Cosmos 2.5 per-role model.

    Inherits most behaviour from WanModel (noise scheduler,
    timestep sampling, attention metadata, backward).  Overrides
    only the pieces that differ for Cosmos 2.5.

    Cosmos 2.5 uses:
    - Cosmos25Transformer3DModel (velocity prediction)
    - EDM noise schedule: x_t = x_0 + sigma * eps
    - No input/output preconditioning (raw latents)
    - Timestep = raw sigma value
    - Model output = velocity ≈ noise
    """

    _transformer_cls_name: str = "Cosmos25Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 1.0,
        enable_gradient_checkpointing_type: str
        | None = None,
        transformer_override_safetensor: str
        | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=(disable_custom_init_weights),
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            transformer_override_safetensor=(transformer_override_safetensor),
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Same flow as Wan, but uses Cosmos VAE
        normalisation."""
        self.ensure_negative_conditioning()
        assert self.training_config is not None
        tc = self.training_config

        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        infos = raw_batch.get("info_list")

        if latents_source == "zeros":
            batch_size = encoder_hidden_states.shape[0]
            vae_config = (
                tc.pipeline_config.vae_config  # type: ignore[union-attr]
                .arch_config)
            num_channels = getattr(
                vae_config,
                "z_dim",
                getattr(vae_config, "latent_channels", 16),
            )
            spatial_compression_ratio = (vae_config.spatial_compression_ratio)
            latent_height = (tc.data.num_height // spatial_compression_ratio)
            latent_width = (tc.data.num_width // spatial_compression_ratio)
            latents = torch.zeros(
                batch_size,
                num_channels,
                tc.data.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError("vae_latent not found in batch "
                                 "and latents_source='data'")
            latents = raw_batch["vae_latent"]
            latents = latents[:, :, :tc.data.num_latent_t]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(f"Unknown latents_source: "
                             f"{latents_source!r}")

        training_batch.latents = latents
        training_batch.encoder_hidden_states = (encoder_hidden_states.to(device, dtype=dtype))
        training_batch.encoder_attention_mask = (encoder_attention_mask.to(device, dtype=dtype))
        training_batch.infos = infos

        # KEY DIFFERENCE: "cosmos" normalisation
        training_batch.latents = normalize_dit_input(
            "cosmos",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

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
        device_type = self.device.type
        dtype = noisy_latents.dtype
        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError(
                    "Missing conditional_dict in TrainingBatch"
                )
        else:
            text_dict = self._get_uncond_text_dict(
                batch, cfg_uncond=cfg_uncond,
            )

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        # finetune.py hands us `noisy_latents` in (B, T, C, H, W)
        # (Wan canonical). Un-permute back to the (B, C, T, H, W)
        # layout Cosmos's transformer expects.
        noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)

        # ----------------------------------------------------------
        # Cosmos 2.5: velocity prediction, no preconditioning.
        #
        # Noise schedule (EDM):  x_t = x_0 + sigma * eps
        # Model input:   raw x_t (no c_in scaling)
        # Timestep:      raw sigma value
        # Model output:  velocity v ≈ noise
        #
        # finetune.py (precondition_outputs=True) computes:
        #   pred_x0 = x_t - pred * sigma
        # Since v ≈ noise, this gives:
        #   pred_x0 = (x_0 + sigma*eps) - eps*sigma = x_0  ✓
        # ----------------------------------------------------------
        assert batch.sigmas is not None
        sigma_1d = batch.sigmas.flatten()[:noisy_latents.shape[0]]

        # Cosmos 2.5 timestep = raw sigma (matching inference
        # convention where scheduler timestep * 0.001 = sigma).
        cosmos_timestep = sigma_1d

        with (
            torch.autocast(device_type, dtype=dtype),
            set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
            ),
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents, cosmos_timestep, text_dict,
            )
            transformer = self._get_transformer(timestep)
            model_output = transformer(**input_kwargs)

        # Raw velocity output (≈ noise).
        # Re-permute (B, C, T, H, W) → (B, T, C, H, W)
        pred = model_output.permute(0, 2, 1, 3, 4)
        return pred

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        """Build transformer forward kwargs for Cosmos 2.5.

        Cosmos 2.5 transformer expects:
        - hidden_states in (B, C, T, H, W)
        - timestep as (B, 1) for T2W (scalar per sample)
        - condition_mask and padding_mask
        - fps tensor
        """
        if text_dict is None:
            raise ValueError(
                "text_dict cannot be None for Cosmos forward"
            )

        b, c, t, h, w = noise_input.shape

        # For T2W fine-tuning: no conditioning frames.
        condition_mask = torch.zeros(
            b, 1, t, h, w,
            device=noise_input.device,
            dtype=noise_input.dtype,
        )

        # Padding mask: zeros (no padding).
        padding_mask = torch.zeros(
            1, 1, h, w,
            device=noise_input.device,
            dtype=noise_input.dtype,
        )

        # Cosmos 2.5 expects timestep as (B, 1) for T2W.
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1)

        return {
            "hidden_states": noise_input,
            "encoder_hidden_states": (
                text_dict["encoder_hidden_states"]
            ),
            "timestep": timestep,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "fps": 16,
        }

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        """Prepare DiT inputs for Cosmos using EDM noise schedule.

        Unlike Wan (flow-matching), Cosmos uses the EDM convention:
            x_t = x_0 + sigma * eps
        where sigma is sampled from a log-normal distribution
        matching the pretrained model's training distribution.
        """
        assert self.training_config is not None
        tc = self.training_config
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Sample EDM sigmas from log-normal distribution.
        # Parameters match the Karras/EDM training convention:
        #   ln(sigma) ~ N(P_mean, P_std²)
        # with P_mean=0.7, P_std=1.6 (NVIDIA Cosmos defaults).
        p_mean = 0.7
        p_std = 1.6
        log_sigma = (
            torch.randn(
                batch_size,
                generator=generator,
                device=latents.device,
                dtype=latents.dtype,
            )
            * p_std
            + p_mean
        )
        sigmas = log_sigma.exp()
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(sigmas, src=0)

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1, 1)
        sigmas_5d = sigmas.view(-1, 1, 1, 1, 1)

        # EDM noise schedule: x_t = x_0 + sigma * eps
        noisy_model_input = latents + sigmas_5d * noise

        # Store flow-matching compatible timesteps for any code
        # that reads them (e.g., attention metadata).
        timesteps = self._sample_timesteps(
            batch_size, latents.device, generator,
        )
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(timesteps, src=0)

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas_5d
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        training_batch.conditional_dict = {
            "encoder_hidden_states": (
                training_batch.encoder_hidden_states
            ),
            "encoder_attention_mask": (
                training_batch.encoder_attention_mask
            ),
        }

        if (
            self.negative_prompt_embeds is not None  # type: ignore[has-type]
            and self.negative_prompt_attention_mask  # type: ignore[has-type]
            is not None
        ):
            neg_embeds = self.negative_prompt_embeds  # type: ignore[has-type]
            neg_mask = self.negative_prompt_attention_mask  # type: ignore[has-type]
            if neg_embeds.shape[0] == 1 and batch_size > 1:
                neg_embeds = neg_embeds.expand(
                    batch_size, *neg_embeds.shape[1:],
                ).contiguous()
            if neg_mask.shape[0] == 1 and batch_size > 1:
                neg_mask = neg_mask.expand(
                    batch_size, *neg_mask.shape[1:],
                ).contiguous()
            training_batch.unconditional_dict = {
                "encoder_hidden_states": neg_embeds,
                "encoder_attention_mask": neg_mask,
            }

        # Match the canonical (B, T, C, H, W) layout that
        # finetune.py expects for `training_batch.latents`.
        # We un-permute back to (B, C, T, H, W) at the
        # transformer boundary in `predict_noise`.
        training_batch.latents = training_batch.latents.permute(
            0, 2, 1, 3, 4,
        )
        return training_batch

    def ensure_negative_conditioning(self) -> None:
        """Create negative (unconditional) prompt embeddings.

        Cosmos 2.5 uses Reason1 (Qwen2.5-VL) which is expensive
        to load.  For training with ``training_cfg_rate=0`` (no
        classifier-free guidance dropout), the negative embedding
        is never used.  We create a zero-valued placeholder that
        matches the text embedding dimension from the dataset.
        """
        if self.negative_prompt_embeds is not None:  # type: ignore[has-type]
            return

        device = self.device
        dtype = self._get_training_dtype()

        # Infer embedding dimension from the pipeline config's
        # text encoder settings, or fall back to a reasonable
        # default for Cosmos 2.5 (Reason1 full_concat: 100352).
        assert self.training_config is not None
        tc = self.training_config
        text_enc_cfgs = tc.pipeline_config.text_encoder_configs
        if text_enc_cfgs:
            arch = text_enc_cfgs[0].arch_config
            embed_dim = getattr(arch, "hidden_size", 100352)
        else:
            embed_dim = 100352

        num_tokens = 512  # Reason1 default padding length

        neg_embeds = torch.zeros(
            1, num_tokens, embed_dim,
            device=device, dtype=dtype,
        )
        neg_mask = torch.ones(
            1, num_tokens,
            device=device, dtype=dtype,
        )

        self.negative_prompt_embeds = neg_embeds
        self.negative_prompt_attention_mask = neg_mask
