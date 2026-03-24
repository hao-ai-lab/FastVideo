# SPDX-License-Identifier: Apache-2.0
"""Hunyuan model plugin (per-role instance).

Subclasses WanModel since HunyuanVideo uses the same
FlowMatchEulerDiscreteScheduler and linear-interpolation noise
schedule.  Differences:
  - transformer class name
  - normalize_dit_input("hunyuan", ...) instead of ("wan", ...)
  - forward kwargs: no encoder_attention_mask, no return_dict
  - default flow_shift = 7
"""

from __future__ import annotations

import copy
import os
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import (
    normalize_dit_input, )

from fastvideo.train.models.wan.wan import WanModel

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


class HunyuanModel(WanModel):
    """HunyuanVideo per-role model.

    Inherits most behaviour from WanModel (noise scheduler,
    timestep sampling, attention metadata, backward).  Overrides
    only the pieces that differ for Hunyuan.
    """

    _transformer_cls_name: str = "HunyuanVideoTransformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 7.0,
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
        """Same flow as Wan, but uses Hunyuan VAE normalisation."""
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
                "latent_channels",
                getattr(vae_config, "z_dim", 16),
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

        # KEY DIFFERENCE: "hunyuan" normalisation
        training_batch.latents = normalize_dit_input(
            "hunyuan",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        """Build transformer forward kwargs for Hunyuan.

        Unlike Wan, Hunyuan does not use encoder_attention_mask
        or return_dict in its forward signature.
        """
        if text_dict is None:
            raise ValueError("text_dict cannot be None for "
                             "Hunyuan forward pass")
        return {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "timestep": timestep,
        }

    def ensure_negative_conditioning(self) -> None:
        """Encode the negative prompt with dual text encoders
        (LLaMA + CLIP).

        Every rank encodes independently to avoid NCCL deadlocks
        when only a subset of ranks would otherwise participate.
        """
        if self.negative_prompt_embeds is not None:  # type: ignore[has-type]
            return

        assert self.training_config is not None
        tc = self.training_config
        device = self.device
        dtype = self._get_training_dtype()

        from transformers import (AutoTokenizer, CLIPTextModel, LlamaModel)

        from fastvideo.configs.pipelines.hunyuan import (
            clip_preprocess_text,
            clip_postprocess_text,
            llama_preprocess_text,
            llama_postprocess_text,
        )
        from fastvideo.utils import (PRECISION_TO_TYPE, maybe_download_model)

        model_path = maybe_download_model(tc.model_path)

        # Use configured precisions for each encoder.
        precisions = tc.pipeline_config.text_encoder_precisions
        llama_dtype = PRECISION_TO_TYPE[precisions[0]]
        clip_dtype = PRECISION_TO_TYPE[precisions[1]]

        # --- LLaMA ---
        llama_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        llama_enc = LlamaModel.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            torch_dtype=llama_dtype,
        ).to(device).eval()

        llama_cfg = tc.pipeline_config.text_encoder_configs[0]
        llama_tok_kwargs = dict(llama_cfg.tokenizer_kwargs)

        negative_prompt = ""
        llama_text = llama_preprocess_text(negative_prompt)

        with torch.no_grad():
            llama_inputs = llama_tok(llama_text, **llama_tok_kwargs).to(device)
            llama_out = llama_enc(**llama_inputs, output_hidden_states=True)
            llama_embeds = llama_postprocess_text(llama_out).squeeze(0)

        del llama_enc, llama_tok

        # --- CLIP ---
        clip_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer_2"))
        clip_enc = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder_2"),
            torch_dtype=clip_dtype,
        ).to(device).eval()

        clip_cfg = tc.pipeline_config.text_encoder_configs[1]
        clip_tok_kwargs = dict(clip_cfg.tokenizer_kwargs)
        clip_text = clip_preprocess_text(negative_prompt)

        with torch.no_grad():
            clip_inputs = clip_tok(clip_text, **clip_tok_kwargs).to(device)
            clip_out = clip_enc(**clip_inputs)
            clip_pooled = clip_postprocess_text(clip_out).squeeze(0)

        del clip_enc, clip_tok

        # --- Combine: [pooled_clip_row, llama_embeds] ---
        llama_dim = llama_embeds.shape[-1]
        pooled_row = torch.zeros(llama_dim, device=device)
        pooled_row[:clip_pooled.shape[-1]] = clip_pooled
        neg_embeds = torch.cat(
            [pooled_row.unsqueeze(0), llama_embeds],
            dim=0,
        ).unsqueeze(0).to(device=device, dtype=dtype)

        # Attention mask: all ones for the combined sequence.
        neg_mask = torch.ones(neg_embeds.shape[:2], device=device, dtype=dtype)

        self.negative_prompt_embeds = neg_embeds
        self.negative_prompt_attention_mask = neg_mask
