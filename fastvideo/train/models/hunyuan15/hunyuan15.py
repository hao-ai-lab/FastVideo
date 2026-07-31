# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo 1.5 model plugin (per-role instance).

Subclasses WanModel since HunyuanVideo 1.5 uses the same
FlowMatchEulerDiscreteScheduler and linear-interpolation noise
schedule.  Differences:
  - transformer class name: HunyuanVideo15Transformer3DModel
  - normalize_dit_input("hunyuan15", ...): latents * scaling_factor
  - dual text encoders: encoder_hidden_states is [qwen, byt5]
  - forward kwargs: no encoder_attention_mask / return_dict; an
    all-zero encoder_hidden_states_image selects the T2V branch
  - hidden_states in (B, C, T, H, W)
  - discrete timestep on the 0..1000 scale
  - default flow_shift = 5.0 (480p T2V)
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import os
import copy
import torch

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import (
    normalize_dit_input, )

logger = init_logger(__name__)

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


class Hunyuan15Model(WanModel):
    """HunyuanVideo 1.5 per-role model.

    Inherits most behaviour from WanModel (noise scheduler,
    timestep sampling, attention metadata, backward).  Overrides
    only the pieces that differ for HunyuanVideo 1.5.
    """

    _transformer_cls_name: str = "HunyuanVideo15Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 5.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        lora: LoraConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=(disable_custom_init_weights),
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            transformer_override_safetensor=(transformer_override_safetensor),
            lora=lora,
        )
        self.negative_prompt_embeds_2: torch.Tensor | None = None
        # No negative-prompt cache: FineTuneMethod hard-codes
        # conditional=True, so encoding the negative prompt would load the
        # ~16GB Qwen encoder on every rank for an embedding nothing reads.
        # DMD2 re-enables this from its own __init__ when its cfg_uncond
        # policy needs negative prompts. Mirrors ltx2.py, which opts out the
        # same way rather than load Gemma.
        self.set_requires_negative_conditioning(False)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Fix the dataloader text padding length before the base
        class builds the dataloader.

        Qwen2_5_VLArchConfig leaves ``text_len`` at the base-config
        default of 0, which would size the parquet dataloader's text
        padding to zero tokens.  The real padded length is the Qwen
        tokenizer max length minus the cropped template tokens
        (1108 - 108 = 1000), matching the preprocess output.
        """
        pipeline_config = training_config.pipeline_config
        assert pipeline_config is not None
        text_len = int(pipeline_config.text_encoder_max_lengths[0] - pipeline_config.text_encoder_crop_start)
        pipeline_config.text_encoder_configs[0].arch_config.text_len = (text_len)
        super().init_preprocessors(training_config)

    @torch.no_grad()
    def decode_latents(
        self,
        latents_b_t_c_h_w: torch.Tensor,
    ) -> torch.Tensor:
        """Decode HY1.5 latents back to pixels.

        The Wan path denormalises with per-channel ``latents_mean`` /
        ``latents_std``, which ``AutoencoderKLHunyuanVideo15`` does not
        expose: it normalises by a single scaling factor instead, so undo
        that rather than inheriting the Wan implementation.
        """
        if self.vae is None:
            raise RuntimeError("HunyuanVideo 1.5 VAE is not initialized")
        latents = latents_b_t_c_h_w.permute(0, 2, 1, 3, 4).float()
        if bool(getattr(self.vae, "handles_latent_denorm", False)):
            denorm = latents
        else:
            denorm = latents / self.vae.scaling_factor
        media = self.vae.to(latents.device).decode(denorm)
        return (media / 2 + 0.5).clamp(0, 1)

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Same flow as Wan, with three HY1.5 differences: dual text
        embeddings (Qwen + ByT5), trim-to-longest padding removal (the
        DiT has no text-mask input) and "hunyuan15" VAE scaling."""
        if self._requires_negative_conditioning:
            self.ensure_negative_conditioning()
        assert self.training_config is not None
        tc = self.training_config

        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        infos = raw_batch.get("info_list")
        batch_size = encoder_hidden_states.shape[0]

        # --- dual text embeddings -------------------------------------
        encoder_hidden_states_2 = raw_batch.get("text_embedding_2")
        encoder_attention_mask_2 = raw_batch.get("text_attention_mask_2")
        if (encoder_hidden_states_2 is None or encoder_attention_mask_2 is None):
            # Legacy parquet without the ByT5 field: zero-token
            # embedding, matching the inference empty-ByT5 convention.
            encoder_hidden_states_2 = torch.zeros(
                batch_size,
                0,
                1472,
                dtype=encoder_hidden_states.dtype,
            )
            encoder_attention_mask_2 = torch.zeros(
                batch_size,
                0,
                dtype=encoder_attention_mask.dtype,
            )

        # --- trim padding to the longest valid length in the batch ----
        # The HY1.5 DiT has no text-mask input; leftover padding would
        # dilute the token refiner's mean-pool. With B=1 this equals the
        # exact-length behaviour of the inference path.
        prompt_lengths = encoder_attention_mask.sum(dim=1)
        max_len = int(prompt_lengths.max().item())
        min_len = int(prompt_lengths.min().item())
        if min_len != max_len:
            logger.warning(
                "Batch mixes prompt lengths (%s..%s tokens). HY1.5's DiT takes "
                "no text mask and mean-pools every token it is given, so the "
                "shorter prompts keep padding and their conditioning depends "
                "on the longest prompt in the batch. Use train_batch_size=1 "
                "or bucket samples by prompt length.",
                min_len,
                max_len,
            )
        encoder_hidden_states = encoder_hidden_states[:, :max_len]
        encoder_attention_mask = encoder_attention_mask[:, :max_len]
        max_len_2 = int(encoder_attention_mask_2.sum(dim=1).max().item())
        encoder_hidden_states_2 = encoder_hidden_states_2[:, :max_len_2]
        encoder_attention_mask_2 = encoder_attention_mask_2[:, :max_len_2]

        # --- latents ---------------------------------------------------
        if latents_source == "zeros":
            vae_config = (
                tc.pipeline_config.vae_config  # type: ignore[union-attr]
                .arch_config)
            num_channels = getattr(vae_config, "latent_channels", 32)
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

        # KEY DIFFERENCE: "hunyuan15" normalisation (latents * 1.03682;
        # parquet stores raw latent_dist.mode() outputs).
        training_batch.latents = normalize_dit_input(
            "hunyuan15",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)

        # The base class only knows the primary text stream; attach the
        # ByT5 stream to the dicts consumed by _build_distill_input_kwargs.
        assert training_batch.conditional_dict is not None
        training_batch.conditional_dict["encoder_hidden_states_2"] = (encoder_hidden_states_2.to(device, dtype=dtype))
        if (training_batch.unconditional_dict is not None and self.negative_prompt_embeds_2 is not None):
            neg_embeds_2 = self.negative_prompt_embeds_2
            if neg_embeds_2.shape[0] == 1 and batch_size > 1:
                neg_embeds_2 = neg_embeds_2.expand(batch_size, *neg_embeds_2.shape[1:]).contiguous()
            training_batch.unconditional_dict["encoder_hidden_states_2"] = neg_embeds_2

        training_batch = self._build_attention_metadata(training_batch)

        # Shallow copy keeps the lru_cache'd LongTensor index fields shared
        # with the original metadata; only the float ``VSA_sparsity`` differs
        # between the two views.
        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        """Build transformer forward kwargs for HunyuanVideo 1.5.

        The HY1.5 transformer expects:
        - hidden_states in (B, C, T, H, W) — predict_noise permutes
        before calling this helper
        - encoder_hidden_states as a list of two tensors
        [qwen, byt5]; a zero-token ByT5 tensor (B, 0, 1472) is valid
        - timestep on the discrete 0..1000 scale
        - encoder_hidden_states_image as a one-element list holding an
        all-zero tensor — the all-zero content selects the T2V
        branch (is_t2v check); passing None crashes
        - no encoder_attention_mask / return_dict / guidance kwargs;
        timestep_r must stay unset while use_meanflow=False
        """
        if text_dict is None:
            raise ValueError("text_dict cannot be None for "
                             "HunyuanVideo 1.5 forward pass")

        batch_size = noise_input.shape[0]
        # HY1.5's img_in takes 65 channels: the 32 latent channels plus a
        # 1-channel conditioning mask and a 32-channel conditioning latent.
        # T2V leaves both conditioning blocks zero, mirroring
        # Hy15ImageEncodingStage + DenoisingStage on the inference path.
        cond_mask = torch.zeros(
            batch_size,
            1,
            *noise_input.shape[2:],
            device=noise_input.device,
            dtype=noise_input.dtype,
        )
        cond_latent = torch.zeros_like(noise_input)
        hidden_states = torch.cat([noise_input, cond_mask, cond_latent], dim=1)

        zero_image_embeds = torch.zeros(
            batch_size,
            729,
            1152,
            device=noise_input.device,
            dtype=noise_input.dtype,
        )
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": [
                text_dict["encoder_hidden_states"],
                text_dict["encoder_hidden_states_2"],
            ],
            "timestep": timestep,
            "encoder_hidden_states_image": [zero_image_embeds],
        }

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
        dtype = self._get_training_dtype()
        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in "
                                   "TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        # The method layer hands noisy_latents in (B, T, C, H, W) (Wan
        # canonical); HY1.5's transformer expects (B, C, T, H, W).
        noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)
        if noisy_latents.is_floating_point():
            noisy_latents = noisy_latents.to(dtype=dtype)

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents,
                timestep,
                text_dict,
            )
            transformer = self._get_transformer(timestep)
            model_output = transformer(**input_kwargs)

        # (B, C, T, H, W) → (B, T, C, H, W) back to the method layer.
        pred = model_output.permute(0, 2, 1, 3, 4)
        return pred

    def ensure_negative_conditioning(self) -> None:
        """Encode the HY1.5 negative prompt with the Qwen encoder.

        The shared ``encode_negative_prompt`` helper does not fit
        HY1.5: its Qwen preprocess returns a chat message list (which
        needs ``apply_chat_template``), its postprocess takes the
        attention mask as a second argument and returns a tuple, and
        it reads ``hidden_states[-3]`` so the encoder must run with
        ``output_hidden_states=True``.  Mirror the preprocess script's
        ``encode_qwen_caption`` instead, so training and the parquet
        data are produced by the same code path.

        Every rank encodes independently to avoid NCCL deadlocks when
        only a subset of ranks would otherwise participate.
        """
        if self.negative_prompt_embeds is not None:  # type: ignore[has-type]
            return
        assert self.training_config is not None
        tc = self.training_config
        pipeline_config = tc.pipeline_config
        assert pipeline_config is not None
        device = self.device
        dtype = self._get_training_dtype()

        from transformers import AutoTokenizer, Qwen2_5_VLTextModel

        from fastvideo.configs.pipelines.hunyuan15 import (
            qwen_postprocess_text,
            qwen_preprocess_text,
        )
        from fastvideo.utils import maybe_download_model

        model_path = maybe_download_model(tc.model_path)
        max_length = int(pipeline_config.text_encoder_max_lengths[0])

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        encoder = Qwen2_5_VLTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            torch_dtype=dtype,
        ).to(device).eval()

        # HY1.5 encodes an empty negative prompt as "." rather than the
        # empty string (treat_empty_as_dot on the inference path).
        messages = qwen_preprocess_text(".")

        with torch.no_grad():
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device)

            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            neg_embeds, neg_mask = qwen_postprocess_text(outputs, attention_mask)

            # Trim to the valid length: the DiT has no text-mask input,
            # matching the trim done in prepare_batch.
            valid_len = int(neg_mask.sum().item())
            neg_embeds = neg_embeds[:, :valid_len]
            neg_mask = neg_mask[:, :valid_len]

        del encoder, tokenizer

        self.negative_prompt_embeds = neg_embeds.to(device=device, dtype=dtype)
        self.negative_prompt_attention_mask = (neg_mask.to(device=device, dtype=dtype))
        # The negative prompt carries no glyph text: inference uses a
        # zero-length ByT5 tensor, not an encoded empty string.
        # 1472 is ByT5's d_model (the static T5ArchConfig default is
        # 512, so it cannot be derived from the config here).
        self.negative_prompt_embeds_2 = torch.zeros(
            1,
            0,
            1472,
            device=device,
            dtype=dtype,
        )
