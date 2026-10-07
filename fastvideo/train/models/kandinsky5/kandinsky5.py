# SPDX-License-Identifier: Apache-2.0
"""Kandinsky5 model plugin (per-role instance).

Written directly against ``ModelBase`` (not a ``WanModel`` subclass) because
Kandinsky5 differs structurally in ways that don't compose cleanly with
Wan's implementation:

  - Dual text encoders: Qwen/Reason1 sequence embeddings *and* a CLIP pooled
    projection, vs. Wan's single encoder. The pair is packed into the
    existing single ``text_embedding`` parquet field by zero-padding the
    CLIP pooled vector into a row and prepending it to the Qwen sequence
    (the same trick already used by ``HunyuanModel``/
    ``preprocess_hunyuan_overfit.py`` for LLaMA+CLIP).
  - The transformer forward signature needs RoPE position tensors
    (``visual_rope_pos``, ``text_rope_pos``) and a ``scale_factor`` that Wan
    never has.
  - Kandinsky5's native hidden_states layout is channel-last
    ``[B, T, H, W, C]``; the common ModelBase convention used at the
    predict_noise/decode_latents boundary is ``[B, T, C, H, W]``.
  - VAE denorm uses only ``scaling_factor`` (Kandinsky5 shares Hunyuan's
    VAE, which has no ``latents_mean``/``latents_std``/
    ``handles_latent_denorm``).
  - flow_shift default is 5.0 (vs. Wan's 3.0).

Scope: T2V at 480p only, dense/local attention only (NABLA sparse attention
is never engaged -- ``sparse_params`` is always ``None``).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing, )
from fastvideo.training.training_utils import (
    compute_density_for_timestep_sampling,
    get_sigmas,
    normalize_dit_input,
    shift_timestep,
)

from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.module_state import (
    apply_trainable, )
from fastvideo.train.utils.moduleloader import (
    load_module_from_path,
    make_inference_args,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.train.utils.lora import LoraConfig

# 480p is the only supported resolution band for this recipe. Matches
# Kandinsky5DenoisingStage._scale_factor's low-resolution branch exactly;
# outside this band Kandinsky5 uses a different visual RoPE scale_factor
# ((1.0, 3.16, 3.16)) that this wrapper does not implement.
_SCALE_FACTOR_480P = (1.0, 2.0, 2.0)
_MIN_480P_SIDE = 480
_MAX_480P_SIDE = 854


class Kandinsky5Model(ModelBase):
    """Kandinsky5 per-role model: owns transformer + noise_scheduler."""

    _transformer_cls_name: str = "Kandinsky5Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 5.0,
        enable_gradient_checkpointing_type: str
        | None = None,
        transformer_override_safetensor: str
        | None = None,
        lora: LoraConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            trainable=trainable,
            lora=lora,
        )
        self._init_from = str(init_from)

        self.transformer = self._load_transformer(
            init_from=self._init_from,
            trainable=self._trainable,
            disable_custom_init_weights=(disable_custom_init_weights),
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            training_config=training_config,
            transformer_override_safetensor=(transformer_override_safetensor),
        )

        self.noise_scheduler = (FlowMatchEulerDiscreteScheduler(shift=float(flow_shift)))

        # Filled by init_preprocessors (student only).
        self.vae: Any = None
        self.training_config: TrainingConfig = training_config
        self.dataloader: Any = None
        self.validator: Any = None
        self.start_step: int = 0

        self.world_group: Any = None
        self.sp_group: Any = None

        # Qwen sequence embeds + mask, and CLIP pooled projection.
        self.negative_prompt_embeds: (torch.Tensor | None) = None
        self.negative_prompt_attention_mask: (torch.Tensor | None) = None
        self.negative_pooled_embeds: (torch.Tensor | None) = None
        self._requires_negative_conditioning = True

        # Timestep mechanics.
        self.timestep_shift: float = float(flow_shift)
        self.num_train_timestep: int = int(self.noise_scheduler.num_train_timesteps)
        self.min_timestep: int = 0
        self.max_timestep: int = self.num_train_timestep

    def _load_transformer(
        self,
        *,
        init_from: str,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        training_config: TrainingConfig,
        transformer_override_safetensor: str | None = None,
    ) -> torch.nn.Module:
        transformer = load_module_from_path(
            model_path=init_from,
            module_type="transformer",
            training_config=training_config,
            disable_custom_init_weights=(disable_custom_init_weights),
            override_transformer_cls_name=(self._transformer_cls_name),
            transformer_override_safetensor=(transformer_override_safetensor),
        )
        ckpt_type = (enable_gradient_checkpointing_type or getattr(
            getattr(training_config, "model", None),
            "enable_gradient_checkpointing_type",
            None,
        ))
        if trainable and ckpt_type:
            transformer = apply_activation_checkpointing(
                transformer,
                checkpointing_type=ckpt_type,
            )
        if self._enable_lora_if_configured(transformer):
            return transformer
        transformer = apply_trainable(transformer, trainable=trainable)
        return transformer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()

        self._init_timestep_mechanics()

        from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
        from fastvideo.train.utils.dataloader import (
            build_parquet_t2v_train_dataloader, )

        preprocessed_data_type = str(getattr(
            training_config.data,
            "preprocessed_data_type",
            "t2v",
        )).strip().lower()
        if preprocessed_data_type != "t2v":
            raise ValueError("Unsupported Kandinsky5 preprocessed_data_type: "
                             f"{preprocessed_data_type!r}")

        # Qwen's usable (post-template-trim) embedding length, +1 for the
        # prepended CLIP pooled row (see module docstring / prepare_batch).
        qwen_text_len = int(training_config.pipeline_config.text_encoder_configs[  # type: ignore[union-attr]
            0].arch_config.text_len)
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=qwen_text_len + 1,
            parquet_schema=pyarrow_schema_t2v,
        )
        self.start_step = 0

    @property
    def num_train_timesteps(self) -> int:
        return int(self.num_train_timestep)

    def set_requires_negative_conditioning(self, requires: bool) -> None:
        self._requires_negative_conditioning = bool(requires)

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = shift_timestep(
            timestep,
            self.timestep_shift,
            self.num_train_timestep,
        )
        return timestep.clamp(self.min_timestep, self.max_timestep)

    def on_train_start(self) -> None:
        if self._requires_negative_conditioning:
            self.ensure_negative_conditioning()

    @torch.no_grad()
    def decode_latents(
        self,
        latents_b_t_c_h_w: torch.Tensor,
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Kandinsky5 VAE is not initialized")
        latents = latents_b_t_c_h_w.permute(0, 2, 1, 3, 4).float()
        # Kandinsky5 shares Hunyuan's VAE: only a scalar scaling_factor, no
        # latents_mean/latents_std, no handles_latent_denorm. Inverse of the
        # encode-side `normalize_dit_input("hunyuan", ...)` multiply.
        denorm = latents / self.vae.scaling_factor
        media = self.vae.to(latents.device).decode(denorm)
        return (media / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # Runtime primitives
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        if self._requires_negative_conditioning:
            self.ensure_negative_conditioning()
        assert self.training_config is not None
        tc = self.training_config

        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()

        # Unpack the CLIP-pooled-row-prepended text_embedding field: row 0
        # is the zero-padded CLIP pooled vector, rows 1: are the Qwen
        # sequence embeddings. See module docstring.
        packed_embeds = raw_batch["text_embedding"]
        packed_mask = raw_batch["text_attention_mask"]
        pooled_dim = int(tc.pipeline_config.dit_config.arch_config.in_text_dim2  # type: ignore[union-attr]
                         )
        pooled_projections = packed_embeds[:, 0, :pooled_dim]
        qwen_embeds = packed_embeds[:, 1:, :]
        qwen_mask = packed_mask[:, 1:]
        infos = raw_batch.get("info_list")

        if latents_source == "zeros":
            batch_size = packed_embeds.shape[0]
            vae_config = (
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            # Read off the loaded transformer module, not
            # tc.pipeline_config.dit_config.arch_config: the latter carries
            # YAML/dataclass defaults and isn't synced with the actual
            # checkpoint's transformer/config.json (e.g. the shipped
            # Kandinsky5-Lite checkpoint has in_visual_dim=16, vs. the
            # dataclass default of 4).
            num_channels = int(self.transformer.in_visual_dim)
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
        training_batch.encoder_hidden_states = qwen_embeds.to(device, dtype=dtype)
        training_batch.encoder_attention_mask = qwen_mask.to(device, dtype=dtype)
        training_batch.infos = infos
        pooled_projections = pooled_projections.to(device, dtype=dtype)

        training_batch.latents = normalize_dit_input("hunyuan", training_batch.latents, self.vae)
        training_batch = self._prepare_dit_inputs(training_batch, generator, pooled_projections)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, t = clean_latents.shape[:2]
        noisy = self.noise_scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep,
        ).unflatten(0, (b, t))
        return noisy

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

        if noisy_latents.is_floating_point():
            noisy_latents = noisy_latents.to(dtype=dtype)

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
        ):
            input_kwargs = (self._build_distill_input_kwargs(noisy_latents, timestep, text_dict))
            transformer = self._get_transformer(timestep)
            out = transformer(**input_kwargs)
            sample = out.sample if hasattr(out, "sample") else out
            # Kandinsky5's native hidden_states layout is channel-last
            # [B, T, H, W, C]; convert to the common ModelBase convention
            # [B, T, C, H, W].
            pred_noise = sample.permute(0, 1, 4, 2, 3)
        return pred_noise

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        timesteps, attn_metadata = ctx
        with set_forward_context(
                current_timestep=timesteps,
                attn_metadata=attn_metadata,
        ):
            (loss / max(1, int(grad_accum_rounds))).backward()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_training_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _init_timestep_mechanics(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        self.timestep_shift = float(tc.pipeline_config.flow_shift  # type: ignore[union-attr]
                                    )
        self.num_train_timestep = int(self.noise_scheduler.num_train_timesteps)
        self.min_timestep = 0
        self.max_timestep = self.num_train_timestep

    def ensure_negative_conditioning(self) -> None:
        """Encode the negative prompt with dual text encoders (Qwen + CLIP).

        Every rank encodes independently (same rationale as Hunyuan's
        override: avoids NCCL deadlocks from asymmetric rank-0-only
        encoding). Cannot reuse ``encode_negative_prompt`` unmodified for
        the Qwen encoder: ``kandinsky5_qwen_postprocess_text`` requires the
        attention mask as a second positional argument and returns a
        ``(embeds, trimmed_mask)`` tuple, unlike the single-arg
        postprocess-func contract that helper assumes.
        """
        if self.negative_prompt_embeds is not None:
            return

        assert self.training_config is not None
        tc = self.training_config
        device = self.device
        dtype = self._get_training_dtype()

        from transformers import AutoTokenizer

        from fastvideo.configs.pipelines.base import preprocess_text
        from fastvideo.configs.pipelines.kandinsky5 import (
            kandinsky5_clip_postprocess_text,
            kandinsky5_qwen_postprocess_text,
            kandinsky5_qwen_preprocess_text,
        )
        from fastvideo.models.loader.component_loader import TextEncoderLoader
        from fastvideo.utils import maybe_download_model

        pipeline_config = tc.pipeline_config
        assert pipeline_config is not None
        model_path = maybe_download_model(tc.model_path)
        inference_args = make_inference_args(tc, model_path=model_path)
        inference_args.text_encoder_cpu_offload = False

        sampling_param = SamplingParam.from_pretrained(tc.model_path)
        negative_prompt = sampling_param.negative_prompt

        qwen_cfg = pipeline_config.text_encoder_configs[0]
        clip_cfg = pipeline_config.text_encoder_configs[1]
        loader = TextEncoderLoader()

        # --- Qwen / Reason1 ---
        qwen_enc = loader.load(
            os.path.join(model_path, "text_encoder"),
            inference_args,
        ).to(device).eval()
        qwen_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        qwen_tok_kwargs = dict(qwen_cfg.tokenizer_kwargs)
        qwen_text = kandinsky5_qwen_preprocess_text(negative_prompt)

        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            qwen_inputs = qwen_tok(qwen_text, **qwen_tok_kwargs).to(device)
            qwen_out = qwen_enc(
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                output_hidden_states=True,
            )
            qwen_embeds, qwen_mask = kandinsky5_qwen_postprocess_text(qwen_out, qwen_inputs.attention_mask)

        del qwen_enc, qwen_tok

        # --- CLIP ---
        clip_enc = loader.load(
            os.path.join(model_path, "text_encoder_2"),
            inference_args,
        ).to(device).eval()
        clip_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer_2"))
        clip_tok_kwargs = dict(clip_cfg.tokenizer_kwargs)
        clip_text = preprocess_text(negative_prompt)

        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            clip_inputs = clip_tok(clip_text, **clip_tok_kwargs).to(device)
            clip_out = clip_enc(
                input_ids=clip_inputs.input_ids,
                attention_mask=clip_inputs.attention_mask,
            )
            clip_pooled = kandinsky5_clip_postprocess_text(clip_out)

        del clip_enc, clip_tok

        self.negative_prompt_embeds = qwen_embeds.to(device=device, dtype=dtype)
        self.negative_prompt_attention_mask = qwen_mask.to(device=device, dtype=dtype)
        self.negative_pooled_embeds = clip_pooled.to(device=device, dtype=dtype)

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert self.training_config is not None
        tc = self.training_config

        u = compute_density_for_timestep_sampling(
            weighting_scheme=tc.model.weighting_scheme,
            batch_size=batch_size,
            generator=generator,
            device=device,
            logit_mean=tc.model.logit_mean,
            logit_std=tc.model.logit_std,
            mode_scale=tc.model.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        return self.noise_scheduler.timesteps[indices.cpu()].to(device=device)

    def _build_attention_metadata(self, training_batch: TrainingBatch) -> TrainingBatch:
        # Scope is 480p T2V with dense/local attention only -- VSA/VMOBA
        # sparse attention is never engaged for Kandinsky5 in this recipe.
        training_batch.attn_metadata = None
        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
        pooled_projections: torch.Tensor,
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]
        device = latents.device

        num_height = int(tc.data.num_height)
        num_width = int(tc.data.num_width)
        if not (_MIN_480P_SIDE <= num_height <= _MAX_480P_SIDE and _MIN_480P_SIDE <= num_width <= _MAX_480P_SIDE):
            raise ValueError("Kandinsky5Model only supports 480p training "
                             f"(height/width in [{_MIN_480P_SIDE}, {_MAX_480P_SIDE}]); "
                             f"got num_height={num_height}, num_width={num_width}. "
                             "A larger resolution needs a different visual RoPE "
                             "scale_factor, which this wrapper does not implement.")

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=device,
            dtype=latents.dtype,
        )
        timesteps = self._sample_timesteps(batch_size, device, generator)
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = ((1.0 - sigmas) * latents + sigmas * noise)

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        # Trim the Qwen sequence to the longest real (non-pad) sample in
        # this batch, mirroring Kandinsky5DenoisingStage._text_rope_pos.
        # Kandinsky5's cross-attention has no attention-mask plumbing at
        # all -- it always attends to the full encoder_hidden_states
        # sequence -- so any padding left in the tensor beyond this point
        # would silently leak into cross-attention. Batched *inference*
        # already relies on the same property (tokenizer padding="True"
        # pads only to the batch's longest real sample).
        qwen_embeds = training_batch.encoder_hidden_states
        qwen_mask = training_batch.encoder_attention_mask
        assert qwen_embeds is not None and qwen_mask is not None
        # bf16 cannot exactly accumulate sums beyond ~256, so count valid
        # tokens in float32 to avoid an off-by-a-few error on long prompts.
        valid_len = max(1, int(qwen_mask.float().sum(dim=1).max().item()))
        qwen_embeds = qwen_embeds[:, :valid_len]
        qwen_mask = qwen_mask[:, :valid_len]
        text_rope_pos = torch.arange(valid_len, device=device)

        patch_size = (
            tc.pipeline_config.dit_config.arch_config.patch_size  # type: ignore[union-attr]
        )
        spatial_compression_ratio = (
            tc.pipeline_config.vae_config.arch_config.spatial_compression_ratio  # type: ignore[union-attr]
        )
        latent_h = num_height // spatial_compression_ratio
        latent_w = num_width // spatial_compression_ratio
        visual_rope_pos = [
            torch.arange(int(tc.data.num_latent_t), device=device),
            torch.arange(latent_h // patch_size[1], device=device),
            torch.arange(latent_w // patch_size[2], device=device),
        ]

        training_batch.conditional_dict = {
            "encoder_hidden_states": qwen_embeds,
            "encoder_attention_mask": qwen_mask,
            "pooled_projections": pooled_projections,
            "text_rope_pos": text_rope_pos,
            "visual_rope_pos": visual_rope_pos,
            "scale_factor": _SCALE_FACTOR_480P,
            "sparse_params": None,
        }

        if (self.negative_prompt_embeds is not None and self.negative_prompt_attention_mask is not None
                and self.negative_pooled_embeds is not None):
            neg_embeds = self.negative_prompt_embeds
            neg_mask = self.negative_prompt_attention_mask
            neg_pooled = self.negative_pooled_embeds
            if neg_embeds.shape[0] == 1 and batch_size > 1:
                neg_embeds = neg_embeds.expand(batch_size, *neg_embeds.shape[1:]).contiguous()
            if neg_mask.shape[0] == 1 and batch_size > 1:
                neg_mask = neg_mask.expand(batch_size, *neg_mask.shape[1:]).contiguous()
            if neg_pooled.shape[0] == 1 and batch_size > 1:
                neg_pooled = neg_pooled.expand(batch_size, *neg_pooled.shape[1:]).contiguous()
            neg_text_rope_pos = torch.arange(neg_embeds.shape[1], device=device)
            training_batch.unconditional_dict = {
                "encoder_hidden_states": neg_embeds,
                "encoder_attention_mask": neg_mask,
                "pooled_projections": neg_pooled,
                "text_rope_pos": neg_text_rope_pos,
                "visual_rope_pos": visual_rope_pos,
                "scale_factor": _SCALE_FACTOR_480P,
                "sparse_params": None,
            }

        training_batch.latents = (training_batch.latents.permute(0, 2, 1, 3, 4))
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for "
                             "Kandinsky5 distillation")
        # noise_input is [B, T, C, H, W] (common ModelBase convention);
        # Kandinsky5's transformer expects channel-last [B, T, H, W, C].
        hidden_states = noise_input.permute(0, 1, 3, 4, 2)
        if getattr(self.transformer, "visual_cond", False):
            # The shipped Kandinsky5-Lite checkpoint has visual_cond=True
            # (unified T2V/I2V conditioning): Kandinsky5VisualEmbeddings
            # always expects [real_latent | zero_cond | zero_mask]
            # concatenated on the channel dim (2*in_visual_dim+1 total),
            # even for pure T2V with no actual image conditioning. Mirrors
            # Kandinsky5LatentPreparationStage's inference-time padding
            # (fastvideo/pipelines/stages/kandinsky5.py).
            cond = torch.zeros_like(hidden_states)
            mask = torch.zeros(*hidden_states.shape[:-1], 1, device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, cond, mask], dim=-1)
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "pooled_projections": text_dict["pooled_projections"],
            "timestep": timestep,
            "visual_rope_pos": text_dict["visual_rope_pos"],
            "text_rope_pos": text_dict["text_rope_pos"],
            "scale_factor": text_dict["scale_factor"],
            "sparse_params": text_dict["sparse_params"],
            "return_dict": True,
        }

    def _get_transformer(self, timestep: torch.Tensor) -> torch.nn.Module:
        return self.transformer

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if cfg_uncond is None:
            text_dict = getattr(batch, "unconditional_dict", None)
            if text_dict is None:
                raise RuntimeError("Missing unconditional_dict; "
                                   "ensure_negative_conditioning() "
                                   "may have failed")
            return text_dict

        on_missing_raw = cfg_uncond.get("on_missing", "error")
        if not isinstance(on_missing_raw, str):
            raise ValueError("method_config.cfg_uncond.on_missing "
                             "must be a string, got "
                             f"{type(on_missing_raw).__name__}")
        on_missing = on_missing_raw.strip().lower()
        if on_missing not in {"error", "ignore"}:
            raise ValueError("method_config.cfg_uncond.on_missing "
                             "must be one of {error, ignore}, got "
                             f"{on_missing_raw!r}")

        for channel, policy_raw in cfg_uncond.items():
            if channel in {"on_missing", "text"}:
                continue
            if policy_raw is None:
                continue
            if not isinstance(policy_raw, str):
                raise ValueError("method_config.cfg_uncond values "
                                 "must be strings, got "
                                 f"{channel}="
                                 f"{type(policy_raw).__name__}")
            policy = policy_raw.strip().lower()
            if policy == "keep":
                continue
            if on_missing == "ignore":
                continue
            raise ValueError("Kandinsky5Model does not support "
                             "cfg_uncond channel "
                             f"{channel!r} (policy={policy!r}). "
                             "Set cfg_uncond.on_missing=ignore or "
                             "remove the channel.")

        text_policy_raw = cfg_uncond.get("text", None)
        if text_policy_raw is None:
            text_policy = "negative_prompt"
        elif not isinstance(text_policy_raw, str):
            raise ValueError("method_config.cfg_uncond.text must be "
                             "a string, got "
                             f"{type(text_policy_raw).__name__}")
        else:
            text_policy = (text_policy_raw.strip().lower())

        if text_policy in {"negative_prompt"}:
            text_dict = getattr(batch, "unconditional_dict", None)
            if text_dict is None:
                raise RuntimeError("Missing unconditional_dict; "
                                   "ensure_negative_conditioning() "
                                   "may have failed")
            return text_dict
        if text_policy == "keep":
            if batch.conditional_dict is None:
                raise RuntimeError("Missing conditional_dict in "
                                   "TrainingBatch")
            return batch.conditional_dict
        if text_policy == "zero":
            if batch.conditional_dict is None:
                raise RuntimeError("Missing conditional_dict in "
                                   "TrainingBatch")
            cond = batch.conditional_dict
            enc = cond["encoder_hidden_states"]
            mask = cond["encoder_attention_mask"]
            pooled = cond["pooled_projections"]
            if not torch.is_tensor(enc) or not torch.is_tensor(mask) or not torch.is_tensor(pooled):
                raise TypeError("conditional_dict must contain "
                                "tensor text inputs")
            return {
                "encoder_hidden_states": torch.zeros_like(enc),
                "encoder_attention_mask": torch.zeros_like(mask),
                "pooled_projections": torch.zeros_like(pooled),
                "text_rope_pos": cond["text_rope_pos"],
                "visual_rope_pos": cond["visual_rope_pos"],
                "scale_factor": cond["scale_factor"],
                "sparse_params": cond["sparse_params"],
            }
        if text_policy == "drop":
            raise ValueError("cfg_uncond.text=drop is not supported "
                             "for Kandinsky5. Use "
                             "{negative_prompt, keep, zero}.")
        raise ValueError("cfg_uncond.text must be one of "
                         "{negative_prompt, keep, zero, drop}, got "
                         f"{text_policy_raw!r}")
