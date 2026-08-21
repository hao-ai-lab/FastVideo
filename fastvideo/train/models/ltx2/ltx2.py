# SPDX-License-Identifier: Apache-2.0
"""LTX-2 model plugin (per-role instance).

Subclasses WanModel but replaces the pieces where LTX-2 differs:
  - transformer class name: LTX2Transformer3DModel (blocks nested
    under ``transformer.model``, so activation checkpointing must
    target the inner module)
  - latents come pre-normalized from the LTX-2 VAE encoder
    (per-channel statistics applied inside ``encode``), so no
    ``normalize_dit_input`` call
  - sigma sampling follows the official LTX-2 trainer: stretched
    shifted-logit-normal with a token-count-dependent shift
    (0.95 @ 1024 tokens -> 2.05 @ 4096 tokens) and a 10% uniform
    mixture, instead of scheduler-index sampling
  - the DiT consumes PER-TOKEN sigmas in [0, 1] shaped [B, tokens]
    (not integer 0-1000 timesteps) and returns the DENOISED x0
    prediction; ``predict_noise`` converts it back to the
    framework's velocity convention v = (x_t - x0) / sigma so the
    default FineTune target ``noise - clean`` applies unchanged
  - temporal RoPE coordinates are divided by fps read from
    ``get_forward_context().forward_batch.fps``; training must set
    it to the data fps or validation (which uses the preset fps)
    would see different temporal frequencies than training
  - the ~5.8B audio / cross-modal (a2v, v2a, av_ca) parameters are
    frozen: video-only forwards never give them
    gradients, and freezing keeps them out of DCP checkpoints and
    optimizer state
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

import fastvideo.envs as envs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import VideoLatentShape
from fastvideo.pipelines import ForwardBatch, TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing, )

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.module_state import (
    apply_trainable, )
from fastvideo.train.utils.moduleloader import (
    load_module_from_path, )

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.train.utils.lora import LoraConfig

logger = init_logger(__name__)

# Parameters whose names match any of these substrings belong to the
# audio branch or the audio<->video cross-modal machinery. They are
# unused (no gradients) in video-only forwards.
_AUDIO_PARAM_PATTERNS = ("audio", "a2v", "v2a", "av_ca")

# Official LTX-2 trainer timestep sampler constants
# (ltx_trainer/timestep_samplers.py: ShiftedLogitNormalTimestepSampler).
_SIGMA_MIN_TOKENS = 1024.0
_SIGMA_MAX_TOKENS = 4096.0
_SIGMA_MIN_SHIFT = 0.95
_SIGMA_MAX_SHIFT = 2.05
_SIGMA_STD = 1.0
_SIGMA_EPS = 1e-3
# 0.5% / 99.9% normal percentiles used to stretch the logit-normal
# samples so the sigma distribution covers the full [0, 1] range.
_SIGMA_Z_LO = -2.5758
_SIGMA_Z_HI = 3.0902

# Preset fps for LTX-2 checkpoints; used when a batch carries no fps
# metadata so training RoPE still matches validation inference.
_DEFAULT_ROPE_FPS = 24.0


class LTX2Model(WanModel):
    """LTX-2 per-role model for the modular trainer."""

    _transformer_cls_name: str = "LTX2Transformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        enable_gradient_checkpointing_type: str
        | None = None,
        transformer_override_safetensor: str
        | None = None,
        lora: LoraConfig | dict[str, Any] | None = None,
        attention_backend: AttentionBackendEnum | str | None = None,
        train_audio: bool = False,
        timestep_uniform_prob: float = 0.1,
    ) -> None:
        if train_audio:
            raise NotImplementedError("LTX2Model only supports video-only training; "
                                      "train_audio=True requires audio batch, forward, "
                                      "and loss plumbing.")

        cfg_rate = float(getattr(training_config.data, "training_cfg_rate", 0.0) or 0.0)
        if cfg_rate > 0.0:
            raise NotImplementedError("LTX2Model only supports training_cfg_rate=0. CFG dropout "
                                      "zeroes the post-connector text embeddings, which is not "
                                      "what LTX-2 inference uses as the unconditional input "
                                      "(an empty prompt encoded through Gemma + connector). Set "
                                      "training.data.training_cfg_rate: 0.0.")

        self._timestep_uniform_prob = float(timestep_uniform_prob)
        self._rope_fps: float = _DEFAULT_ROPE_FPS
        self._rope_forward_batch: ForwardBatch | None = None

        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=(disable_custom_init_weights),
            # The LTX-2 sigma sampler below does not use the scheduler;
            # keep an unshifted scheduler for the inherited distillation
            # helpers (add_noise / num_train_timesteps).
            flow_shift=1.0,
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            transformer_override_safetensor=(transformer_override_safetensor),
            lora=lora,
            attention_backend=attention_backend,
        )

        if trainable:
            self._freeze_audio_parameters()

        # No negative-prompt cache: cfg_rate is forced to 0 above and
        # loading Gemma (~23GB) on every rank just for an unused
        # negative embedding is wasteful.
        self.set_requires_negative_conditioning(False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_transformer(
        self,
        *,
        init_from: str,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        training_config: TrainingConfig,
        transformer_override_safetensor: str | None = None,
        attention_backend: AttentionBackendEnum | str | None = None,
    ) -> torch.nn.Module:
        transformer = load_module_from_path(
            model_path=init_from,
            module_type="transformer",
            training_config=training_config,
            disable_custom_init_weights=(disable_custom_init_weights),
            override_transformer_cls_name=(self._transformer_cls_name),
            transformer_override_safetensor=(transformer_override_safetensor),
            attention_backend=attention_backend,
        )
        ckpt_type = (enable_gradient_checkpointing_type or getattr(
            getattr(training_config, "model", None),
            "enable_gradient_checkpointing_type",
            None,
        ))
        if trainable and ckpt_type:
            # LTX-2 nests transformer_blocks under ``.model``; applying
            # checkpointing at the wrapper level raises because no block
            # list is found there.
            transformer.model = apply_activation_checkpointing(
                transformer.model,
                checkpointing_type=ckpt_type,
            )
        if self._enable_lora_if_configured(transformer):
            return transformer
        transformer = apply_trainable(transformer, trainable=trainable)
        return transformer

    def _freeze_audio_parameters(self) -> None:
        frozen_params = 0
        for name, param in self.transformer.named_parameters():
            if any(pattern in name for pattern in _AUDIO_PARAM_PATTERNS):
                param.requires_grad_(False)
                frozen_params += param.numel()
        logger.info(
            "LTX2Model: froze %.2fB audio/cross-modal parameters "
            "(video-only training)",
            frozen_params / 1e9,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_negative_conditioning(self) -> None:
        raise NotImplementedError("LTX2Model does not implement negative conditioning; "
                                  "training_cfg_rate must stay 0.")

    @torch.no_grad()
    def decode_latents(
        self,
        latents_b_t_c_h_w: torch.Tensor,
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("LTX-2 VAE is not initialized")
        latents = latents_b_t_c_h_w.permute(0, 2, 1, 3, 4)
        # The LTX-2 decoder un-normalizes internally via its
        # per-channel statistics buffers; no external denorm.
        media = self.vae.decode(latents.to(next(self.vae.parameters()).dtype))
        return (media.float() / 2 + 0.5).clamp(0, 1)

    def _init_timestep_mechanics(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        # LTX2T2VConfig carries no flow_shift (sigmas are computed
        # inline by the pipeline); the inherited shift/clamp helpers
        # are unused for fine-tuning, so default to identity shift.
        flow_shift = tc.pipeline_config.flow_shift  # type: ignore[union-attr]
        self.timestep_shift = (float(flow_shift) if flow_shift is not None else 1.0)
        self.num_train_timestep = int(self.noise_scheduler.num_train_timesteps)
        self.min_timestep = 0
        self.max_timestep = self.num_train_timestep

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
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            num_channels = getattr(
                vae_config,
                "z_dim",
                getattr(vae_config, "latent_channels", 128),
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

        self._check_text_embedding_dim(encoder_hidden_states)

        # LTX-2 VAE encode() already applies per-channel normalization
        # (scaling_factor is 1.0); latents are used as stored.
        training_batch.latents = latents
        training_batch.encoder_hidden_states = (encoder_hidden_states.to(device, dtype=dtype))
        training_batch.encoder_attention_mask = (encoder_attention_mask.to(device, dtype=dtype))
        training_batch.infos = infos

        self._update_rope_fps(infos)

        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)
        training_batch.attn_metadata_vsa = None

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
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clean_x is not None or aug_t is not None:
            raise NotImplementedError("LTX2Model does not support teacher forcing inputs")
        device_type = self.device.type
        dtype = self._get_training_dtype()
        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in "
                                   "TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)

        if attn_kind not in ("dense", "vsa"):
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")
        attn_metadata = batch.attn_metadata

        assert batch.sigmas is not None
        # finetune.py hands noisy_latents in (B, T, C, H, W); the LTX-2
        # DiT expects (B, C, T, H, W).
        noisy_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype)

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
                forward_batch=self._make_rope_forward_batch(),
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_bcthw,
                timestep,
                text_dict,
            )
            transformer = self._get_transformer(timestep)
            denoised = transformer(**input_kwargs)

        if isinstance(denoised, tuple):
            denoised = denoised[0]

        # The wrapper returns the denoised x0 prediction
        # (sample - velocity * sigma). Convert back to velocity so the
        # framework's default target ``noise - clean`` applies. Compute
        # in fp32: sigma can be ~1e-3 and bf16 division would amplify
        # quantization error.
        sigmas = batch.sigmas.float()
        velocity = (noisy_bcthw.float() - denoised.float()) / sigmas
        return velocity.permute(0, 2, 1, 3, 4)

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        timesteps, attn_metadata = ctx
        # Re-enter the forward context with the same fps-carrying batch
        # so activation-checkpoint recompute sees identical RoPE inputs.
        with set_forward_context(
                current_timestep=timesteps,
                attn_metadata=attn_metadata,
                forward_batch=self._make_rope_forward_batch(),
        ):
            (loss / max(1, int(grad_accum_rounds))).backward()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_text_embedding_dim(self, encoder_hidden_states: torch.Tensor) -> None:
        """Fail fast when the parquet was preprocessed with a mismatched
        text stack (LTX-2.0 stores 3840-d post-connector embeddings; 2.3
        has no in-DiT caption projection and expects 4096-d)."""
        # Read the LOADED transformer's config: the transformer loader
        # deepcopies pipeline_config.dit_config before applying the
        # checkpoint's arch flags, so training_config.pipeline_config
        # never sees version flags like caption_proj_before_connector.
        arch = self.transformer.config.arch_config
        if bool(getattr(arch, "caption_proj_before_connector", False)):
            expected_dim = int(arch.cross_attention_dim)
        else:
            expected_dim = int(arch.caption_channels)
        actual_dim = int(encoder_hidden_states.shape[-1])
        if actual_dim != expected_dim:
            raise ValueError(f"text_embedding width {actual_dim} does not match the "
                             f"checkpoint's expected text context dim {expected_dim}. "
                             "The parquet was likely preprocessed with a different "
                             "LTX-2 version (2.0 stores 3840-d, 2.3 stores 4096-d); "
                             "re-run preprocess_ltx2_overfit.py with LTX2_OVERFIT_MODEL "
                             "set to the same checkpoint as models.student.init_from.")

    def _update_rope_fps(self, infos: list[dict[str, Any]] | None) -> None:
        fps: float | None = None
        if infos:
            raw_fps = infos[0].get("fps")
            if raw_fps:
                fps = float(raw_fps)
        if fps is None or fps <= 0:
            fps = _DEFAULT_ROPE_FPS
        if self._rope_forward_batch is None or fps != self._rope_fps:
            self._rope_fps = fps
            self._rope_forward_batch = ForwardBatch(data_type="video", fps=fps)

    def _make_rope_forward_batch(self) -> ForwardBatch:
        if self._rope_forward_batch is None:
            self._rope_forward_batch = ForwardBatch(
                data_type="video",
                fps=self._rope_fps,
            )
        return self._rope_forward_batch

    def _sample_ltx2_sigmas(
        self,
        batch_size: int,
        token_count: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Official LTX-2 stretched shifted-logit-normal sigma sampler.

        mu is linearly interpolated from the patchified video token
        count (unclamped, matching the official trainer), the
        logit-normal samples are stretched to cover [0, 1] using the
        0.5%/99.9% percentiles, and 10% of samples are replaced by
        U(eps, 1).
        """
        slope = ((_SIGMA_MAX_SHIFT - _SIGMA_MIN_SHIFT) / (_SIGMA_MAX_TOKENS - _SIGMA_MIN_TOKENS))
        mu = slope * float(token_count) + (_SIGMA_MIN_SHIFT - slope * _SIGMA_MIN_TOKENS)

        normal = torch.randn(
            (batch_size, ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ) * _SIGMA_STD + mu
        sigmas = torch.sigmoid(normal)

        lo = torch.sigmoid(torch.tensor(mu + _SIGMA_Z_LO * _SIGMA_STD, device=device))
        hi = torch.sigmoid(torch.tensor(mu + _SIGMA_Z_HI * _SIGMA_STD, device=device))
        raw = (sigmas - lo) / (hi - lo)
        stretched = torch.where(raw >= _SIGMA_EPS, raw, 2 * _SIGMA_EPS - raw)
        stretched = stretched.clamp(0.0, 1.0)

        if self._timestep_uniform_prob > 0.0:
            prob = torch.rand(
                (batch_size, ),
                generator=generator,
                device=device,
            )
            uniform = torch.rand(
                (batch_size, ),
                generator=generator,
                device=device,
            ) * (1.0 - _SIGMA_EPS) + _SIGMA_EPS
            stretched = torch.where(
                prob > self._timestep_uniform_prob,
                stretched,
                uniform,
            )
        return stretched

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]

        video_shape = VideoLatentShape.from_torch_shape(latents.shape)
        patchifier = getattr(self.transformer, "patchifier", None)
        if patchifier is not None:
            token_count = int(patchifier.get_token_count(video_shape))
        else:
            token_count = int(latents.shape[2] * latents.shape[3] * latents.shape[4])
        self._token_count = token_count

        sigmas = self._sample_ltx2_sigmas(
            batch_size,
            token_count,
            latents.device,
            generator,
        )
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(sigmas, src=0)

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = ((1.0 - sigmas_expanded) * latents.float() + sigmas_expanded * noise.float()).to(
            latents.dtype)

        training_batch.noisy_model_input = noisy_model_input
        # Keep the framework-wide "timesteps ~ sigma * 1000" convention
        # (used for logging and the forward context); sigmas stay fp32
        # for the velocity conversion.
        training_batch.timesteps = sigmas * 1000.0
        training_batch.sigmas = sigmas_expanded
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        training_batch.conditional_dict = {
            "encoder_hidden_states": (training_batch.encoder_hidden_states),
            "encoder_attention_mask": (training_batch.encoder_attention_mask),
        }

        training_batch.latents = (training_batch.latents.permute(0, 2, 1, 3, 4))
        return training_batch

    def _build_attention_metadata(self, training_batch: TrainingBatch) -> TrainingBatch:
        if envs.FASTVIDEO_ATTENTION_BACKEND in ("VIDEO_SPARSE_ATTN", "VMOBA_ATTN"):
            raise NotImplementedError("LTX2Model does not support VSA/VMOBA attention backends")
        training_batch.attn_metadata = None
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for LTX-2 forward")
        if clean_x is not None or aug_t is not None:
            raise NotImplementedError("LTX2Model does not support teacher forcing inputs")

        # noise_input arrives already in (B, C, T, H, W).
        batch_size = noise_input.shape[0]
        token_count = getattr(self, "_token_count", None)
        if token_count is None:
            token_count = int(noise_input.shape[2] * noise_input.shape[3] * noise_input.shape[4])

        # The DiT wants per-token sigmas in [0, 1] shaped [B, tokens]
        # (i2v-style conditioning would zero conditioned tokens; plain
        # T2V uses the same sigma everywhere). ``timestep`` follows the
        # framework's sigma*1000 convention.
        sigma = (timestep.to(torch.float32) / 1000.0).view(batch_size, 1)
        per_token_timestep = sigma.expand(batch_size, token_count).contiguous()

        return {
            "hidden_states": noise_input,
            "encoder_hidden_states": (text_dict["encoder_hidden_states"]),
            # Post-connector embeddings are all-valid; the connector
            # replaced pad positions with learnable registers.
            "encoder_attention_mask": None,
            "timestep": per_token_timestep,
            "return_dict": False,
        }
