# SPDX-License-Identifier: Apache-2.0
"""MMAudio model plugin for the modular FastVideo trainer."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.configs.pipelines.mmaudio import MMAudioV2AConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig


class MMAudioModel(ModelBase):
    """Official-compatible conditional flow-matching training adapter.

    Only the MMAudio transformer participates in the training graph. Audio VAE,
    DFN5B, and Synchformer outputs are read from an offline feature cache.
    """

    _transformer_cls_name = "MMAudioTransformer"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        allow_v2_training: bool = False,
        lora: LoraConfig | dict[str, Any] | None = None,
        attention_backend: AttentionBackendEnum | str | None = None,
        transformer: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(
            trainable=trainable,
            lora=lora,
            attention_backend=attention_backend,
        )
        if int(training_config.distributed.sp_size or 1) != 1:
            raise ValueError("MMAudio training does not yet support sequence parallelism; set sp_size=1")
        if int(training_config.distributed.tp_size or 1) != 1:
            raise ValueError("MMAudio training does not yet support tensor parallelism; set tp_size=1")

        self._init_from = str(init_from)
        self.training_config = training_config
        if self.training_config.pipeline_config is None:
            self.training_config.pipeline_config = MMAudioV2AConfig()

        if transformer is None:
            transformer = load_module_from_path(
                model_path=self._init_from,
                module_type="transformer",
                training_config=self.training_config,
                override_transformer_cls_name=self._transformer_cls_name,
                attention_backend=self.attention_backend,
            )
        self.transformer = transformer
        if bool(getattr(self.transformer, "v2", False)) and not allow_v2_training:
            raise ValueError("The official MMAudio training recipe does not support `_v2` "
                             "checkpoints. Use small_44k, medium_44k, or large_44k.")

        if not self._enable_lora_if_configured(self.transformer):
            self.transformer = apply_trainable(self.transformer, trainable=self._trainable)
        if self._trainable:
            # These are checkpoint statistics/fixed text features in the
            # official model, not optimization variables. ``apply_trainable``
            # intentionally enables a whole module, so restore their contract.
            for name in ("latent_mean", "latent_std", "empty_string_feat"):
                parameter = getattr(self.transformer, name, None)
                if isinstance(parameter, torch.nn.Parameter):
                    parameter.requires_grad_(False)
            # The two learned null-video tokens are trained by MMAudio.
            for name in ("empty_clip_feat", "empty_sync_feat"):
                parameter = getattr(self.transformer, name, None)
                if isinstance(parameter, torch.nn.Parameter):
                    parameter.requires_grad_(True)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            invert_sigmas=True,
            sigma_min=0.0,
            use_reference_discrete_timesteps=True,
        )
        self.dataloader: Any = None
        self.start_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def _unshard_transformer(self) -> None:
        unshard = getattr(self.transformer, "unshard", None)
        if callable(unshard):
            unshard()

    def eager_optimizer_state_parameters(self, ) -> tuple[torch.nn.Parameter, ...]:
        parameters = (
            getattr(self.transformer, "empty_clip_feat", None),
            getattr(self.transformer, "empty_sync_feat", None),
        )
        return tuple(parameter for parameter in parameters
                     if isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        if training_config.data.preprocessed_data_type != "mmaudio_features":
            raise ValueError("MMAudioModel requires "
                             "training.data.preprocessed_data_type=mmaudio_features")
        from fastvideo.dataset.mmaudio_feature_dataset import (
            build_mmaudio_feature_dataloader, )

        feature_shapes = {
            "latent_seq_len": int(self.transformer.latent_seq_len),
            "latent_dim": int(self.transformer.latent_dim),
            "clip_seq_len": int(self.transformer.clip_seq_len),
            "clip_dim": int(self.transformer.config.arch_config.clip_dim),
            "sync_seq_len": int(self.transformer.sync_seq_len),
            "sync_dim": int(self.transformer.config.arch_config.sync_dim),
            "text_seq_len": int(self.transformer.config.arch_config.text_seq_len),
            "text_dim": int(self.transformer.config.arch_config.text_dim),
        }
        self.dataloader = build_mmaudio_feature_dataloader(
            training_config.data.data_path,
            batch_size=training_config.data.train_batch_size,
            num_data_workers=training_config.data.dataloader_num_workers,
            seed=training_config.data.seed,
            pin_memory=training_config.distributed.pin_cpu_memory,
            feature_shapes=feature_shapes,
        )

    @staticmethod
    def _batch_tensor(raw_batch: dict[str, Any], primary: str, alias: str | None = None) -> torch.Tensor:
        value = raw_batch.get(primary)
        if value is None and alias is not None:
            value = raw_batch.get(alias)
        if not isinstance(value, torch.Tensor):
            names = primary if alias is None else f"{primary!r} (or {alias!r})"
            raise ValueError(f"MMAudio training batch is missing tensor {names}")
        return value

    @staticmethod
    def _existence_mask(
        raw_batch: dict[str, Any],
        primary: str,
        alias: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = raw_batch.get(primary, raw_batch.get(alias))
        if value is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=torch.bool).reshape(-1)
        if value.numel() != batch_size:
            raise ValueError(f"{primary} must have {batch_size} values, got {value.numel()}")
        return value

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        # MMAudio uses transformer-owned latent statistics and learned null
        # tokens while constructing the batch, before ``transformer.forward``
        # would normally trigger FSDP2's pre-forward all-gather. Explicitly
        # unshard here so these tensors can be broadcast/expanded safely. The
        # following transformer forward performs the normal reshard.
        self._unshard_transformer()

        device = self.device
        latent_mean = self._batch_tensor(raw_batch, "audio_latent_mean", "a_mean").to(device=device)
        latent_std = self._batch_tensor(raw_batch, "audio_latent_std", "a_std").to(device=device)
        clip_features = self._batch_tensor(raw_batch, "clip_features").to(device=device).clone()
        sync_features = self._batch_tensor(raw_batch, "sync_features").to(device=device).clone()
        text_features = self._batch_tensor(raw_batch, "text_features").to(device=device).clone()

        if latent_mean.shape != latent_std.shape:
            raise ValueError("MMAudio audio latent mean/std shapes must match, got "
                             f"{tuple(latent_mean.shape)} and {tuple(latent_std.shape)}")
        batch_size = int(latent_mean.shape[0])
        expected_latent_shape = (
            batch_size,
            int(self.transformer.latent_seq_len),
            int(self.transformer.latent_dim),
        )
        if tuple(latent_mean.shape) != expected_latent_shape:
            raise ValueError(f"MMAudio audio latents must have shape {expected_latent_shape}, "
                             f"got {tuple(latent_mean.shape)}")

        video_exists = self._existence_mask(
            raw_batch,
            "video_exists",
            "video_exist",
            batch_size=batch_size,
            device=device,
        )
        text_exists = self._existence_mask(
            raw_batch,
            "text_exists",
            "text_exist",
            batch_size=batch_size,
            device=device,
        )
        empty_clip = self.transformer.get_empty_clip_sequence(1).to(dtype=clip_features.dtype)
        empty_sync = self.transformer.get_empty_sync_sequence(1).to(dtype=sync_features.dtype)
        empty_text = self.transformer.get_empty_string_sequence(1).to(dtype=text_features.dtype)
        clip_features[~video_exists] = empty_clip
        sync_features[~video_exists] = empty_sync
        text_features[~text_exists] = empty_text

        if latents_source == "data":
            posterior_noise = torch.empty_like(latent_mean).normal_(generator=generator)
            clean_latents = latent_mean + latent_std * posterior_noise
            clean_latents = self.transformer.normalize(clean_latents)
        elif latents_source == "zeros":
            clean_latents = torch.zeros_like(latent_mean)
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        logit_mean = float(self.training_config.model.logit_mean)
        logit_std = float(self.training_config.model.logit_std)
        timestep = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator) * logit_std + logit_mean)
        prior_noise = torch.empty_like(clean_latents).normal_(generator=generator)
        timestep_expanded = timestep[:, None, None]
        noisy_latents = (1.0 - timestep_expanded) * prior_noise + timestep_expanded * clean_latents

        null_probability = float(self.training_config.data.training_cfg_rate)
        if not 0.0 <= null_probability <= 1.0:
            raise ValueError("training.data.training_cfg_rate must be in [0, 1]")
        null_video = torch.rand(batch_size, device=device, generator=generator) < null_probability
        clip_features[null_video] = empty_clip
        sync_features[null_video] = empty_sync
        null_text = torch.rand(batch_size, device=device, generator=generator) < null_probability
        text_features[null_text] = empty_text

        # Keep learned null-video tokens in every step's autograd graph even
        # when this batch samples no null conditions. AdamW then creates a
        # stable optimizer-state schema, which is required for strict DCP
        # checkpoint resume. Multiplication by zero leaves forward values
        # unchanged; selected null conditions still receive their real grads.
        clip_features = clip_features + empty_clip.sum() * 0.0
        sync_features = sync_features + empty_sync.sum() * 0.0

        batch = TrainingBatch()
        batch.latents = clean_latents
        batch.noisy_model_input = noisy_latents
        batch.noise = prior_noise
        batch.timesteps = timestep
        batch.sigmas = timestep_expanded
        batch.training_target = clean_latents - prior_noise
        batch.raw_latent_shape = tuple(clean_latents.shape)
        batch.conditional_dict = {
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_features": text_features,
        }
        return batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        while timestep.ndim < clean_latents.ndim:
            timestep = timestep.unsqueeze(-1)
        return (1.0 - timestep) * noise + timestep * clean_latents

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
        if attn_kind != "dense":
            raise ValueError("MMAudio training currently supports dense Torch SDPA only")
        if conditional:
            conditions = batch.conditional_dict
            if conditions is None:
                raise RuntimeError("MMAudio TrainingBatch is missing conditional_dict")
        elif cfg_uncond is not None:
            conditions = cfg_uncond
        else:
            self._unshard_transformer()
            batch_size = noisy_latents.shape[0]
            conditions = {
                "clip_features": self.transformer.get_empty_clip_sequence(batch_size),
                "sync_features": self.transformer.get_empty_sync_sequence(batch_size),
                "text_features": self.transformer.get_empty_string_sequence(batch_size),
            }

        device_type = self.device.type
        with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
        ), set_forward_context(current_timestep=timestep, attn_metadata=None):
            return self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=conditions,
                timestep=timestep,
            )

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        timestep, attn_metadata = ctx
        with set_forward_context(current_timestep=timestep, attn_metadata=attn_metadata):
            (loss / max(1, int(grad_accum_rounds))).backward()


__all__ = ["MMAudioModel"]
