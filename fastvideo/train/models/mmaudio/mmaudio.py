# SPDX-License-Identifier: Apache-2.0
"""MMAudio model plugin for the modular FastVideo trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.configs.models.dits import (
    MMAUDIO_44K_TRAINING_VARIANTS,
    MMAUDIO_TRAINING_VARIANTS,
    MMAUDIO_VARIANT_ARCHITECTURES,
    get_mmaudio_transformer_config,
)
from fastvideo.configs.pipelines.mmaudio import (
    MMAUDIO_PIPELINE_CONFIGS,
    MMAudioV2AConfig,
)
from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.mmaudio import MMAudioTransformer
from fastvideo.models.loader.fsdp_load import build_fsdp_model_from_scratch
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.distributed_strategy import (
    build_replicated_model_from_scratch,
    is_ddp_strategy,
    unwrap_ddp_module,
    wrap_module_ddp,
)
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.utils import PRECISION_TO_TYPE

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)


def _load_empty_string_features(path: str | Path) -> torch.Tensor:
    """Load only the fixed CLIP empty-string embedding from an official asset."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"MMAudio empty-string feature source does not exist: {source}")

    if source.is_dir():
        preferred = (
            source / "transformer" / "diffusion_pytorch_model.safetensors",
            source / "diffusion_pytorch_model.safetensors",
        )
        candidates = [candidate for candidate in preferred if candidate.is_file()]
        if not candidates:
            candidates = sorted(source.rglob("*.safetensors"))
        for candidate in candidates:
            from safetensors import safe_open

            with safe_open(candidate, framework="pt", device="cpu") as handle:
                if "empty_string_feat" in handle:
                    return handle.get_tensor("empty_string_feat").float()
        raise ValueError(f"No 'empty_string_feat' tensor was found in safetensors under {source}")

    if source.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            if "empty_string_feat" not in handle:
                raise ValueError(f"MMAudio safetensors {source} has no 'empty_string_feat' tensor")
            return handle.get_tensor("empty_string_feat").float()

    state = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        tensor = state
    elif isinstance(state, dict):
        tensor = state.get("empty_string_feat")
        if tensor is None and isinstance(state.get("state_dict"), dict):
            tensor = state["state_dict"].get("empty_string_feat")
        if tensor is None:
            for key, value in state.items():
                if str(key).endswith("empty_string_feat") and isinstance(value, torch.Tensor):
                    tensor = value
                    break
    else:
        tensor = None
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"MMAudio asset {source} does not contain an empty-string tensor")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.float()


def _load_or_compute_latent_stats(
    training_config: TrainingConfig,
    *,
    variant: str,
    cache_path: str | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute/cache official latent statistics once, then share across ranks."""
    world_group = get_world_group()
    payload: dict[str, Any] | None = None
    if world_group.is_first_rank:
        stats_path = Path(cache_path).expanduser().resolve() if cache_path else None
        if stats_path is not None and stats_path.is_file():
            loaded = torch.load(stats_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid MMAudio latent statistics file: {stats_path}")
            payload = loaded
            logger.info("Loaded MMAudio latent statistics from %s", stats_path)
        else:
            from fastvideo.dataset.mmaudio_feature_dataset import (
                compute_mmaudio_latent_stats, )

            architecture = MMAUDIO_VARIANT_ARCHITECTURES[variant]
            logger.info("Computing MMAudio latent statistics from the first feature cache")
            latent_mean, latent_std = compute_mmaudio_latent_stats(
                training_config.data.data_path,
                latent_seq_len=int(architecture["latent_seq_len"]),
                latent_dim=int(architecture["latent_dim"]),
                chunk_size=chunk_size,
            )
            payload = {
                "variant": variant,
                "latent_mean": latent_mean,
                "latent_std": latent_std,
            }
            if stats_path is not None:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, stats_path)
                logger.info("Saved MMAudio latent statistics to %s", stats_path)

    payload = world_group.broadcast_object(payload, src=0)
    if not isinstance(payload, dict):
        raise RuntimeError("Failed to broadcast MMAudio latent statistics")
    cached_variant = payload.get("variant")
    same_44k_contract = (cached_variant in MMAUDIO_44K_TRAINING_VARIANTS and variant in MMAUDIO_44K_TRAINING_VARIANTS)
    if cached_variant is not None and cached_variant != variant and not same_44k_contract:
        raise ValueError(f"MMAudio latent statistics are for {cached_variant!r}, not {variant!r}")
    latent_mean = payload.get("latent_mean")
    latent_std = payload.get("latent_std")
    if not isinstance(latent_mean, torch.Tensor) or not isinstance(latent_std, torch.Tensor):
        raise ValueError("MMAudio latent statistics must contain tensor mean/std values")
    expected_shape = (1, 1, int(MMAUDIO_VARIANT_ARCHITECTURES[variant]["latent_dim"]))
    if tuple(latent_mean.shape) != expected_shape or tuple(latent_std.shape) != expected_shape:
        raise ValueError(f"MMAudio latent statistics for {variant!r} must have shape "
                         f"{expected_shape}, got {tuple(latent_mean.shape)} and "
                         f"{tuple(latent_std.shape)}")
    return latent_mean, latent_std


class MMAudioModel(ModelBase):
    """Official-compatible conditional flow-matching training adapter.

    Only the MMAudio transformer participates in the training graph. Audio VAE,
    DFN5B, and Synchformer outputs are read from an offline feature cache.
    """

    _transformer_cls_name = "MMAudioTransformer"

    def __init__(
        self,
        *,
        init_from: str | None = None,
        training_config: TrainingConfig,
        variant: str | None = None,
        from_scratch: bool = False,
        empty_string_features_path: str | None = None,
        latent_statistics_path: str | None = None,
        latent_statistics_chunk_size: int = 32,
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
        use_ddp = is_ddp_strategy(training_config)
        if use_ddp and (int(training_config.distributed.hsdp_replicate_dim) != 1
                        or int(training_config.distributed.hsdp_shard_dim) not in {-1, 1}):
            raise ValueError("MMAudio DDP strategy does not use HSDP dimensions; "
                             "set hsdp_replicate_dim=1 and hsdp_shard_dim=1")

        self.training_config = training_config
        if variant is not None:
            try:
                pipeline_config_cls = MMAUDIO_PIPELINE_CONFIGS[variant]
            except KeyError as exc:
                supported = ", ".join(MMAUDIO_PIPELINE_CONFIGS)
                raise ValueError(f"Unknown MMAudio variant {variant!r}; expected one of: {supported}") from exc
            self.training_config.pipeline_config = pipeline_config_cls()
        elif self.training_config.pipeline_config is None:
            self.training_config.pipeline_config = MMAudioV2AConfig()

        if transformer is None:
            if from_scratch:
                if init_from:
                    raise ValueError("MMAudio from_scratch=true cannot also set init_from; use "
                                     "empty_string_features_path for the fixed CLIP embedding")
                if variant is None:
                    raise ValueError("MMAudio from-scratch training requires variant")
                if variant not in MMAUDIO_TRAINING_VARIANTS:
                    supported = ", ".join(MMAUDIO_TRAINING_VARIANTS)
                    raise ValueError(f"The official MMAudio recipe does not train {variant!r}; "
                                     f"supported from-scratch variants: {supported}")
                if not empty_string_features_path:
                    raise ValueError("MMAudio from-scratch training requires "
                                     "empty_string_features_path")
                latent_mean, latent_std = _load_or_compute_latent_stats(
                    self.training_config,
                    variant=variant,
                    cache_path=latent_statistics_path,
                    chunk_size=latent_statistics_chunk_size,
                )
                empty_string_feat = _load_empty_string_features(empty_string_features_path)
                architecture = MMAUDIO_VARIANT_ARCHITECTURES[variant]
                expected_text_shape = (77, 1024)
                if tuple(empty_string_feat.shape) != expected_text_shape:
                    raise ValueError("MMAudio empty-string features must have shape "
                                     f"{expected_text_shape}, got {tuple(empty_string_feat.shape)}")
                model_config = get_mmaudio_transformer_config(variant)
                hf_config = {
                    "_class_name": self._transformer_cls_name,
                    **architecture,
                    "clip_dim": 1024,
                    "clip_seq_len": 64,
                    "sync_dim": 768,
                    "sync_seq_len": 192,
                    "text_dim": 1024,
                    "text_seq_len": 77,
                    "mlp_ratio": 4.0,
                }
                default_dtype = PRECISION_TO_TYPE[self.training_config.dit_precision]
                init_params = {
                    "config": model_config,
                    "hf_config": hf_config,
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "empty_string_feat": empty_string_feat,
                }
                if use_ddp:
                    transformer = build_replicated_model_from_scratch(
                        MMAudioTransformer,
                        init_params,
                        device=get_local_torch_device(),
                        default_dtype=default_dtype,
                        seed=self.training_config.data.seed,
                    )
                else:
                    transformer = build_fsdp_model_from_scratch(
                        model_cls=MMAudioTransformer,
                        init_params=init_params,
                        device=get_local_torch_device(),
                        hsdp_replicate_dim=self.training_config.distributed.hsdp_replicate_dim,
                        hsdp_shard_dim=self.training_config.distributed.hsdp_shard_dim,
                        default_dtype=default_dtype,
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        seed=self.training_config.data.seed,
                        pin_cpu_memory=self.training_config.distributed.pin_cpu_memory,
                    )
                self._init_from = f"scratch:{variant}"
            else:
                if not init_from:
                    raise ValueError("MMAudio pretrained training requires init_from, or set "
                                     "from_scratch=true with a variant")
                if use_ddp:
                    raise NotImplementedError("MMAudio DDP currently supports from_scratch=true only; "
                                              "pretrained component loading remains on the FSDP path")
                self._init_from = str(init_from)
                transformer = load_module_from_path(
                    model_path=self._init_from,
                    module_type="transformer",
                    training_config=self.training_config,
                    override_transformer_cls_name=self._transformer_cls_name,
                    attention_backend=self.attention_backend,
                )
        else:
            self._init_from = str(init_from or f"provided:{variant or 'unknown'}")
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
        if use_ddp:
            # Official MMAudio wraps the fully initialized FP32 model after
            # selecting trainable parameters. It disables buffer broadcasts;
            # the fixed latent statistics and positional buffers are identical
            # because every rank uses the same initialization seed.
            self.transformer = wrap_module_ddp(
                self.transformer,
                device=get_local_torch_device(),
                broadcast_buffers=False,
            )

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
        video_exists_expanded = video_exists[:, None, None]
        text_exists_expanded = text_exists[:, None, None]
        clip_features = torch.where(video_exists_expanded, clip_features, empty_clip)
        sync_features = torch.where(video_exists_expanded, sync_features, empty_sync)
        text_features = torch.where(text_exists_expanded, text_features, empty_text)

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
        null_video_expanded = null_video[:, None, None]
        clip_features = torch.where(null_video_expanded, empty_clip, clip_features)
        sync_features = torch.where(null_video_expanded, empty_sync, sync_features)
        null_text = torch.rand(batch_size, device=device, generator=generator) < null_probability
        null_text_expanded = null_text[:, None, None]
        text_features = torch.where(null_text_expanded, empty_text, text_features)

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

    def compile_training_forward(
        self,
        compile_kwargs: dict[str, Any],
    ) -> None:
        """Compile transformer math while leaving DDP control flow eager."""
        module = unwrap_ddp_module(self.transformer)
        compiled_forward = torch.compile(module.forward, **compile_kwargs)
        module.forward = compiled_forward
        logger.info(
            "Compiled inner MMAudio transformer forward with kwargs=%s",
            compile_kwargs,
        )

    def flow_matching_forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        clip_features: torch.Tensor,
        sync_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-only adapter around the DDP-wrapped training forward."""
        device_type = noisy_latents.device.type
        with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
        ):
            return self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states={
                    "clip_features": clip_features,
                    "sync_features": sync_features,
                    "text_features": text_features,
                },
                timestep=timestep,
            )

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

        with set_forward_context(current_timestep=timestep, attn_metadata=None):
            return self.flow_matching_forward(
                noisy_latents,
                timestep,
                conditions["clip_features"],
                conditions["sync_features"],
                conditions["text_features"],
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
