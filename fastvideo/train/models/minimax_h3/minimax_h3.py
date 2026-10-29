# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 joint text-to-video-and-audio training plugin."""

from __future__ import annotations

import copy
import dataclasses
import math
from numbers import Integral
from typing import Any, Literal, TYPE_CHECKING

import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn_h3 import (
    MiniMaxH3VSAMetadataBuilder, )
from fastvideo.dataset.shape_bucket import parse_video_shape_bucket_id
from fastvideo.dataset.sequence_capacity import select_sequence_capacity, validate_fixed_sequence_capacities
from fastvideo.distributed import get_sp_group
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.pdd import shifted_noise_amount
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedBatchLayout,
    MiniMaxH3PackedLayout,
    audio_latent_num_frames,
    build_packed_batch_sequence,
    build_packed_sequence,
    build_row_timesteps,
    pad_packed_sequence,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
    _h3_vsa_single_region_segments, )
from fastvideo.platforms import AttentionBackendEnum

from fastvideo.train.models.base import ModelBase, NoisePrediction
from fastvideo.train.utils.activation_checkpoint import (activation_offload_context, apply_activation_checkpointing,
                                                         resolve_checkpointing_type)
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.train.utils.lora import finalize_lora_training, LoraConfig

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

# Released H3 checkpoints map one shared denoising stage through these
# modality-specific scheduler shifts. Pipeline config may explicitly override
# them for a scientific treatment, while existing recipes retain 12/3.
_VIDEO_SCHEDULER_SHIFT = 12.0
_AUDIO_SCHEDULER_SHIFT = 3.0
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32
_AUDIO_SAMPLE_RATE = 32_000

# Dense TORCH_SDPA is the default; per-role overrides allow FLASH_ATTN
# (teacher/critic, FA4 via FASTVIDEO_FA4=1) and the packed-sequence VSA-H3
# backend (distillation student).
_ALLOWED_ATTENTION_BACKENDS = (
    AttentionBackendEnum.TORCH_SDPA,
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
)
# The H3 trajectory grid: integer labels 0..T, sigma = shift*u/(1+(shift-1)u)
# with u = label/T. TDM's denoising steps are labels on this grid.
_H3_TRAJECTORY_TIMESTEPS = 1000
_H3_TDM_MODALITIES = ("video", "audio")


def shift_noise_amount(
    base_noise_amount: torch.Tensor,
    shift: float,
    *,
    max_noise_amount: float = 1.0,
) -> torch.Tensor:
    """Apply the MiniMax H3 rational shift on ``[0, max_noise_amount]``."""
    if not 0.0 < max_noise_amount <= 1.0:
        raise ValueError("max_noise_amount must satisfy 0 < max <= 1, "
                         f"got {max_noise_amount}")
    return shifted_noise_amount(base_noise_amount, shift, max_t=max_noise_amount)


def tdm_h3_shift(modality: str) -> float:
    """Scheduler shift for one H3 modality."""
    if modality == "video":
        return _VIDEO_SCHEDULER_SHIFT
    if modality == "audio":
        return _AUDIO_SCHEDULER_SHIFT
    raise ValueError(f"unknown H3 TDM modality: {modality!r}")


def tdm_h3_sigma_grid(timesteps: torch.Tensor, modality: str) -> torch.Tensor:
    """Map integer H3 trajectory labels to sigmas for one modality."""
    shift = tdm_h3_shift(modality)
    u = timesteps.to(dtype=torch.float32) / float(_H3_TRAJECTORY_TIMESTEPS)
    return shift * u / (1.0 + (shift - 1.0) * u)


def tdm_h3_model_timestep(sigma: torch.Tensor) -> torch.Tensor:
    """H3 model time is ``1 - sigma`` in ``[0, 1]`` (``1`` is clean)."""
    return 1.0 - sigma.to(dtype=torch.float32)


def _broadcast_sigma(sigma: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[B]``/scalar sigma to ``latents``' trailing dimensions."""
    sigma = sigma.to(device=latents.device, dtype=latents.dtype)
    if sigma.ndim >= latents.ndim:
        return sigma
    return sigma.reshape(*sigma.shape, *([1] * (latents.ndim - sigma.ndim)))


class MiniMaxH3Model(ModelBase):
    """Adapt the H3 joint transformer to the modular fine-tuning contract."""

    _transformer_cls_name = "MiniMaxH3Transformer3DModel"
    _transformer_module_type = "transformer"
    export_diffusers_weight_layout = True
    _supported_preprocessed_data_types = ("t2va", "text_only")
    _supported_attention_backends = _ALLOWED_ATTENTION_BACKENDS
    _supports_cross_sample_sequence_packing = False
    _sequence_packing_requires_prepared_noise = True
    # Ref2VA installs a larger final layout after the target-only preparation.
    _defer_sequence_capacity = False

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        attention_backend: AttentionBackendEnum | str | None = AttentionBackendEnum.TORCH_SDPA,
        lora: LoraConfig | dict[str, Any] | None = None,
        freeze_adaln: bool = False,
    ) -> None:
        """Validate this role's single-document contract and load its transformer."""
        super().__init__(
            trainable=trainable,
            lora=lora,
            attention_backend=attention_backend,
        )
        # Attention layers bind their backend during construction, so this is
        # the per-role selection point (student/teacher/critic can differ).
        if self.attention_backend not in self._supported_attention_backends:
            allowed = ", ".join(b.name for b in self._supported_attention_backends)
            raise ValueError(f"{type(self).__name__} supports the attention backends "
                             f"{{{allowed}}}, got {self.attention_backend}")
        if training_config.pipeline_config is None:
            raise ValueError("MiniMaxH3Model requires a resolved MiniMax H3 pipeline config")
        # Packed row indices describe one text-video-audio document without a
        # batch offset, so each data-parallel replica consumes one sample.
        if int(training_config.data.train_batch_size) != 1:
            raise ValueError("MiniMaxH3Model requires training.data.train_batch_size=1")
        # Classifier-free guidance (CFG) dropout replaces text embeddings with
        # zeros, but H3 training does not define a zero-vector branch.
        if float(training_config.data.training_cfg_rate) != 0.0:
            raise ValueError("MiniMaxH3Model requires training.data.training_cfg_rate=0.0")
        # Each concrete role declares its supported preprocessed contract. The
        # base T2VA role accepts paired latents or text-only simulated rollout;
        # Ref2VA uses a distinct subclass and schema.
        data_type = str(training_config.data.preprocessed_data_type)
        if data_type not in self._supported_preprocessed_data_types:
            allowed_data = ", ".join(repr(item) for item in self._supported_preprocessed_data_types)
            raise ValueError(f"{type(self).__name__} requires training.data.preprocessed_data_type "
                             f"to be one of {{{allowed_data}}}, got {data_type!r}")

        self._init_from = str(init_from)
        self.training_config = training_config
        self.transformer = self._load_transformer(
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=resolve_checkpointing_type(enable_gradient_checkpointing_type,
                                                                          training_config),
            transformer_override_safetensor=transformer_override_safetensor,
        )
        if freeze_adaln:
            frozen = self._freeze_adaln_parameters(self.transformer)
            if frozen == 0:
                raise RuntimeError("freeze_adaln=true found no MiniMax H3 AdaLN parameters")
        pipeline_config = training_config.pipeline_config
        video_shift = getattr(pipeline_config, "video_scheduler_shift", None)
        audio_shift = getattr(pipeline_config, "audio_scheduler_shift", None)
        video_scheduler_shift = float(_VIDEO_SCHEDULER_SHIFT if video_shift is None else video_shift)
        audio_scheduler_shift = float(_AUDIO_SCHEDULER_SHIFT if audio_shift is None else audio_shift)
        self.noise_scheduler = MiniMaxH3Scheduler(shift=video_scheduler_shift)
        self.audio_noise_scheduler = MiniMaxH3Scheduler(shift=audio_scheduler_shift)
        self.dataloader: Any = None
        self.validator: Any = None
        self.start_step = 0
        self.sp_group: Any = None

    @staticmethod
    def _freeze_adaln_parameters(transformer: torch.nn.Module) -> int:
        """Freeze the per-block AdaLN projections used by FastGen's fake score."""
        frozen = 0
        for name, parameter in transformer.named_parameters():
            if "adaln" in name.lower():
                parameter.requires_grad_(False)
                frozen += 1
        return frozen

    @property
    def video_scheduler_shift(self) -> float:
        """Effective video noise warp owned by this model role."""
        scheduler = getattr(self, "noise_scheduler", None)
        return float(getattr(scheduler, "shift", _VIDEO_SCHEDULER_SHIFT))

    @property
    def audio_scheduler_shift(self) -> float:
        """Effective audio noise warp owned by this model role."""
        scheduler = getattr(self, "audio_noise_scheduler", None)
        return float(getattr(scheduler, "shift", _AUDIO_SCHEDULER_SHIFT))

    def _load_transformer(
        self,
        *,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        transformer_override_safetensor: str | None,
    ) -> torch.nn.Module:
        """Load H3 through the training FSDP loader and apply block checkpointing."""
        training_config = self.training_config
        arch_overrides = self._dit_arch_overrides()
        lora_enabled = self._lora_config is not None and self._lora_config.enable
        if arch_overrides or lora_enabled:
            # Only this role sees these settings; the shared config keeps the
            # released architecture and dtype policy for the other roles and
            # validation, whatever order the roles load in.
            pipeline_config = copy.deepcopy(training_config.pipeline_config)
            for name, value in arch_overrides.items():
                setattr(pipeline_config.dit_config.arch_config, name, value)
            if lora_enabled:
                # Preserve the released LoRA training dtype policy. Full-weight
                # roles retain the model's FP32 projection groups.
                pipeline_config.dit_config.uniform_parameter_dtype = True
            training_config = dataclasses.replace(training_config, pipeline_config=pipeline_config)
        pre_fsdp_model_transform = None
        if lora_enabled:

            def _prepare_lora(transformer: torch.nn.Module) -> None:
                enabled = self._enable_lora_if_configured(
                    transformer,
                    prepare_for_fsdp=True,
                    initialization_seed=int(self.training_config.data.seed or 0),
                )
                if not enabled:
                    raise RuntimeError("Failed to prepare MiniMax H3 LoRA before FSDP sharding")

            pre_fsdp_model_transform = _prepare_lora

        transformer = load_module_from_path(
            model_path=self._init_from,
            module_type=self.transformer_module_type,
            training_config=training_config,
            disable_custom_init_weights=disable_custom_init_weights,
            override_transformer_cls_name=self._transformer_cls_name,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=self.attention_backend,
            pre_fsdp_model_transform=pre_fsdp_model_transform,
        )
        if trainable:
            # No-grad rollout forwards must observe gate updates too. Keep the
            # branch static for these roles rather than probing weights or
            # mutating the inference cache inside compiled transformer blocks.
            for module in transformer.modules():
                if hasattr(module, "_gate_compress_training"):
                    module._gate_compress_training = True
        checkpointing_type = enable_gradient_checkpointing_type
        if trainable and checkpointing_type:
            transformer = apply_activation_checkpointing(
                transformer,
                checkpointing_type=checkpointing_type,
            )
        if self._lora_config is not None and self._lora_config.enable:
            finalized_layers = finalize_lora_training(transformer)
            if finalized_layers != self._num_lora_layers:
                raise RuntimeError("MiniMax H3 LoRA layer count changed during FSDP loading: "
                                   f"{self._num_lora_layers} -> {finalized_layers}")
            return transformer
        return apply_trainable(transformer, trainable=trainable)

    def _dit_arch_overrides(self) -> dict[str, Any]:
        """``arch_config`` fields this role loads its transformer with (none by default)."""
        return {}

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Load precomputed text embeddings and paired video-audio latents."""
        from fastvideo.dataset.dataloader.schema import (pyarrow_schema_t2va, pyarrow_schema_text_only)
        from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader

        data_config = training_config.data
        packing = getattr(data_config, "cross_sample_sequence_packing", None)
        if bool(getattr(packing, "enabled", False)) and not self._supports_cross_sample_sequence_packing:
            raise ValueError("Cross-sample H3 sequence packing is currently implemented only by "
                             "MiniMaxH3DMDModel")
        self.sp_group = get_sp_group()
        max_presentation_tokens = getattr(data_config, "max_presentation_tokens", None)
        text_len: int | None = None
        presentation_bound: int | None = None
        if max_presentation_tokens is None and not bool(getattr(data_config, "native_shape_bucketing", False)):
            # Unset keeps the fixed text-length contract of existing T2VA
            # configs: crop/pad each prompt to the text encoder length rather
            # than rejecting longer rows mid-run.
            text_len = int(training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len)
        elif (isinstance(max_presentation_tokens, bool) or not isinstance(max_presentation_tokens, Integral)
              or int(max_presentation_tokens) <= 0):
            raise ValueError("MiniMax H3 T2VA/text-only training requires training.data.max_presentation_tokens "
                             "to be a positive integer safety bound")
        else:
            presentation_bound = int(max_presentation_tokens)
        parquet_schema = (pyarrow_schema_text_only
                          if str(data_config.preprocessed_data_type) == "text_only" else pyarrow_schema_t2va)
        dataloader_kwargs: dict[str, Any] = {
            "text_len": text_len,
            "parquet_schema": parquet_schema,
            "max_presentation_tokens": presentation_bound,
        }
        if bool(getattr(packing, "enabled", False)):
            dataloader_kwargs["gradient_accumulation_steps"] = int(
                getattr(getattr(training_config, "loop", None), "gradient_accumulation_steps", 1) or 1)
        self.dataloader = build_parquet_t2v_train_dataloader(data_config, **dataloader_kwargs)
        self.start_step = 0

    def _resolve_clean_latents(
        self,
        raw_batch: dict[str, Any],
        latents_source: Literal["data", "zeros"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve fixed or native visual and stereo-audio latents."""
        data_config = self.training_config.data
        native_shapes = bool(getattr(data_config, "native_shape_bucketing", False))
        if latents_source == "data":
            if "vae_latent" not in raw_batch or "audio_latent" not in raw_batch:
                raise ValueError("A latent-bearing H3 batch requires vae_latent and audio_latent tensors")
            video_latents = raw_batch["vae_latent"]
            audio_latents = raw_batch["audio_latent"]
        elif latents_source == "zeros":
            latent_frames = int(data_config.num_latent_t)
            height = int(data_config.num_height)
            width = int(data_config.num_width)
            num_frames = int(data_config.num_frames)
            if native_shapes:
                bucket_id = raw_batch.get("_shape_bucket_id")
                if not isinstance(bucket_id, str):
                    raise ValueError(
                        "Native-shape data-free batches require the exact-shape sampler to set _shape_bucket_id")
                bucket = parse_video_shape_bucket_id(bucket_id)
                width = bucket.width
                height = bucket.height
                num_frames = bucket.num_frames
                latent_frames = video_latent_num_frames(num_frames)
                if width % MINIMAX_H3_CANVAS_MULTIPLE or height % MINIMAX_H3_CANVAS_MULTIPLE:
                    raise ValueError(f"Native pixel geometry {width}x{height} must use the H3 canvas multiple "
                                     f"{MINIMAX_H3_CANVAS_MULTIPLE}")
                latent_geometry = (latent_frames, height // 16, width // 16)
                patch_size = tuple(int(value) for value in self.transformer.patch_size)
                if any(value % patch for value, patch in zip(latent_geometry, patch_size, strict=True)):
                    raise ValueError(
                        f"Native latent geometry {latent_geometry} is not divisible by transformer patch {patch_size}")
            video_latents = torch.zeros(
                1,
                _VIDEO_LATENT_CHANNELS,
                latent_frames,
                height // 16,
                width // 16,
            )
            audio_latents = torch.zeros(
                1,
                MINIMAX_H3_AUDIO_CHANNELS,
                _AUDIO_LATENT_CHANNELS,
                audio_latent_num_frames(num_frames),
            )
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        if not isinstance(video_latents, torch.Tensor):
            raise ValueError(f"vae_latent must be a tensor, got {type(video_latents).__name__}")
        if not isinstance(audio_latents, torch.Tensor):
            raise ValueError(f"audio_latent must be a tensor, got {type(audio_latents).__name__}")
        if video_latents.ndim != 5 or tuple(video_latents.shape[:2]) != (1, _VIDEO_LATENT_CHANNELS):
            raise ValueError("vae_latent must have shape [1, 24, latent_frames, latent_height, latent_width], "
                             f"got {tuple(video_latents.shape)}")
        if audio_latents.ndim != 4 or tuple(audio_latents.shape[:3]) != (
                1,
                MINIMAX_H3_AUDIO_CHANNELS,
                _AUDIO_LATENT_CHANNELS,
        ):
            raise ValueError("audio_latent must have shape [1, 2, 32, audio_frames], "
                             f"got {tuple(audio_latents.shape)}")

        if latents_source == "data" and native_shapes:
            self._validate_native_latents(raw_batch, video_latents, audio_latents)
        elif not native_shapes:
            # Preserve the legacy fixed-shape contract for configs that have
            # not opted into exact-shape bucketing.
            if data_config.num_latent_t > 0:
                video_latents = video_latents[:, :, :data_config.num_latent_t]
            expected_audio_frames = audio_latent_num_frames(data_config.num_frames)
            audio_latents = audio_latents[:, :, :, :expected_audio_frames]
            if video_latents.shape[2] != data_config.num_latent_t:
                raise ValueError("vae_latent contains fewer frames than training.data.num_latent_t")
            if audio_latents.shape[-1] != expected_audio_frames:
                raise ValueError("audio_latent length does not match training.data.num_frames")
        return (
            video_latents.to(device=device, dtype=dtype),
            audio_latents.to(device=device, dtype=dtype),
        )

    def _validate_native_latents(
        self,
        raw_batch: dict[str, Any],
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> None:
        """Cross-check the canonical bucket, row metadata, and latent clocks."""
        bucket_id = raw_batch.get("_shape_bucket_id")
        if not isinstance(bucket_id, str):
            raise ValueError("Native-shape T2VA batches require the exact-shape sampler to set _shape_bucket_id")
        bucket = parse_video_shape_bucket_id(bucket_id)

        infos = raw_batch.get("info_list")
        if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
            raise ValueError("Native-shape T2VA batches require exactly one info_list metadata record")
        info = infos[0]

        def _metadata_int(name: str) -> int:
            value = info.get(name)
            if value is None or isinstance(value, bool):
                raise ValueError(f"T2VA metadata {name!r} must be a positive integer, got {value!r}")
            try:
                result = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"T2VA metadata {name!r} must be a positive integer, got {value!r}") from exc
            if result <= 0 or result != value:
                raise ValueError(f"T2VA metadata {name!r} must be a positive integer, got {value!r}")
            return result

        width = _metadata_int("width")
        height = _metadata_int("height")
        num_frames = _metadata_int("num_frames")
        audio_sample_rate = _metadata_int("audio_sample_rate")
        if (width, height, num_frames) != (bucket.width, bucket.height, bucket.num_frames):
            raise ValueError(f"Shape bucket {bucket_id!r} disagrees with row metadata "
                             f"width={width}, height={height}, num_frames={num_frames}")
        if audio_sample_rate != _AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"MiniMax H3 T2VA audio must be encoded at {_AUDIO_SAMPLE_RATE} Hz, got {audio_sample_rate}")
        fps_value = info.get("fps")
        if fps_value is None:
            raise ValueError("T2VA metadata 'fps' must be numeric, got None")
        try:
            fps = float(fps_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T2VA metadata 'fps' must be numeric, got {info.get('fps')!r}") from exc
        if not math.isfinite(fps) or not math.isclose(fps, 24.0, rel_tol=0.0, abs_tol=0.05):
            raise ValueError(f"MiniMax H3 T2VA video must use the 24 fps clock, got {fps}")

        if width % MINIMAX_H3_CANVAS_MULTIPLE or height % MINIMAX_H3_CANVAS_MULTIPLE:
            raise ValueError(f"Native pixel geometry {width}x{height} must use the H3 canvas multiple "
                             f"{MINIMAX_H3_CANVAS_MULTIPLE}")
        expected_video_shape = (
            1,
            _VIDEO_LATENT_CHANNELS,
            video_latent_num_frames(num_frames),
            height // 16,
            width // 16,
        )
        expected_audio_shape = (
            1,
            MINIMAX_H3_AUDIO_CHANNELS,
            _AUDIO_LATENT_CHANNELS,
            audio_latent_num_frames(num_frames),
        )
        if tuple(video_latents.shape) != expected_video_shape:
            raise ValueError(f"vae_latent shape does not match {bucket_id!r}: expected {expected_video_shape}, "
                             f"got {tuple(video_latents.shape)}")
        if tuple(audio_latents.shape) != expected_audio_shape:
            raise ValueError(f"audio_latent shape does not match the {num_frames}-frame H3 audio clock: "
                             f"expected {expected_audio_shape}, got {tuple(audio_latents.shape)}")
        latent_geometry = (expected_video_shape[2], expected_video_shape[3], expected_video_shape[4])
        patch_size = tuple(int(value) for value in self.transformer.patch_size)
        if any(value % patch for value, patch in zip(latent_geometry, patch_size, strict=True)):
            raise ValueError(
                f"Native latent geometry {latent_geometry} is not divisible by transformer patch {patch_size}")

    def _sample_noise_amounts(
        self,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one shared denoising stage and apply both modality shifts."""
        base_noise_amount = torch.rand(
            (1, ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        if int(self.training_config.distributed.sp_size) > 1:
            # Sequence-parallel ranks shard one document and therefore require
            # identical video and audio noise amounts for that document.
            self.sp_group.broadcast(base_noise_amount, src=0)
        return (
            shift_noise_amount(base_noise_amount, self.video_scheduler_shift),
            shift_noise_amount(base_noise_amount, self.audio_scheduler_shift),
        )

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Prepare normalized T2VA latents, shifted noise, text, and row layout."""
        packed_samples = raw_batch.get("_packed_samples")
        if packed_samples is not None:
            if not self._supports_cross_sample_sequence_packing:
                raise ValueError(f"{type(self).__name__} does not support cross-sample sequence packing; "
                                 "use fixed_sequence_capacities with one document instead")
            if not isinstance(packed_samples, list) or not packed_samples:
                raise ValueError("_packed_samples must contain at least one singleton H3 batch")
            prepared = tuple(
                self._prepare_single_batch(
                    sample,
                    generator=generator,
                    latents_source=latents_source,
                    build_attention_metadata=False,
                    build_noisy_inputs=self._sequence_packing_requires_prepared_noise,
                ) for sample in packed_samples)
            if len(prepared) == 1:
                batch = prepared[0]
                batch.sequence_packing_num_documents = 1
                layout = batch.minimax_h3_layout
                if not isinstance(layout, MiniMaxH3PackedLayout):
                    raise RuntimeError("A singleton packed envelope must retain one H3 document layout")
                declared_tokens = int(raw_batch.get("_packed_joint_token_count", layout.sequence_length))
                if declared_tokens != layout.sequence_length:
                    raise ValueError("Packed sampler token accounting disagrees with the prepared H3 sequence: "
                                     f"{declared_tokens} != {layout.sequence_length}")
                batch.sequence_packing_num_tokens = declared_tokens
                batch.sequence_packing_loss_scale = float(raw_batch.get("_sequence_packing_loss_scale", 1.0))
                self._apply_sequence_capacity(batch, raw_batch)
                self._maybe_build_dense_packed_metadata(batch)
                self._maybe_build_vsa_metadata(batch)
                return batch
            return self._combine_prepared_documents(prepared, raw_batch)
        return self._prepare_single_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
            build_attention_metadata=True,
            build_noisy_inputs=True,
        )

    def _prepare_single_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"],
        build_attention_metadata: bool,
        build_noisy_inputs: bool,
    ) -> TrainingBatch:
        """Prepare one H3 document; cross-sample packing composes this contract."""
        dtype = torch.bfloat16
        device = self.device
        text_embedding = raw_batch["text_embedding"]
        text_attention_mask = raw_batch["text_attention_mask"]
        if text_embedding.ndim != 3 or text_embedding.shape[0] != 1 or text_embedding.shape[-1] != 5120:
            raise ValueError(f"text_embedding must have shape [1, length, 5120], got {tuple(text_embedding.shape)}")
        if text_attention_mask.shape != text_embedding.shape[:2]:
            raise ValueError("text_attention_mask must match the text embedding sequence axes")
        valid_text = text_attention_mask[0].to(torch.bool)
        if not bool(valid_text.any()):
            raise ValueError("A T2VA batch requires at least one text token")

        video_latents, audio_latents = self._resolve_clean_latents(
            raw_batch,
            latents_source,
            dtype,
            device,
        )
        if build_noisy_inputs:
            video_noise = torch.randn(video_latents.shape, generator=generator, device=device, dtype=dtype)
            audio_noise = torch.randn(audio_latents.shape, generator=generator, device=device, dtype=dtype)
            video_noise_amount, audio_noise_amount = self._sample_noise_amounts(generator, device)
            video_sigmas = video_noise_amount.to(dtype).view(1, 1, 1, 1, 1)
            audio_sigmas = audio_noise_amount.to(dtype).view(1, 1, 1, 1)
            noisy_video_latents = (1.0 - video_sigmas) * video_latents + video_sigmas * video_noise
            noisy_audio_latents = (1.0 - audio_sigmas) * audio_latents + audio_sigmas * audio_noise
        else:
            # DMD2 constructs its own noise at explicit rollout/score times.
            # Avoid allocating two additional ragged latent copies for every
            # document solely to discard them after semantic packing.
            video_noise = None
            audio_noise = None
            video_sigmas = None
            audio_sigmas = None
            noisy_video_latents = None
            noisy_audio_latents = None
            video_noise_amount = torch.ones(1, device=device, dtype=torch.float32)
            audio_noise_amount = torch.ones(1, device=device, dtype=torch.float32)

        _, _, video_frames, latent_height, latent_width = video_latents.shape
        num_audio_latents = audio_latents.shape[-1]
        text_token_tags = torch.full((int(valid_text.sum()), ), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        # H3 self-attention consumes one interleaved document, so the layout
        # owns the row tags, positions, and modality output indices together.
        layout = build_packed_sequence(
            text_token_tags,
            video_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.transformer.patch_size,
        )

        training_batch = TrainingBatch()
        training_batch.latents = video_latents.permute(0, 2, 1, 3, 4)
        training_batch.audio_latents = audio_latents
        training_batch.encoder_hidden_states = text_embedding[:, valid_text].to(device=device, dtype=dtype)
        training_batch.encoder_attention_mask = torch.ones(
            (1, int(valid_text.sum())),
            device=device,
            dtype=dtype,
        )
        training_batch.infos = raw_batch.get("info_list")
        training_batch.raw_latent_shape = tuple(video_latents.shape)
        training_batch.noisy_model_input = noisy_video_latents
        training_batch.audio_noisy_model_input = noisy_audio_latents
        training_batch.noise = video_noise
        training_batch.audio_noise = audio_noise
        training_batch.sigmas = video_sigmas
        training_batch.audio_sigmas = audio_sigmas
        # ModelBase exposes clean-time timesteps while the loss consumes the
        # complementary noise amounts stored in the sigma fields.
        training_batch.timesteps = 1.0 - video_noise_amount
        training_batch.audio_timesteps = 1.0 - audio_noise_amount
        training_batch.minimax_h3_layout = layout
        training_batch.attn_metadata = None
        training_batch.attn_metadata_vsa = None
        if build_attention_metadata:
            if not self._defer_sequence_capacity:
                self._apply_sequence_capacity(training_batch, raw_batch)
                self._maybe_build_dense_packed_metadata(training_batch)
            self._maybe_build_vsa_metadata(training_batch)
        return training_batch

    def _combine_prepared_documents(
        self,
        documents: tuple[TrainingBatch, ...],
        raw_batch: dict[str, Any],
    ) -> TrainingBatch:
        """Compose singleton T2VA batches while retaining ragged latent geometry."""
        layouts = tuple(document.minimax_h3_layout for document in documents)
        if not all(isinstance(layout, MiniMaxH3PackedLayout) for layout in layouts):
            raise RuntimeError("Every packed H3 document must carry a singleton layout")
        if any(document.latents is None or document.audio_latents is None for document in documents):
            raise RuntimeError("Every packed H3 document must carry clean video and audio latents")
        if any(document.encoder_hidden_states is None for document in documents):
            raise RuntimeError("Every packed H3 document must carry text embeddings")

        packed_layout = build_packed_batch_sequence(layouts)  # type: ignore[arg-type]
        batch = TrainingBatch()
        batch.encoder_hidden_states = torch.cat(
            [document.encoder_hidden_states for document in documents if document.encoder_hidden_states is not None],
            dim=1,
        )
        batch.encoder_attention_mask = torch.ones(
            (1, batch.encoder_hidden_states.shape[1]),
            device=batch.encoder_hidden_states.device,
            dtype=batch.encoder_hidden_states.dtype,
        )
        batch.infos = [info for document in documents for info in (document.infos or [])]
        batch.timesteps = torch.cat([document.timesteps for document in documents if document.timesteps is not None])
        batch.audio_timesteps = torch.cat(
            [document.audio_timesteps for document in documents if document.audio_timesteps is not None])
        batch.minimax_h3_layout = packed_layout
        batch.minimax_h3_document_video_latents = tuple(document.latents for document in documents
                                                        if document.latents is not None)
        batch.minimax_h3_document_audio_latents = tuple(document.audio_latents for document in documents
                                                        if document.audio_latents is not None)
        batch.sequence_packing_num_documents = len(documents)
        declared_tokens = int(raw_batch.get("_packed_joint_token_count", packed_layout.sequence_length))
        if declared_tokens != packed_layout.sequence_length:
            raise ValueError("Packed sampler token accounting disagrees with the prepared H3 sequence: "
                             f"{declared_tokens} != {packed_layout.sequence_length}")
        batch.sequence_packing_num_tokens = declared_tokens
        batch.sequence_packing_loss_scale = float(raw_batch.get("_sequence_packing_loss_scale", 1.0))
        batch.attn_metadata = None
        batch.attn_metadata_vsa = None
        self._apply_sequence_capacity(batch, raw_batch)
        self._maybe_build_dense_packed_metadata(batch)
        self._maybe_build_vsa_metadata(batch)
        return batch

    def _apply_sequence_capacity(self, batch: TrainingBatch, raw_batch: dict[str, Any]) -> None:
        """Pad only the final joint DiT stream, never the target rollout state."""
        capacities = validate_fixed_sequence_capacities(
            getattr(self.training_config.data, "fixed_sequence_capacities", None))
        if not capacities:
            if raw_batch.get("_fixed_sequence_capacity") is not None:
                raise ValueError("Sampler supplied a fixed sequence capacity but the model config did not")
            return
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout | MiniMaxH3PackedBatchLayout):
            raise RuntimeError("Capacity padding requires a final H3 joint layout")
        real_tokens = layout.real_sequence_length
        capacity = select_sequence_capacity(real_tokens, capacities)
        declared_capacity = raw_batch.get("_fixed_sequence_capacity", capacity)
        declared_tokens = raw_batch.get("_packed_joint_token_count", real_tokens)
        if declared_capacity != capacity or declared_tokens != real_tokens:
            raise ValueError("Fixed-capacity sampler accounting disagrees with the final H3 layout: "
                             f"sampler=({declared_tokens}, {declared_capacity}), layout=({real_tokens}, {capacity})")
        batch.minimax_h3_layout = pad_packed_sequence(layout, capacity - real_tokens)
        batch.sequence_packing_num_tokens = real_tokens
        batch.sequence_padded_num_tokens = capacity
        batch.sequence_padding_ratio = capacity / real_tokens

    def _maybe_build_dense_packed_metadata(self, batch: TrainingBatch) -> None:
        """Dense boundaries exclude trailing padding as well as other documents."""
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedBatchLayout) and not (isinstance(layout, MiniMaxH3PackedLayout)
                                                                       and batch.sequence_padded_num_tokens):
            return
        # Imported lazily so single-document training remains independent from
        # packed-attention metadata implementation details.
        from fastvideo.attention.backends.abstract import PackedAttentionMetadata, PackedSequenceMetadata

        batch.attn_metadata = PackedAttentionMetadata(
            current_timestep=0,
            packed_sequence=PackedSequenceMetadata.from_lengths(
                layout.document_lengths if isinstance(layout, MiniMaxH3PackedBatchLayout) else
                (layout.real_sequence_length, ),
                device=self.device,
                pad_tokens=layout.pad_tokens,
                sequence_capacity=batch.sequence_padded_num_tokens or None),
        )
        # The refiner must override the joint forward context even for a
        # singleton; its input contains neither capacity nor reference rows.
        batch.minimax_h3_text_attn_metadata = PackedAttentionMetadata(
            current_timestep=0,
            packed_sequence=PackedSequenceMetadata.from_lengths(
                layout.text_lengths if isinstance(layout, MiniMaxH3PackedBatchLayout) else
                (int(layout.text_indices.numel()), ),
                device=self.device),
        )

    def _maybe_build_vsa_metadata(self, batch: TrainingBatch) -> None:
        """Populate the VSA view when this model runs the VSA-H3 backend.

        Methods route it through ``predict_noise(attn_kind="vsa")``; dense
        models keep ``attn_metadata_vsa`` as ``None`` so their forwards stay
        dense. One builder per model keeps the padded tile buffer reused
        across steps (see ``MiniMaxH3VSAMetadata.tile_buf_holder``).
        """
        backend = (getattr(self, "attention_backend_name", None) or envs.FASTVIDEO_ATTENTION_BACKEND.get())
        if backend != "VIDEO_SPARSE_ATTN_H3":
            return
        layout = batch.minimax_h3_layout
        patch_size = tuple(self.transformer.patch_size)
        builder = getattr(self, "_vsa_metadata_builder", None)
        if builder is None:
            builder = self._vsa_metadata_builder = MiniMaxH3VSAMetadataBuilder()
        if isinstance(layout, MiniMaxH3PackedBatchLayout):
            raise NotImplementedError("Cross-document VSA packing is not supported; use one document with capacities")
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("H3 VSA metadata requires a semantic packed layout")
        batch.attn_metadata_vsa = builder.build(
            # Training builds one metadata per batch and reuses it across the
            # step's forwards; the step index only feeds probe bookkeeping.
            current_timestep=0,
            patch_size=patch_size,
            VSA_sparsity=float(self.training_config.vsa_sparsity),
            packed_segments=_h3_vsa_single_region_segments(layout, patch_size),
            device=self.device,
            tile_size=int(self.training_config.vsa_tile_size),
            pad_tokens=layout.pad_tokens,
        )

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Mix clean and noise tensors using H3's clean-time convention."""
        clean_time = timestep.to(device=clean_latents.device, dtype=clean_latents.dtype)
        while clean_time.ndim < clean_latents.ndim:
            clean_time = clean_time.unsqueeze(-1)
        return clean_time * clean_latents + (1.0 - clean_time) * noise

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> NoisePrediction:
        """Pack modality timesteps and convert H3 outputs to noise-minus-clean."""
        del timestep
        if batch.audio_noisy_model_input is None:
            raise RuntimeError("prepare_batch() must set audio and text transformer inputs")
        video_input_dtype = noisy_latents.dtype
        audio_input_dtype = batch.audio_noisy_model_input.dtype
        (
            video_velocity,
            audio_velocity,
            (num_video_latents, latent_height, latent_width),
            num_audio_latents,
        ) = self._joint_forward_rows(
            noisy_latents,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        video_prediction = unpatchify_video_tokens(
            video_velocity,
            num_video_latents,
            latent_height,
            latent_width,
            _VIDEO_LATENT_CHANNELS,
            self.transformer.patch_size,
        ).permute(0, 2, 1, 3, 4)
        audio_prediction = unpack_audio_tokens(audio_velocity[0], num_audio_latents)[None]
        return (
            (-video_prediction).to(video_input_dtype),
            (-audio_prediction).to(audio_input_dtype),
        )

    def _joint_forward_rows(
        self,
        noisy_latents: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], int]:
        """Run one packed joint forward and return the generated output rows.

        Returns the raw transformer outputs restricted to the generated rows,
        ``(video_rows [1, R_v, D_v], audio_rows [1, R_a, D_a])``, plus the
        video latent geometry ``(frames, height, width)`` and the audio latent
        count. ``predict_noise`` unpatchifies them into the released
        single-head layout; a PDD-widened role splits ``D`` into
        ``(pdd_steps, C)`` heads first.
        """
        # Under dense backends both metadata views are None, so "vsa" silently
        # means dense (mirrors WanModel). MiniMaxH3DMDModel.prepare_batch
        # populates attn_metadata_vsa when the role runs VIDEO_SPARSE_ATTN_H3.
        if attn_kind not in ("dense", "vsa"):
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")
        attn_metadata = (batch.attn_metadata_vsa
                         if attn_kind == "vsa" and batch.attn_metadata_vsa is not None else batch.attn_metadata)
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_layout")
        if batch.audio_noisy_model_input is None or batch.encoder_hidden_states is None:
            raise RuntimeError("prepare_batch() must set audio and text transformer inputs")
        if batch.timesteps is None or batch.audio_timesteps is None:
            raise RuntimeError("prepare_batch() must set video and audio timesteps")

        encoder_hidden_states = batch.encoder_hidden_states
        if not conditional:
            # H3 has no negative-prompt encoder at training time, so the only
            # supported unconditional branch (teacher CFG in distillation)
            # zeroes the text embeddings.
            if (cfg_uncond or {}).get("text") != "zero":
                raise ValueError("MiniMaxH3Model unconditional forwards require "
                                 "method.cfg_uncond={'text': 'zero'}")
            encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        dtype = torch.bfloat16
        device = self.device
        video_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype)
        # Match H3 checkpoint token order: video rows flatten
        # (C, patch_t, patch_h, patch_w), while audio rows flatten stereo
        # channel, time, and latent feature dimensions in that order.
        video_rows = patchify_video_latents(video_bcthw, self.transformer.patch_size)
        audio_latents = batch.audio_noisy_model_input.to(dtype)
        num_audio_latents = audio_latents.shape[-1]
        audio_rows = audio_latents.permute(0, 1, 3, 2).reshape(-1, _AUDIO_LATENT_CHANNELS)
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=float(batch.timesteps[0]),
            audio_timestep=float(batch.audio_timesteps[0]),
            condition_video_timestep=max(float(batch.timesteps[0]), MINIMAX_H3_KEYFRAME_NOISE_AUG),
            condition_audio_timestep=1.0,
        )
        unique_timesteps = unique_timesteps.to(device)
        timestep_indices = timestep_indices.to(device)

        condition_video_rows = batch.minimax_h3_condition_video_rows
        condition_audio_rows = batch.minimax_h3_condition_audio_rows
        if layout.num_condition_video_rows:
            if condition_video_rows is None:
                raise RuntimeError("The Ref2VA layout reserves visual reference rows, but the batch carries none")
            if tuple(condition_video_rows.shape) != (layout.num_condition_video_rows, video_rows.shape[-1]):
                raise ValueError("Ref2VA visual reference rows do not match the packed layout: "
                                 f"expected {(layout.num_condition_video_rows, video_rows.shape[-1])}, "
                                 f"got {tuple(condition_video_rows.shape)}")
            video_rows = torch.cat((condition_video_rows.to(device=device, dtype=dtype), video_rows), dim=0)
        elif condition_video_rows is not None and condition_video_rows.numel():
            raise ValueError("The batch carries visual reference rows but its layout reserves none")
        if layout.num_condition_audio_rows:
            if condition_audio_rows is None:
                raise RuntimeError("The Ref2VA layout reserves audio reference rows, but the batch carries none")
            if tuple(condition_audio_rows.shape) != (layout.num_condition_audio_rows, audio_rows.shape[-1]):
                raise ValueError("Ref2VA audio reference rows do not match the packed layout: "
                                 f"expected {(layout.num_condition_audio_rows, audio_rows.shape[-1])}, "
                                 f"got {tuple(condition_audio_rows.shape)}")
            audio_rows = torch.cat((condition_audio_rows.to(device=device, dtype=dtype), audio_rows), dim=0)
        elif condition_audio_rows is not None and condition_audio_rows.numel():
            raise ValueError("The batch carries audio reference rows but its layout reserves none")

        with activation_offload_context(
                int(layout.token_tags.numel()),
                getattr(getattr(self.training_config, "model", None), "minimax_h3_activation_offload_min_tokens",
                        0)), set_forward_context(
                            current_timestep=unique_timesteps,
                            attn_metadata=attn_metadata,
                        ):
            video_velocity, audio_velocity = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=encoder_hidden_states,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=layout.token_tags.to(device),
                position_ids=layout.position_ids.to(device),
                video_indices=layout.video_indices.to(device),
                audio_indices=layout.audio_indices.to(device),
                text_indices=layout.text_indices.to(device),
                text_attn_metadata=batch.minimax_h3_text_attn_metadata,
            )

        video_velocity = video_velocity[:, layout.num_condition_video_rows:]
        audio_velocity = audio_velocity[:, layout.num_condition_audio_rows:]
        _, _, num_video_latents, latent_height, latent_width = video_bcthw.shape
        return (
            video_velocity,
            audio_velocity,
            (int(num_video_latents), int(latent_height), int(latent_width)),
            int(num_audio_latents),
        )

    def _predict_noise_packed_documents(
        self,
        video_documents: tuple[torch.Tensor, ...],
        audio_documents: tuple[torch.Tensor, ...],
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None,
        attn_kind: Literal["dense", "vsa"],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run one isolated-attention DiT call for a ragged document pack.

        Individual videos are patchified and reconstructed independently;
        only their row representations are concatenated.  Attention metadata
        carries the document boundaries, so the launch has the same semantics
        as independent forwards while avoiding shape padding.
        """
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedBatchLayout):
            raise RuntimeError("Packed-document prediction requires MiniMaxH3PackedBatchLayout")
        document_count = len(layout.documents)
        if len(video_documents) != document_count or len(audio_documents) != document_count:
            raise ValueError("Packed H3 video/audio document counts must match the semantic layout")
        if attn_kind not in ("dense", "vsa"):
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")
        # DMD calls the student with ``attn_kind="vsa"`` even for the dense
        # ablation. In that case no VSA view exists, but packed documents still
        # require their dense variable-length boundaries.
        attn_metadata = (batch.attn_metadata_vsa
                         if attn_kind == "vsa" and batch.attn_metadata_vsa is not None else batch.attn_metadata)
        if attn_metadata is None:
            raise RuntimeError(f"Packed H3 {attn_kind} attention requires document-boundary metadata")
        if batch.encoder_hidden_states is None:
            raise RuntimeError("Packed H3 prediction requires text embeddings")
        if batch.timesteps is None or batch.audio_timesteps is None:
            raise RuntimeError("Packed H3 prediction requires per-document video/audio timesteps")

        encoder_hidden_states = batch.encoder_hidden_states
        if not conditional:
            if (cfg_uncond or {}).get("text") != "zero":
                raise ValueError("MiniMaxH3Model unconditional forwards require "
                                 "method.cfg_uncond={'text': 'zero'}")
            encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if batch.timesteps.numel() == 1:
            video_timesteps = batch.timesteps.expand(document_count)
        elif batch.timesteps.numel() == document_count:
            video_timesteps = batch.timesteps.reshape(-1)
        else:
            raise ValueError("Packed H3 video timesteps must be scalar or one value per document")
        if batch.audio_timesteps.numel() == 1:
            audio_timesteps = batch.audio_timesteps.expand(document_count)
        elif batch.audio_timesteps.numel() == document_count:
            audio_timesteps = batch.audio_timesteps.reshape(-1)
        else:
            raise ValueError("Packed H3 audio timesteps must be scalar or one value per document")

        dtype = torch.bfloat16
        device = self.device
        patch_size = tuple(int(value) for value in self.transformer.patch_size)
        video_rows: list[torch.Tensor] = []
        audio_rows: list[torch.Tensor] = []
        video_bcthw_documents: list[torch.Tensor] = []
        for index, (video, audio,
                    document_layout) in enumerate(zip(video_documents, audio_documents, layout.documents, strict=True)):
            if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != _VIDEO_LATENT_CHANNELS:
                raise ValueError(f"Packed H3 video document {index} must have shape [1,T,24,H,W], "
                                 f"got {tuple(video.shape)}")
            if audio.ndim != 4 or tuple(audio.shape[:3]) != (1, MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_LATENT_CHANNELS):
                raise ValueError(f"Packed H3 audio document {index} must have shape [1,2,32,Ta], "
                                 f"got {tuple(audio.shape)}")
            video_bcthw = video.permute(0, 2, 1, 3, 4).to(dtype)
            if tuple(video_bcthw.shape[2:]) != (
                    document_layout.num_video_latent_frames,
                    document_layout.latent_height,
                    document_layout.latent_width,
            ):
                raise ValueError(f"Packed H3 video document {index} disagrees with its layout")
            if audio.shape[-1] != document_layout.num_audio_latents:
                raise ValueError(f"Packed H3 audio document {index} disagrees with its layout")
            video_bcthw_documents.append(video_bcthw)
            video_rows.append(patchify_video_latents(video_bcthw, patch_size))
            audio_rows.append(audio.to(dtype).permute(0, 1, 3, 2).reshape(-1, _AUDIO_LATENT_CHANNELS))

        video_time_values = tuple(float(value) for value in video_timesteps)
        audio_time_values = tuple(float(value) for value in audio_timesteps)
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=video_time_values,
            audio_timestep=audio_time_values,
            condition_video_timestep=tuple(max(value, MINIMAX_H3_KEYFRAME_NOISE_AUG) for value in video_time_values),
            condition_audio_timestep=(1.0, ) * document_count,
        )

        with set_forward_context(
                current_timestep=unique_timesteps.to(device),
                attn_metadata=attn_metadata,
        ):
            video_velocity, audio_velocity = self.transformer(
                hidden_states=torch.cat(video_rows, dim=0)[None],
                audio_hidden_states=torch.cat(audio_rows, dim=0)[None],
                encoder_hidden_states=encoder_hidden_states,
                timestep=unique_timesteps.to(device),
                timestep_indices=timestep_indices.to(device),
                token_tags=layout.token_tags.to(device),
                position_ids=layout.position_ids.to(device),
                video_indices=layout.video_indices.to(device),
                audio_indices=layout.audio_indices.to(device),
                text_indices=layout.text_indices.to(device),
                text_attn_metadata=batch.minimax_h3_text_attn_metadata,
            )

        video_predictions: list[torch.Tensor] = []
        audio_predictions: list[torch.Tensor] = []
        video_cursor = 0
        audio_cursor = 0
        for video, audio, video_bcthw, document_layout in zip(
                video_documents,
                audio_documents,
                video_bcthw_documents,
                layout.documents,
                strict=True,
        ):
            video_rows_count = int(document_layout.video_indices.numel())
            audio_rows_count = int(document_layout.audio_indices.numel())
            document_video_velocity = video_velocity[:, video_cursor:video_cursor + video_rows_count]
            document_audio_velocity = audio_velocity[:, audio_cursor:audio_cursor + audio_rows_count]
            video_cursor += video_rows_count
            audio_cursor += audio_rows_count
            video_predictions.append(-unpatchify_video_tokens(
                document_video_velocity,
                document_layout.num_video_latent_frames,
                document_layout.latent_height,
                document_layout.latent_width,
                _VIDEO_LATENT_CHANNELS,
                patch_size,
            ).permute(0, 2, 1, 3, 4).to(video.dtype))
            audio_predictions.append(-unpack_audio_tokens(document_audio_velocity[0], document_layout.num_audio_latents)
                                     [None].to(audio.dtype))
        if video_cursor != video_velocity.shape[1] or audio_cursor != audio_velocity.shape[1]:
            raise RuntimeError("Packed H3 transformer outputs were not consumed exactly once")
        return tuple(video_predictions), tuple(audio_predictions)

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        """Restore the forward context and average accumulated microbatch gradients."""
        timesteps, attn_metadata = ctx
        expected_device_type = self.device.type
        offenders = []
        for name, param in self.transformer.named_parameters():
            local = getattr(param, "_local_tensor", param)
            if local.device.type != expected_device_type:
                offenders.append(f"param {name} on {local.device}")
            if param.grad is not None:
                grad_local = getattr(param.grad, "_local_tensor", param.grad)
                if grad_local.device.type != expected_device_type:
                    offenders.append(f"grad {name} on {grad_local.device}")
        if offenders:
            raise RuntimeError(f"{len(offenders)} training tensors off-CUDA before backward; "
                               f"first offenders: {offenders[:8]}")
        with set_forward_context(
                current_timestep=timesteps,
                attn_metadata=attn_metadata,
        ):
            (loss / max(1, int(grad_accum_rounds))).backward()

    # ------------------------------------------------------------------
    # TDM contract: one joint video+audio document
    # ------------------------------------------------------------------

    @property
    def num_train_timesteps(self) -> int:
        """H3's trajectory grid runs over integer labels ``0..1000``.

        The released scheduler exposes no training horizon, so the joint TDM
        grid (the range used by the H3 TDM recipe) supplies it.
        """
        return _H3_TRAJECTORY_TIMESTEPS

    def tdm_modalities(self) -> tuple[str, ...]:
        """H3 trains video and audio jointly on one packed sequence."""
        return _H3_TDM_MODALITIES

    def tdm_clean_latents(self, batch: TrainingBatch) -> dict[str, torch.Tensor]:
        """Return the video and stereo-audio latents the trajectory shares."""
        if batch.latents is None or batch.audio_latents is None:
            raise RuntimeError("MiniMaxH3Model requires video and audio latents for TDM")
        return {"video": batch.latents, "audio": batch.audio_latents}

    def tdm_sigma_grid(
        self,
        timesteps: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Map integer H3 trajectory labels to one modality's sigmas."""
        return tdm_h3_sigma_grid(timesteps, modality)

    def tdm_terminal_sigma(self, modality: str) -> torch.Tensor:
        """Pure noise sits at trajectory label ``T`` where sigma is ``1``."""
        del modality
        return torch.ones((), dtype=torch.float32)

    def tdm_max_trajectory_label(self, modality: str) -> int:
        """The H3 grid runs over integer labels ``0..T``."""
        del modality
        return _H3_TRAJECTORY_TIMESTEPS

    def tdm_sigma_to_model_timestep(
        self,
        sigma: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """H3 model time is ``1 - sigma``; the shift is already in the grid."""
        del modality
        return tdm_h3_model_timestep(sigma)

    def tdm_predict_x0(
        self,
        noisy: dict[str, torch.Tensor],
        sigmas: dict[str, torch.Tensor],
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> dict[str, torch.Tensor]:
        """Run one packed transformer forward and convert both modalities to x0."""
        if not conditional or cfg_uncond is not None:
            raise ValueError("MiniMaxH3Model predicts one conditional T2VA sample")
        video_sigma = sigmas["video"].to(dtype=torch.float32)
        audio_sigma = sigmas["audio"].to(dtype=torch.float32)
        batch.noisy_model_input = noisy["video"]
        batch.audio_noisy_model_input = noisy["audio"]
        batch.timesteps = tdm_h3_model_timestep(video_sigma)
        batch.audio_timesteps = tdm_h3_model_timestep(audio_sigma)
        video_prediction, audio_prediction = self.predict_noise(
            noisy["video"],
            batch.timesteps,
            batch,
            conditional=True,
            cfg_uncond=None,
            attn_kind=attn_kind,
        )
        # H3's transformer emits a data-ward velocity, so
        # ``x0 = x_t - sigma * pred_noise`` with the negated prediction
        # reconstructs ``x0 = x_t + sigma * velocity``.
        return {
            "video": noisy["video"] - _broadcast_sigma(video_sigma, noisy["video"]) * video_prediction,
            "audio": noisy["audio"] - _broadcast_sigma(audio_sigma, noisy["audio"]) * audio_prediction,
        }


class MiniMaxH3LoraModel(MiniMaxH3Model):
    """Enable LoRA-only training for the MiniMax H3 transformer."""

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        lora: LoraConfig | dict[str, Any] | None = None,
        expected_lora_layers: int | None = None,
        attention_backend: AttentionBackendEnum | str | None = AttentionBackendEnum.TORCH_SDPA,
    ) -> None:
        lora_config = LoraConfig.coerce(lora)
        if lora_config is None or not lora_config.enable:
            raise ValueError("MiniMaxH3LoraModel requires models.student.lora.enable=true")
        if not lora_config.target_modules:
            raise ValueError("MiniMaxH3LoraModel requires a non-empty "
                             "models.student.lora.target_modules list")
        if not trainable:
            raise ValueError("MiniMaxH3LoraModel requires trainable=true")

        expected_layers = (int(expected_lora_layers) if expected_lora_layers is not None else None)
        if expected_layers is not None and expected_layers <= 0:
            raise ValueError("expected_lora_layers must be positive")

        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=attention_backend,
            lora=lora_config,
        )

        if (expected_layers is not None and self._num_lora_layers != expected_layers):
            raise ValueError("Unexpected MiniMax H3 LoRA layer count: "
                             f"expected {expected_layers}, got {self._num_lora_layers}")

        unexpected_trainable = [
            name for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad and name.rsplit(".", maxsplit=1)[-1] not in {"lora_A", "lora_B"}
        ]
        if unexpected_trainable:
            raise ValueError("MiniMax H3 LoRA enabled non-LoRA trainable parameters: "
                             f"{unexpected_trainable[:10]}")


__all__ = [
    "MiniMaxH3LoraModel",
    "MiniMaxH3Model",
    "shift_noise_amount",
    "tdm_h3_model_timestep",
    "tdm_h3_shift",
    "tdm_h3_sigma_grid",
]