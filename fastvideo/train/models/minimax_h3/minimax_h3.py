# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 joint text-to-video-and-audio training plugin."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distributed import get_sp_group
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedLayout,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.platforms import AttentionBackendEnum

from fastvideo.train.models.base import ModelBase, NoisePrediction
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing
from fastvideo.train.utils.module_state import apply_trainable
from fastvideo.train.utils.moduleloader import load_module_from_path

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig

# H3 maps one shared denoising stage through modality-specific scheduler
# shifts so video and audio remain synchronized at different noise amounts.
_VIDEO_SCHEDULER_SHIFT = 12.0
_AUDIO_SCHEDULER_SHIFT = 3.0
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32


def shift_noise_amount(base_noise_amount: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the MiniMax H3 rational shift to a unit noise amount."""
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    return shift * base_noise_amount / (1.0 + (shift - 1.0) * base_noise_amount)


class MiniMaxH3Model(ModelBase):
    """Adapt the H3 joint transformer to the modular fine-tuning contract."""

    _transformer_cls_name = "MiniMaxH3Transformer3DModel"

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
    ) -> None:
        """Validate the single-document T2VA contract and load the transformer."""
        super().__init__(
            trainable=trainable,
            attention_backend=attention_backend,
        )
        # PyTorch scaled dot product attention (SDPA) provides dense attention
        # without adding another attention-kernel dependency to H3 training.
        if self.attention_backend != AttentionBackendEnum.TORCH_SDPA:
            raise ValueError("MiniMaxH3Model requires the TORCH_SDPA attention backend")
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
        # Joint supervision requires paired video and stereo-audio latents from
        # every parquet row.
        if str(training_config.data.preprocessed_data_type) != "t2va":
            raise ValueError("MiniMaxH3Model requires training.data.preprocessed_data_type='t2va'")

        # FastVideo's Fully Sharded Data Parallel loading path requires one BF16
        # parameter dtype, including modules that H3 inference keeps in FP32.
        training_config.pipeline_config.dit_config.uniform_parameter_dtype = True  # type: ignore[attr-defined]

        self._init_from = str(init_from)
        self.training_config = training_config
        self.transformer = self._load_transformer(
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
        )
        self.noise_scheduler = MiniMaxH3Scheduler(shift=_VIDEO_SCHEDULER_SHIFT)
        self.audio_noise_scheduler = MiniMaxH3Scheduler(shift=_AUDIO_SCHEDULER_SHIFT)
        self.dataloader: Any = None
        self.validator: Any = None
        self.start_step = 0
        self.sp_group: Any = None

    def _load_transformer(
        self,
        *,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
        transformer_override_safetensor: str | None,
    ) -> torch.nn.Module:
        """Load H3 through the training FSDP loader and apply block checkpointing."""
        transformer = load_module_from_path(
            model_path=self._init_from,
            module_type="transformer",
            training_config=self.training_config,
            disable_custom_init_weights=disable_custom_init_weights,
            override_transformer_cls_name=self._transformer_cls_name,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=self.attention_backend,
        )
        checkpointing_type = (enable_gradient_checkpointing_type
                              or self.training_config.model.enable_gradient_checkpointing_type)
        if trainable and checkpointing_type:
            transformer = apply_activation_checkpointing(
                transformer,
                checkpointing_type=checkpointing_type,
            )
        return apply_trainable(transformer, trainable=trainable)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Load precomputed text embeddings and paired video-audio latents."""
        from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2va
        from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader

        self.sp_group = get_sp_group()
        text_config = training_config.pipeline_config.text_encoder_configs[0]  # type: ignore[union-attr]
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_config.arch_config.text_len),
            parquet_schema=pyarrow_schema_t2va,
        )
        self.start_step = 0

    def _resolve_clean_latents(
        self,
        raw_batch: dict[str, Any],
        latents_source: Literal["data", "zeros"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve fixed visual and stereo-audio latent tensors for one sample."""
        data_config = self.training_config.data
        if latents_source == "data":
            if "vae_latent" not in raw_batch or "audio_latent" not in raw_batch:
                raise ValueError("A T2VA batch requires vae_latent and audio_latent tensors")
            video_latents = raw_batch["vae_latent"]
            audio_latents = raw_batch["audio_latent"]
        elif latents_source == "zeros":
            video_latents = torch.zeros(
                1,
                _VIDEO_LATENT_CHANNELS,
                data_config.num_latent_t,
                data_config.num_height // 16,
                data_config.num_width // 16,
            )
            audio_latents = torch.zeros(
                1,
                MINIMAX_H3_AUDIO_CHANNELS,
                _AUDIO_LATENT_CHANNELS,
                audio_latent_num_frames(data_config.num_frames),
            )
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

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
            shift_noise_amount(base_noise_amount, _VIDEO_SCHEDULER_SHIFT),
            shift_noise_amount(base_noise_amount, _AUDIO_SCHEDULER_SHIFT),
        )

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Prepare normalized T2VA latents, shifted noise, text, and row layout."""
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
        video_noise = torch.randn(video_latents.shape, generator=generator, device=device, dtype=dtype)
        audio_noise = torch.randn(audio_latents.shape, generator=generator, device=device, dtype=dtype)
        video_noise_amount, audio_noise_amount = self._sample_noise_amounts(generator, device)
        video_sigmas = video_noise_amount.to(dtype).view(1, 1, 1, 1, 1)
        audio_sigmas = audio_noise_amount.to(dtype).view(1, 1, 1, 1)
        noisy_video_latents = (1.0 - video_sigmas) * video_latents + video_sigmas * video_noise
        noisy_audio_latents = (1.0 - audio_sigmas) * audio_latents + audio_sigmas * audio_noise

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
        return training_batch

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
        if not conditional or cfg_uncond is not None:
            raise ValueError("MiniMaxH3Model predicts one conditional T2VA sample")
        if attn_kind != "dense":
            raise ValueError("MiniMaxH3Model supports dense attention for training")
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_layout")
        if batch.audio_noisy_model_input is None or batch.encoder_hidden_states is None:
            raise RuntimeError("prepare_batch() must set audio and text transformer inputs")
        if batch.timesteps is None or batch.audio_timesteps is None:
            raise RuntimeError("prepare_batch() must set video and audio timesteps")

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
            condition_video_timestep=float(batch.timesteps[0]),
            condition_audio_timestep=float(batch.audio_timesteps[0]),
        )
        unique_timesteps = unique_timesteps.to(device)
        timestep_indices = timestep_indices.to(device)

        with torch.autocast(device.type, dtype=dtype), set_forward_context(
                current_timestep=unique_timesteps,
                attn_metadata=None,
        ):
            video_velocity, audio_velocity = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=batch.encoder_hidden_states,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=layout.token_tags.to(device),
                position_ids=layout.position_ids.to(device),
                video_indices=layout.video_indices.to(device),
                audio_indices=layout.audio_indices.to(device),
                text_indices=layout.text_indices.to(device),
            )

        _, _, num_video_latents, latent_height, latent_width = video_bcthw.shape
        video_prediction = unpatchify_video_tokens(
            video_velocity,
            num_video_latents,
            latent_height,
            latent_width,
            _VIDEO_LATENT_CHANNELS,
            self.transformer.patch_size,
        ).permute(0, 2, 1, 3, 4)
        audio_prediction = unpack_audio_tokens(audio_velocity[0], num_audio_latents)[None]
        return -video_prediction, -audio_prediction

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        """Restore the forward context and average accumulated microbatch gradients."""
        timesteps, attn_metadata = ctx
        with set_forward_context(
                current_timestep=timesteps,
                attn_metadata=attn_metadata,
        ):
            (loss / max(1, int(grad_accum_rounds))).backward()


__all__ = ["MiniMaxH3Model", "shift_noise_amount"]
