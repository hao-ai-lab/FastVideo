# SPDX-License-Identifier: Apache-2.0
"""WanTrack training integration shared by bidirectional and causal models."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.training.training_utils import normalize_dit_input


@dataclass(slots=True)
class TrackAugmentationConfig:
    """Configuration-driven MotionStream-style training augmentation."""

    enabled: bool = True
    min_points: int = 1000
    max_points: int = 2500
    sparse_object_sampling: bool = False
    extra_points: int = 0
    extra_point_sampling: Literal["random", "weighted"] = "random"
    track_dropout_probability: float = 0.0
    temporal_mask_probability: float = 0.2
    temporal_mask_chunk_size: int = 8
    motion_dropout_probability: float = 0.0
    text_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.min_points < 0 or self.max_points < 0:
            raise ValueError("min_points and max_points must be non-negative")
        if self.max_points and self.min_points > self.max_points:
            raise ValueError("min_points cannot exceed max_points")
        if self.extra_points < 0:
            raise ValueError("extra_points must be non-negative")
        if self.extra_point_sampling not in {"random", "weighted"}:
            raise ValueError("extra_point_sampling must be 'random' or "
                             "'weighted'")
        if self.temporal_mask_chunk_size <= 0:
            raise ValueError("temporal_mask_chunk_size must be positive")
        for name in (
                "track_dropout_probability",
                "temporal_mask_probability",
                "motion_dropout_probability",
                "text_dropout_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @classmethod
    def from_value(
        cls,
        value: TrackAugmentationConfig | Mapping[str, Any] | None,
    ) -> TrackAugmentationConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("track_augmentation must be a mapping or "
                            "TrackAugmentationConfig")
        valid_fields = {item.name for item in fields(cls)}
        unknown = set(value) - valid_fields
        if unknown:
            raise ValueError("Unknown track_augmentation fields: "
                             f"{sorted(unknown)}")
        return cls(**dict(value))


class WanTrackModel(WanModel):
    """Bidirectional WanTrack model and shared causal conditioning."""

    _transformer_cls_name = "TrackWanTransformer3DModel"

    def __init__(
        self,
        *,
        track_augmentation: TrackAugmentationConfig
        | Mapping[str, Any]
        | None = None,
        freeze_track_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        self.track_augmentation = TrackAugmentationConfig.from_value(track_augmentation)
        self.freeze_track_encoder = bool(freeze_track_encoder)
        super().__init__(**kwargs)
        if self.freeze_track_encoder:
            track_encoder = getattr(self.transformer, "track_encoder", None)
            if track_encoder is None:
                raise ValueError("freeze_track_encoder=true requires a "
                                 "transformer.track_encoder")
            track_encoder.requires_grad_(False)

    def init_preprocessors(self, training_config: Any) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()

        preprocessed_data_type = str(getattr(
            training_config.data,
            "preprocessed_data_type",
            "",
        )).strip().lower()
        if preprocessed_data_type != "i2v_track":
            raise ValueError("WanTrack requires data.preprocessed_data_type='i2v_track', "
                             f"got {preprocessed_data_type!r}")

        text_len = (training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len)
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_len),
            parquet_schema=pyarrow_schema_i2v_track,
        )
        self.start_step = 0

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
        training_config = self.training_config
        dtype = self._get_training_dtype()
        device = self.device

        required = (
            "first_frame_latent",
            "clip_feature",
            "track_points",
            "track_visibility",
            "text_embedding",
            "text_attention_mask",
        )
        missing = [name for name in required if name not in raw_batch]
        if missing:
            raise ValueError(f"WanTrack batch is missing required fields: {missing}")

        latent_t = int(training_config.data.num_latent_t)
        if latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError("WanTrack requires vae_latent when "
                                 "latents_source='data'")
            latents = raw_batch["vae_latent"][:, :, :latent_t].to(device=device, dtype=dtype)
        elif latents_source == "zeros":
            batch_size = int(raw_batch["text_embedding"].shape[0])
            vae_config = (training_config.pipeline_config.vae_config.arch_config)
            latents = torch.zeros(
                batch_size,
                int(vae_config.z_dim),
                latent_t,
                int(training_config.data.num_height) // int(vae_config.spatial_compression_ratio),
                int(training_config.data.num_width) // int(vae_config.spatial_compression_ratio),
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")
        first_frame_latent = raw_batch["first_frame_latent"][:, :, :latent_t].to(device=device, dtype=dtype)
        image_embeds = raw_batch["clip_feature"]
        if not torch.is_tensor(image_embeds) or image_embeds.numel() == 0:
            raise ValueError("WanTrack requires non-empty clip_feature")
        image_embeds = image_embeds.to(device=device, dtype=dtype)

        expected_track_frames = ((latent_t - 1) * self._temporal_compression_ratio() + 1)
        track_points = raw_batch["track_points"]
        track_visibility = raw_batch["track_visibility"]
        if track_points.ndim != 4 or track_points.shape[-1] != 2:
            raise ValueError("track_points must have shape [B, T, N, 2], got "
                             f"{tuple(track_points.shape)}")
        if track_visibility.shape != track_points.shape[:-1]:
            raise ValueError("track_visibility must have shape [B, T, N], got "
                             f"{tuple(track_visibility.shape)}")
        if track_points.shape[1] < expected_track_frames:
            raise ValueError("track sequence is shorter than the requested latent clip: "
                             f"{track_points.shape[1]} < {expected_track_frames}")
        track_points = track_points[:, :expected_track_frames].to(device=device, dtype=torch.float32)
        track_visibility = track_visibility[:, :expected_track_frames].to(device=device, dtype=torch.float32)
        num_tracks = int(track_points.shape[2])

        object_ids = raw_batch.get("object_ids")
        if (torch.is_tensor(object_ids) and object_ids.numel() > 0
                and object_ids.shape == (track_points.shape[0], num_tracks)):
            object_ids = object_ids.to(device=device, dtype=torch.long)
        else:
            object_ids = None

        track_weights = raw_batch.get("track_weights")
        if (torch.is_tensor(track_weights) and track_weights.numel() > 0
                and track_weights.shape == (track_points.shape[0], num_tracks)):
            track_weights = track_weights.to(device=device, dtype=torch.float32)
        else:
            track_weights = None

        encoder_hidden_states = raw_batch["text_embedding"].to(device=device, dtype=dtype)
        encoder_hidden_states, track_visibility = self._augment_conditions(
            encoder_hidden_states,
            track_visibility,
            object_ids=object_ids,
            track_weights=track_weights,
            generator=generator,
        )
        batch_size = int(track_points.shape[0])
        max_track_id = int(getattr(
            getattr(self.transformer, "track_encoder", None),
            "max_track_id",
            100_000,
        ))
        if num_tracks > max_track_id:
            raise ValueError(f"num_tracks ({num_tracks}) exceeds max_track_id "
                             f"({max_track_id})")
        track_ids = torch.stack([
            torch.randperm(
                max_track_id,
                generator=generator,
                device=device,
            )[:num_tracks] for _ in range(batch_size)
        ])

        training_batch = TrainingBatch(
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=raw_batch["text_attention_mask"].to(device=device, dtype=dtype),
            image_embeds=image_embeds,
            image_latents=first_frame_latent,
            track_points=track_points,
            track_visibility=track_visibility,
            track_ids=track_ids,
            infos=raw_batch.get("info_list"),
        )
        training_batch.latents = normalize_dit_input(
            "wan",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)
        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]
        return training_batch

    def _augment_conditions(
        self,
        encoder_hidden_states: torch.Tensor,
        track_visibility: torch.Tensor,
        *,
        object_ids: torch.Tensor | None,
        track_weights: torch.Tensor | None,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.track_augmentation
        if not config.enabled:
            return encoder_hidden_states, track_visibility

        visibility = track_visibility.clone()
        batch_size, num_frames, num_tracks = visibility.shape
        device = visibility.device

        # Keep tensors rectangular for the collator; sampling drops tracks by
        # zeroing their visibility. Sparse mode preserves one visible query per
        # segmented object, then adds background/context queries.
        if config.sparse_object_sampling:
            if object_ids is None:
                raise ValueError("sparse_object_sampling requires object_ids "
                                 "in the WanTrack parquet")
            for batch_index in range(batch_size):
                valid = visibility[batch_index, 0] > 0.5
                keep = torch.zeros(
                    num_tracks,
                    device=device,
                    dtype=torch.bool,
                )
                sample_object_ids = object_ids[batch_index]
                for object_id in torch.unique(sample_object_ids):
                    if int(object_id.item()) < 0:
                        continue
                    candidates = torch.nonzero(
                        (sample_object_ids == object_id) & valid,
                        as_tuple=True,
                    )[0]
                    if candidates.numel() == 0:
                        candidates = torch.nonzero(
                            sample_object_ids == object_id,
                            as_tuple=True,
                        )[0]
                    if candidates.numel() > 0:
                        choice = torch.randint(
                            int(candidates.numel()),
                            (1, ),
                            generator=generator,
                            device=device,
                        )
                        keep[candidates[choice]] = True

                pool = valid & ~keep
                extra_count = min(
                    config.extra_points,
                    int(pool.sum().item()),
                )
                if extra_count > 0:
                    weights = pool.to(dtype=torch.float32)
                    if (config.extra_point_sampling == "weighted" and track_weights is not None):
                        weights *= (0.05 + track_weights[batch_index].clamp_min(0))
                    selected = torch.multinomial(
                        weights,
                        extra_count,
                        replacement=False,
                        generator=generator,
                    )
                    keep[selected] = True
                visibility[batch_index] *= keep.to(dtype=visibility.dtype).unsqueeze(0)
        elif config.max_points > 0 and config.min_points < num_tracks:
            high = min(config.max_points, num_tracks)
            low = min(config.min_points, high)
            for batch_index in range(batch_size):
                keep_count = int(torch.randint(
                    low,
                    high + 1,
                    (1, ),
                    generator=generator,
                    device=device,
                ).item())
                permutation = torch.randperm(
                    num_tracks,
                    generator=generator,
                    device=device,
                )
                keep = torch.zeros(
                    num_tracks,
                    device=device,
                    dtype=visibility.dtype,
                )
                keep[permutation[:keep_count]] = 1
                visibility[batch_index] *= keep.unsqueeze(0)

        if config.track_dropout_probability > 0:
            for batch_index in range(batch_size):
                kept = visibility[batch_index].amax(dim=0) > 0.5
                kept_indices = torch.nonzero(kept, as_tuple=True)[0]
                if kept_indices.numel() <= 1:
                    continue
                drop = (torch.rand(
                    num_tracks,
                    generator=generator,
                    device=device,
                ) < config.track_dropout_probability) & kept
                if int((kept & ~drop).sum().item()) == 0:
                    spare = kept_indices[torch.randint(
                        int(kept_indices.numel()),
                        (1, ),
                        generator=generator,
                        device=device,
                    )]
                    drop[spare] = False
                visibility[batch_index, :, drop] = 0

        chunk_size = config.temporal_mask_chunk_size
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        if config.temporal_mask_probability > 0:
            drop_chunks = torch.rand(
                batch_size,
                num_chunks,
                generator=generator,
                device=device,
            ) < config.temporal_mask_probability
            for chunk_index in range(num_chunks):
                start = chunk_index * chunk_size
                end = min(start + chunk_size, num_frames)
                visibility[:, start:end] *= (~drop_chunks[:, chunk_index]).to(dtype=visibility.dtype).view(-1, 1, 1)

        if config.motion_dropout_probability > 0:
            drop_motion = torch.rand(
                batch_size,
                generator=generator,
                device=device,
            ) < config.motion_dropout_probability
            visibility[drop_motion] = 0

        if config.text_dropout_probability > 0:
            drop_text = torch.rand(
                batch_size,
                generator=generator,
                device=device,
            ) < config.text_dropout_probability
            encoder_hidden_states = encoder_hidden_states.clone()
            encoder_hidden_states[drop_text] = 0

        return encoder_hidden_states, visibility

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        training_batch = super()._prepare_dit_inputs(
            training_batch,
            generator,
        )
        if training_batch.image_latents is None:
            raise RuntimeError("WanTrack requires first_frame_latent")
        if training_batch.image_embeds is None:
            raise RuntimeError("WanTrack requires clip_feature")

        image_condition = self._build_i2v_condition(training_batch.image_latents)
        training_batch.image_latents = image_condition
        assert training_batch.noisy_model_input is not None
        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, image_condition],
            dim=1,
        )
        assert training_batch.conditional_dict is not None
        extras = {
            "image_latents": image_condition,
            "encoder_hidden_states_image": training_batch.image_embeds,
            "track_points": training_batch.track_points,
            "track_visibility": training_batch.track_visibility,
            "track_ids": training_batch.track_ids,
        }
        training_batch.conditional_dict.update(extras)
        if training_batch.unconditional_dict is None:
            training_batch.unconditional_dict = dict(training_batch.conditional_dict)
        else:
            training_batch.unconditional_dict.update(extras)
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        start_frame: int = 0,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for WanTrack")
        hidden_states = self._append_i2v_for_window(
            noise_input,
            text_dict,
            start_frame=start_frame,
        )
        kwargs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "encoder_hidden_states_image": text_dict.get("encoder_hidden_states_image"),
            "track_points": text_dict.get("track_points"),
            "track_visibility": text_dict.get("track_visibility"),
            "track_ids": text_dict.get("track_ids"),
            "timestep": timestep,
            "return_dict": False,
        }
        if clean_x is not None:
            kwargs["clean_x"] = self._append_i2v_for_window(
                clean_x,
                text_dict,
                start_frame=start_frame,
            )
            kwargs["aug_t"] = aug_t
        return kwargs

    def _append_i2v_for_window(
        self,
        latents: torch.Tensor,
        text_dict: dict[str, Any],
        *,
        start_frame: int,
    ) -> torch.Tensor:
        hidden_states = latents.permute(0, 2, 1, 3, 4)
        if hidden_states.shape[1] == 36:
            return hidden_states
        if hidden_states.shape[1] != 16:
            raise ValueError("WanTrack expects a 16-channel latent or a 36-channel "
                             f"latent+I2V input, got {hidden_states.shape[1]}")
        image_condition = text_dict.get("image_latents")
        if image_condition is None:
            raise RuntimeError("WanTrack requires image_latents for a 16-channel input")
        start = int(start_frame)
        end = start + hidden_states.shape[2]
        if start < 0 or end > image_condition.shape[2]:
            raise ValueError("I2V condition does not cover the requested latent window "
                             f"[{start}, {end})")
        return torch.cat(
            [hidden_states, image_condition[:, :, start:end]],
            dim=1,
        )

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sanitized_cfg = None
        track_policy = "keep"
        if cfg_uncond is not None:
            sanitized_cfg = dict(cfg_uncond)
            policies = []
            for channel in ("track", "motion"):
                value = sanitized_cfg.pop(channel, None)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError(f"cfg_uncond.{channel} must be a string")
                    policies.append(value.strip().lower())
            if policies:
                if len(set(policies)) != 1:
                    raise ValueError("cfg_uncond.track and cfg_uncond.motion conflict")
                track_policy = policies[0]

        result = dict(super()._get_uncond_text_dict(
            batch,
            cfg_uncond=sanitized_cfg,
        ))
        conditional = batch.conditional_dict
        if conditional is None:
            raise RuntimeError("Missing conditional_dict in TrainingBatch")
        for key in (
                "image_latents",
                "encoder_hidden_states_image",
                "track_points",
                "track_visibility",
                "track_ids",
        ):
            result.setdefault(key, conditional.get(key))

        if track_policy not in {"keep", "zero", "drop"}:
            raise ValueError("cfg_uncond.track/motion must be one of {keep, zero, drop}, "
                             f"got {track_policy!r}")
        if track_policy in {"zero", "drop"}:
            result["track_points"] = None
            result["track_visibility"] = None
            result["track_ids"] = None
        return result

    def _build_i2v_condition(
        self,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        if image_latents.ndim != 5:
            raise ValueError("first_frame_latent must be [B, C, T, H, W], got "
                             f"{tuple(image_latents.shape)}")
        if image_latents.shape[1] == 20:
            return image_latents
        if image_latents.shape[1] != 16:
            raise ValueError("first_frame_latent must have 16 or 20 channels, got "
                             f"{image_latents.shape[1]}")

        ratio = self._temporal_compression_ratio()
        batch_size, _, latent_t, height, width = image_latents.shape
        pixel_frames = (latent_t - 1) * ratio + 1
        mask = torch.ones(
            batch_size,
            1,
            pixel_frames,
            height,
            width,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        mask[:, :, 1:] = 0
        mask = torch.cat([
            mask[:, :, :1].repeat_interleave(ratio, dim=2),
            mask[:, :, 1:],
        ], dim=2)
        mask = mask.view(
            batch_size,
            -1,
            ratio,
            height,
            width,
        ).transpose(1, 2)
        return torch.cat([mask, image_latents], dim=1)

    def _temporal_compression_ratio(self) -> int:
        assert self.training_config is not None
        return int(self.training_config.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
