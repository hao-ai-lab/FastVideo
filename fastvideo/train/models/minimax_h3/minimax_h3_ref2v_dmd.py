# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Ref2VA distribution-matching adapter."""

from __future__ import annotations

import math
from collections import Counter
from numbers import Integral
from typing import Any, Literal, cast

import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn_h3 import (
    MiniMaxH3VSAMetadata,
    MiniMaxH3VSAMetadataBuilder,
    assert_ref2va_vsa_metadata,
)
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedLayout,
    build_ref2va_packed_sequence,
    keyframe_condition_noise,
    reference_condition_generator,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MiniMaxH3PreparedReference,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (_h3_vsa_ref2va_segments,
                                                                              _h3_vsa_single_region_segments)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    _AUDIO_LATENT_CHANNELS,
    _VIDEO_LATENT_CHANNELS,
    MiniMaxH3Model,
)
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import MiniMaxH3DMDModel


class MiniMaxH3Ref2VDMDModel(MiniMaxH3DMDModel):
    """DMD2 role backed by H3's dedicated ordered-reference denoiser.

    Reference rows are immutable conditioning. They are prepended only for a
    transformer forward and stripped from its outputs; DMD2's rollout state,
    modality slices, and losses therefore continue to contain target video and
    target audio only.
    """

    _transformer_module_type = "transformer_ref"
    _supports_cross_sample_sequence_packing = False
    _defer_sequence_capacity = True
    ref2va_reference_aware_vsa = True
    _supported_preprocessed_data_types = ("ref2va", )
    _supported_attention_backends = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
    )
    _ref2va_vsa_gate_receipts: list[dict[str, Any]]

    def _resolved_vsa_ref_keep_rate(self) -> float:
        """Resolve and validate the P2 reference-video keep fraction."""
        sparsity = float(self.training_config.vsa_sparsity)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Ref2VA VSA-H3 requires training.vsa.sparsity in [0, 1), got {sparsity!r}")
        configured = getattr(self.training_config, "vsa_ref_keep_rate", None)
        if isinstance(configured, bool):
            raise ValueError(f"training.vsa.ref_keep_rate must be numeric, got {configured!r}")
        keep_rate = 1.0 - sparsity if configured is None else float(configured)
        if not 0.0 < keep_rate <= 1.0:
            raise ValueError(f"training.vsa.ref_keep_rate must be in (0, 1], got {keep_rate!r}")
        if sparsity > 0.0 and keep_rate >= 1.0:
            raise ValueError("Ref2VA VSA-H3 cannot use training.vsa.ref_keep_rate=1 for a sparse run: that would leave "
                             "reference-video conditioning dense.")
        return keep_rate

    def _maybe_build_vsa_metadata(self, batch: TrainingBatch) -> None:
        """Build Ref2VA P2 metadata only after the final ordered layout exists.

        ``MiniMaxH3Model.prepare_batch`` first creates a temporary T2VA layout.
        Virtual dispatch reaches this method during that call; leaving metadata
        unset until ``prepare_batch`` installs the final Ref2VA layout prevents
        the target-only/P1 geometry from being reused by training rollouts.
        """
        backend = (getattr(self, "attention_backend_name", None) or envs.FASTVIDEO_ATTENTION_BACKEND.get())
        if backend != "VIDEO_SPARSE_ATTN_H3":
            return
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_layout")
        if batch.minimax_h3_condition_video_rows is None:
            # The parent is still preparing the temporary target-only layout.
            batch.attn_metadata_vsa = None
            return

        patch_size = tuple(int(value) for value in self.transformer.patch_size)
        packed_segments = (_h3_vsa_ref2va_segments(layout, patch_size)
                           if layout.reference_segments else _h3_vsa_single_region_segments(layout, patch_size))
        ref_keep_rate = self._resolved_vsa_ref_keep_rate()
        builder = getattr(self, "_vsa_metadata_builder", None)
        if builder is None:
            builder = self._vsa_metadata_builder = MiniMaxH3VSAMetadataBuilder()
        metadata = builder.build(
            current_timestep=0,
            patch_size=patch_size,
            VSA_sparsity=float(self.training_config.vsa_sparsity),
            packed_segments=packed_segments,
            device=self.device,
            tile_size=int(self.training_config.vsa_tile_size),
            ref_keep_rate=ref_keep_rate,
            pad_tokens=layout.pad_tokens,
        )
        assert_ref2va_vsa_metadata(
            metadata,
            expected_reference_video_regions=sum(isinstance(segment, tuple) for segment in packed_segments) - 1,
            target_sparsity=float(self.training_config.vsa_sparsity),
            ref_keep_rate=ref_keep_rate,
        )
        batch.attn_metadata_vsa = metadata

    def drain_ref2va_vsa_gate_receipts(self) -> list[dict[str, Any]]:
        """Return lightweight P2/kernel receipts collected for a capacity gate."""
        receipts = getattr(self, "_ref2va_vsa_gate_receipts", [])
        self._ref2va_vsa_gate_receipts = []
        result = []
        for receipt in receipts:
            item = dict(receipt)
            item["runtime_routes"] = sorted(receipt["runtime_routes"])
            result.append(item)
        return result

    def init_preprocessors(self, training_config: Any) -> None:
        """Load target latents and precomputed ordered-reference conditions."""
        from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va
        from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader

        from fastvideo.distributed import get_sp_group

        packing = getattr(training_config.data, "cross_sample_sequence_packing", None)
        if bool(getattr(packing, "enabled", False)):
            raise ValueError("Ref2VA supports single-document fixed_sequence_capacities, not cross-sample packing")
        self.sp_group = get_sp_group()
        max_presentation_tokens = getattr(training_config.data, "max_presentation_tokens", None)
        if (isinstance(max_presentation_tokens, bool) or not isinstance(max_presentation_tokens, Integral)
                or int(max_presentation_tokens) <= 0):
            raise ValueError("MiniMax H3 Ref2VA training requires training.data.max_presentation_tokens to be a "
                             "positive integer safety bound")
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=None,
            parquet_schema=pyarrow_schema_ref2va,
            max_presentation_tokens=int(max_presentation_tokens),
        )
        self.start_step = 0

    @staticmethod
    def _metadata_list(info: dict[str, Any], name: str) -> list[Any]:
        value = info.get(name)
        if not isinstance(value, list):
            raise ValueError(f"Ref2VA metadata {name!r} must be a list, got {type(value).__name__}")
        return value

    @classmethod
    def _prepared_references(
        cls,
        batch: TrainingBatch,
    ) -> tuple[list[MiniMaxH3PreparedReference], int]:
        infos = batch.infos
        if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
            raise ValueError("A Ref2VA batch requires exactly one info_list metadata record")
        info = infos[0]
        fields = {
            name: cls._metadata_list(info, name)
            for name in (
                "reference_media_types",
                "reference_has_audio",
                "reference_num_latent_frames",
                "reference_latent_heights",
                "reference_latent_widths",
                "reference_num_audio_latents",
            )
        }
        reference_count = len(fields["reference_media_types"])
        if reference_count == 0:
            raise ValueError("A Ref2VA batch requires at least one ordered reference")
        mismatched = {name: len(values) for name, values in fields.items() if len(values) != reference_count}
        if mismatched:
            raise ValueError("Ref2VA ordered metadata lists must have equal lengths; "
                             f"expected {reference_count}, got {mismatched}")
        if reference_count > MINIMAX_H3_MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references")

        def integer(name: str, index: int, *, positive: bool) -> int:
            value = fields[name][index]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"Ref2VA metadata {name}[{index}] must be an integer, got {value!r}")
            result = int(value)
            if (positive and result <= 0) or (not positive and result < 0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"Ref2VA metadata {name}[{index}] must be {qualifier}, got {result}")
            return result

        references: list[MiniMaxH3PreparedReference] = []
        for index in range(reference_count):
            media_type = fields["reference_media_types"][index]
            if media_type not in ("image", "video", "audio"):
                raise ValueError(f"Unsupported Ref2VA media type at index {index}: {media_type!r}")
            has_audio = fields["reference_has_audio"][index]
            if not isinstance(has_audio, bool):
                raise ValueError(f"Ref2VA metadata reference_has_audio[{index}] must be bool, got {has_audio!r}")
            num_audio_latents = integer("reference_num_audio_latents", index, positive=False)
            if media_type == "audio" and not has_audio:
                raise ValueError("Every audio reference must set reference_has_audio=true")
            if media_type == "image" and has_audio:
                raise ValueError("An image reference cannot carry audio latents")
            if has_audio != (num_audio_latents > 0):
                raise ValueError(f"Reference {index} has_audio={has_audio} but num_audio_latents={num_audio_latents}")

            if media_type == "audio":
                num_frames = integer("reference_num_latent_frames", index, positive=False)
                height = integer("reference_latent_heights", index, positive=False)
                width = integer("reference_latent_widths", index, positive=False)
                if (num_frames, height, width) != (0, 0, 0):
                    raise ValueError("Audio-only references must store zero visual latent geometry")
            else:
                num_frames = integer("reference_num_latent_frames", index, positive=True)
                height = integer("reference_latent_heights", index, positive=True)
                width = integer("reference_latent_widths", index, positive=True)
                if media_type == "image" and num_frames != 1:
                    raise ValueError(f"Reference image {index} must contain one latent frame, got {num_frames}")

            references.append(
                MiniMaxH3PreparedReference(
                    media_type=cast(Literal["image", "video", "audio"], media_type),
                    has_audio=has_audio,
                    num_latent_frames=num_frames,
                    latent_height=height,
                    latent_width=width,
                    num_audio_latents=num_audio_latents,
                ))

        counts = Counter(reference.media_type for reference in references)
        limits = {
            "image": MINIMAX_H3_MAX_REFERENCE_IMAGES,
            "video": MINIMAX_H3_MAX_REFERENCE_VIDEOS,
            "audio": MINIMAX_H3_MAX_REFERENCE_AUDIOS,
        }
        for media_type, limit in limits.items():
            if counts[media_type] > limit:
                raise ValueError(f"MiniMax-H3 accepts at most {limit} {media_type} references")
        if counts["image"] + counts["video"] == 0:
            raise ValueError("Ref2VA requires at least one visual reference")

        noise_seed = info.get("reference_noise_seed")
        if isinstance(noise_seed, bool) or not isinstance(noise_seed, Integral) or int(noise_seed) < 0:
            raise ValueError(f"Ref2VA reference_noise_seed must be a non-negative integer, got {noise_seed!r}")
        return references, int(noise_seed)

    @staticmethod
    def _text_token_tags(raw_batch: dict[str, Any], valid_text: torch.Tensor) -> torch.Tensor:
        tags = raw_batch.get("text_token_tags")
        if not isinstance(tags, torch.Tensor) or tags.ndim != 2 or tags.shape != valid_text[None].shape:
            actual = None if not isinstance(tags, torch.Tensor) else tuple(tags.shape)
            raise ValueError("Ref2VA text_token_tags must match text_attention_mask with shape "
                             f"{tuple(valid_text[None].shape)}, got {actual}")
        tags = tags[0, valid_text]
        valid = (tags == MINIMAX_H3_TEXT_TAG) | (tags == MINIMAX_H3_VIDEO_TAG)
        if not bool(valid.all()):
            raise ValueError("Ref2VA text_token_tags may contain only text (1) and vision (0) tags")
        return tags.to(torch.long)

    @staticmethod
    def _latent_rows(
        raw_batch: dict[str, Any],
        name: str,
        *,
        expected_rows: int,
        feature_dim: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        rows = raw_batch.get(name)
        if not isinstance(rows, torch.Tensor):
            raise ValueError(f"A Ref2VA batch requires tensor {name!r}")
        if expected_rows == 0:
            if rows.numel() != 0:
                raise ValueError(f"{name} must be empty when the layout reserves no rows")
            return None
        expected_shape = (1, expected_rows, feature_dim)
        if tuple(rows.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(rows.shape)}")
        return rows[0].to(device=device, dtype=torch.float32)

    def _prepare_condition_rows(
        self,
        raw_batch: dict[str, Any],
        references: list[MiniMaxH3PreparedReference],
        layout: MiniMaxH3PackedLayout,
        noise_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = self.device
        patch_volume = math.prod(self.transformer.patch_size)
        clean_video = self._latent_rows(
            raw_batch,
            "reference_video_latent_rows",
            expected_rows=layout.num_condition_video_rows,
            feature_dim=_VIDEO_LATENT_CHANNELS * patch_volume,
            device=device,
        )
        if clean_video is None:
            raise ValueError("Ref2VA requires non-empty visual reference rows")
        clean_audio = self._latent_rows(
            raw_batch,
            "reference_audio_latent_rows",
            expected_rows=layout.num_condition_audio_rows,
            feature_dim=_AUDIO_LATENT_CHANNELS,
            device=device,
        )

        generator = reference_condition_generator(noise_seed, device=device)
        visual_shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                              for reference in references if reference.media_type != "audio")
        noise = keyframe_condition_noise(
            visual_shapes,
            self.transformer.patch_size,
            _VIDEO_LATENT_CHANNELS,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        condition_video = self.noise_scheduler.scale_noise(
            clean_video,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        ).to(torch.bfloat16)
        condition_audio = None if clean_audio is None else clean_audio.to(torch.bfloat16)
        return condition_video, condition_audio

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Prepare target-only DMD latents plus fixed ordered references."""
        # Reuse the established H3 target normalization/noise/text contract,
        # then replace its temporary T2VA layout with the Ref2VA layout.
        batch = MiniMaxH3Model.prepare_batch(
            self,
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
        if batch.latents is None or batch.audio_latents is None:
            raise RuntimeError("MiniMax H3 batch preparation did not produce paired target latents")
        mask = raw_batch.get("text_attention_mask")
        if not isinstance(mask, torch.Tensor):
            raise ValueError("A Ref2VA batch requires text_attention_mask")
        valid_text = mask[0].to(torch.bool)
        text_token_tags = self._text_token_tags(raw_batch, valid_text)
        references, noise_seed = self._prepared_references(batch)

        _, video_frames, _, latent_height, latent_width = batch.latents.shape
        layout = build_ref2va_packed_sequence(
            text_token_tags,
            references,
            video_frames,
            latent_height,
            latent_width,
            int(batch.audio_latents.shape[-1]),
            self.transformer.patch_size,
        )
        condition_video, condition_audio = self._prepare_condition_rows(
            raw_batch,
            references,
            layout,
            noise_seed,
        )
        batch.minimax_h3_layout = layout
        batch.minimax_h3_condition_video_rows = condition_video
        batch.minimax_h3_condition_audio_rows = condition_audio
        self._apply_sequence_capacity(batch, raw_batch)
        self._maybe_build_dense_packed_metadata(batch)
        self._maybe_build_vsa_metadata(batch)

        return self._pack_dmd_batch(batch)

    def _assert_reference_forward_contract(self, batch: TrainingBatch, attn_kind: str) -> None:
        """Guard the common row seam, including direct widened PDD heads."""
        backend = (getattr(self, "attention_backend_name", None) or envs.FASTVIDEO_ATTENTION_BACKEND.get())
        if backend == "VIDEO_SPARSE_ATTN_H3":
            if attn_kind != "vsa":
                raise ValueError("A VSA-backed Ref2VA role must be called with attn_kind='vsa'; "
                                 "dense metadata would bypass the trained P2 policy")
            metadata = batch.attn_metadata_vsa
            layout = batch.minimax_h3_layout
            if not isinstance(metadata, MiniMaxH3VSAMetadata) or not isinstance(layout, MiniMaxH3PackedLayout):
                raise RuntimeError("A VSA-backed Ref2VA forward requires final-layout P2 metadata")
            assert_ref2va_vsa_metadata(
                metadata,
                expected_reference_video_regions=sum(kind == "video"
                                                     for kind, _rows, _shape in layout.reference_segments),
                target_sparsity=float(self.training_config.vsa_sparsity),
                ref_keep_rate=self._resolved_vsa_ref_keep_rate(),
            )

    def _joint_forward_rows(
        self,
        noisy_latents: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool = True,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> Any:
        self._assert_reference_forward_contract(batch, attn_kind)
        return super()._joint_forward_rows(
            noisy_latents,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )


__all__ = ["MiniMaxH3Ref2VDMDModel"]
