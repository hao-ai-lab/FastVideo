# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 joint text-to-video-and-audio training (full-tuning and LoRA tuning) plugin"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.dataset.minimax_h3_ref2va_dataset import (
    MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH,
    MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH,
    build_minimax_h3_ref2va_dataloader,
)
from fastvideo.distributed import get_sp_group
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedLayout,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    MiniMaxH3LoraModel,
    MiniMaxH3Model,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.train.models.base import NoisePrediction

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig

_REF_VISUAL_ANCHOR_KEY = "ref_visual_anchor"
_REF_AUDIO_ANCHOR_KEY = "ref_audio_anchor"
_REFERENCE_FIELDS = {
    "media_type",
    "has_audio",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
}


def _non_negative_int(value: Any, *, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"Reference {index} field {field!r} must be a non-negative integer, got {value!r}")
    return int(value)


def _restore_prepared_references(infos: Any) -> list[MiniMaxH3PreparedReference]:
    """Restore only the ordered geometry consumed by the packed-layout builder."""
    if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
        raise ValueError("A Ref2VA batch requires exactly one info_list mapping")
    raw_references = infos[0].get("references")
    if not isinstance(raw_references, list):
        raise TypeError("Ref2VA info_list[0].references must be an ordered list")
    if len(raw_references) > 12:
        raise ValueError("MiniMax H3 supports at most 12 references")

    references: list[MiniMaxH3PreparedReference] = []
    for index, spec in enumerate(raw_references):
        if not isinstance(spec, dict) or set(spec) != _REFERENCE_FIELDS:
            fields = sorted(spec) if isinstance(spec, dict) else type(spec).__name__
            raise ValueError(
                f"Prepared reference {index} must contain exactly {sorted(_REFERENCE_FIELDS)}, got {fields}")
        media_type = spec["media_type"]
        if media_type not in {"image", "video", "audio"}:
            raise ValueError(f"Unsupported prepared reference type at index {index}: {media_type!r}")
        has_audio = spec["has_audio"]
        if not isinstance(has_audio, bool):
            raise TypeError(f"Reference {index} field 'has_audio' must be bool")

        num_latent_frames = _non_negative_int(spec["num_latent_frames"], field="num_latent_frames", index=index)
        latent_height = _non_negative_int(spec["latent_height"], field="latent_height", index=index)
        latent_width = _non_negative_int(spec["latent_width"], field="latent_width", index=index)
        num_audio_latents = _non_negative_int(spec["num_audio_latents"], field="num_audio_latents", index=index)

        if media_type == "audio":
            if not has_audio or (num_latent_frames, latent_height, latent_width) != (0, 0, 0):
                raise ValueError(f"Standalone audio reference {index} has an invalid canonical contract")
        else:
            if min(num_latent_frames, latent_height, latent_width) <= 0:
                raise ValueError(f"Visual reference {index} has incomplete latent geometry")
            if media_type == "image" and has_audio:
                raise ValueError(f"Image reference {index} cannot carry audio")

        if has_audio and num_audio_latents <= 0:
            raise ValueError(f"Audio-bearing reference {index} has no audio latents")
        if not has_audio and num_audio_latents != 0:
            raise ValueError(f"Silent reference {index} must have zero audio latents")

        references.append(
            MiniMaxH3PreparedReference(
                media_type=media_type,
                has_audio=has_audio,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
            ))

    media_types = [reference.media_type for reference in references]
    if media_types.count("image") > 9:
        raise ValueError("MiniMax H3 supports at most 9 image references")
    if media_types.count("video") > 3:
        raise ValueError("MiniMax H3 supports at most 3 video references")
    if media_types.count("audio") > 3:
        raise ValueError("MiniMax H3 supports at most 3 standalone audio references")
    if references and all(media_type == "audio" for media_type in media_types):
        raise ValueError("A non-empty Ref2VA batch requires at least one image or video reference")
    return references


def _valid_text_token_tags(raw_batch: dict[str, Any]) -> torch.Tensor:
    tags = raw_batch.get("text_token_tags")
    mask = raw_batch.get("text_attention_mask")
    if not isinstance(tags, torch.Tensor) or tags.ndim != 2 or tags.shape[0] != 1:
        raise ValueError("text_token_tags must have shape [1, length]")
    if not isinstance(mask, torch.Tensor) or mask.shape != tags.shape:
        raise ValueError("text_attention_mask must align one-to-one with text_token_tags")
    selected = tags[0, mask[0].to(torch.bool)]
    if selected.numel() == 0:
        raise ValueError("A Ref2VA batch requires at least one conditioning token")
    if not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("text_token_tags may contain only vision=0 and text=1")
    return selected.to(device="cpu", dtype=torch.long).contiguous()


def _unbatch_anchor(
    raw_batch: dict[str, Any],
    key: str,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    value = raw_batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"A Ref2VA batch requires tensor {key!r}")
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
        raise ValueError(f"{key} must have shape [1, rows, {width}], got {tuple(value.shape)}")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} must contain finite floating-point values")
    return value[0].to(device=device, dtype=torch.float32).contiguous()


class MiniMaxH3Ref2VAModel(MiniMaxH3Model):
    """Adapt the Ref2VA transformer partition to target-only joint flow loss."""

    _transformer_module_type = "transformer_ref"

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Build the batch-one loader without truncating Ref2VA Qwen tokens."""
        self.sp_group = get_sp_group()
        _dataset, self.dataloader = build_minimax_h3_ref2va_dataloader(
            training_config.data.data_path,
            int(training_config.data.train_batch_size),
            int(training_config.data.dataloader_num_workers),
            drop_last=True,
            seed=int(training_config.data.seed or 0),
        )
        self.start_step = 0

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Reuse target noise preparation, then install ordered Ref2VA conditions."""
        batch = super().prepare_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
        references = _restore_prepared_references(batch.infos)
        text_token_tags = _valid_text_token_tags(raw_batch)
        if batch.encoder_hidden_states is None or batch.encoder_hidden_states.shape[1] != text_token_tags.numel():
            raise ValueError("Filtered text_token_tags do not align with encoder_hidden_states")

        if batch.raw_latent_shape is None or len(batch.raw_latent_shape) != 5:
            raise RuntimeError("Parent MiniMax H3 preparation did not preserve target video geometry")
        if batch.audio_latents is None:
            raise RuntimeError("Parent MiniMax H3 preparation did not preserve target audio latents")
        _, _, num_video_latents, latent_height, latent_width = batch.raw_latent_shape
        num_audio_latents = int(batch.audio_latents.shape[-1])
        patch_size = tuple(self.transformer.patch_size)

        if references:
            layout = build_ref2va_packed_sequence(
                text_token_tags,
                references,
                num_video_latents,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
            )
        else:
            if bool((text_token_tags != MINIMAX_H3_TEXT_TAG).any()):
                raise ValueError("A zero-reference batch cannot contain visual Qwen token tags")
            layout = build_packed_sequence(
                text_token_tags,
                num_video_latents,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
            )

        visual_anchor = _unbatch_anchor(
            raw_batch,
            _REF_VISUAL_ANCHOR_KEY,
            MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH,
            self.device,
        )
        audio_anchor = _unbatch_anchor(
            raw_batch,
            _REF_AUDIO_ANCHOR_KEY,
            MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH,
            self.device,
        )
        if visual_anchor.shape[0] != layout.num_condition_video_rows:
            raise ValueError("ref_visual_anchor row count does not match the ordered reference layout: "
                             f"{visual_anchor.shape[0]} != {layout.num_condition_video_rows}")
        if audio_anchor.shape[0] != layout.num_condition_audio_rows:
            raise ValueError("ref_audio_anchor row count does not match the ordered reference layout: "
                             f"{audio_anchor.shape[0]} != {layout.num_condition_audio_rows}")

        batch.minimax_h3_layout = layout
        batch.input_kwargs = dict(batch.input_kwargs or {})
        batch.input_kwargs[_REF_VISUAL_ANCHOR_KEY] = visual_anchor
        batch.input_kwargs[_REF_AUDIO_ANCHOR_KEY] = audio_anchor
        return batch

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
        """Prepend fixed reference rows and return only target video/audio flow."""
        del timestep
        if not conditional or cfg_uncond is not None:
            raise ValueError("MiniMaxH3Ref2VAModel predicts one conditional sample")
        if attn_kind != "dense":
            raise ValueError("MiniMaxH3Ref2VAModel supports dense attention for training")
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_layout")
        if batch.audio_noisy_model_input is None or batch.encoder_hidden_states is None:
            raise RuntimeError("prepare_batch() must set audio and text transformer inputs")
        if batch.timesteps is None or batch.audio_timesteps is None:
            raise RuntimeError("prepare_batch() must set video and audio timesteps")
        extras = batch.input_kwargs
        if not isinstance(extras, dict):
            raise RuntimeError("prepare_batch() must set Ref2VA anchor inputs")
        visual_anchor = extras.get(_REF_VISUAL_ANCHOR_KEY)
        audio_anchor = extras.get(_REF_AUDIO_ANCHOR_KEY)
        if not isinstance(visual_anchor, torch.Tensor) or not isinstance(audio_anchor, torch.Tensor):
            raise RuntimeError("prepare_batch() must set both Ref2VA anchor tensors")

        dtype = torch.bfloat16
        device = self.device
        target_video_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
        target_video_rows = patchify_video_latents(target_video_bcthw, tuple(self.transformer.patch_size))
        target_audio = batch.audio_noisy_model_input.to(device=device, dtype=dtype)
        num_target_audio_latents = int(target_audio.shape[-1])
        target_audio_rows = target_audio.permute(0, 1, 3, 2).reshape(
            -1,
            MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH,
        )
        if target_video_rows.shape[1] != MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH:
            raise ValueError(f"Unexpected target video row width: {target_video_rows.shape[1]}")

        video_rows = torch.cat((visual_anchor.to(device=device, dtype=dtype), target_video_rows), dim=0)
        audio_rows = torch.cat((audio_anchor.to(device=device, dtype=dtype), target_audio_rows), dim=0)
        if video_rows.shape[0] != layout.video_indices.numel():
            raise ValueError("Packed Ref2VA video row count does not match its layout")
        if audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError("Packed Ref2VA audio row count does not match its layout")

        video_timestep = float(batch.timesteps[0].item())
        audio_timestep = float(batch.audio_timesteps[0].item())
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
            condition_video_timestep=max(video_timestep, MINIMAX_H3_KEYFRAME_NOISE_AUG),
            condition_audio_timestep=1.0,
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

        if video_velocity.ndim != 3 or video_velocity.shape[1] != video_rows.shape[0]:
            raise ValueError(f"Unexpected Ref2VA video output shape: {tuple(video_velocity.shape)}")
        if audio_velocity.ndim != 3 or audio_velocity.shape[1] != audio_rows.shape[0]:
            raise ValueError(f"Unexpected Ref2VA audio output shape: {tuple(audio_velocity.shape)}")
        target_video_velocity = video_velocity[:, layout.num_condition_video_rows:]
        target_audio_velocity = audio_velocity[:, layout.num_condition_audio_rows:]

        _, channels, num_video_latents, latent_height, latent_width = target_video_bcthw.shape
        video_prediction = unpatchify_video_tokens(
            target_video_velocity,
            num_video_latents,
            latent_height,
            latent_width,
            channels,
            tuple(self.transformer.patch_size),
        ).permute(0, 2, 1, 3, 4)
        audio_prediction = unpack_audio_tokens(target_audio_velocity[0], num_target_audio_latents)[None]
        return -video_prediction, -audio_prediction


class MiniMaxH3Ref2VALoraModel(
        MiniMaxH3LoraModel,
        MiniMaxH3Ref2VAModel,
):
    """Use Ref2VA data semantics with LoRA-only
    transformer tuning."""


__all__ = ["MiniMaxH3Ref2VALoraModel", "MiniMaxH3Ref2VAModel"]
