# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native Cosmos3 sequence packing (video subset).

Numerical-parity port of the official ``cosmos_framework`` data packer
(``cosmos_framework.data.vfm.sequence_packing.pack_input_sequence``) restricted
to the VIDEO generation path that the FastVideo Cosmos3 DiT consumes (T2V / I2V
/ T2I). It builds, per sample, two splits:

* a ``causal`` text split (prompt token ids, plus the trailing ``eos`` and
  ``start_of_generation`` markers the framework appends when a generation
  modality follows), and
* a ``full`` vision split (VAE latent patch tokens).

The 3D-MRoPE position ids ``[3, seq]`` are produced exactly like the framework:
text tokens broadcast a single monotone id across the (t, h, w) axes, the
temporal offset is bumped by ``temporal_modality_margin`` at the text->vision
boundary, and vision tokens lay out a (T, H, W) grid with spatial ids reset per
segment. Condition frames (I2V cond frame 0, T2I single conditioned frame, ...)
are kept in the packed sequence and rope grid but excluded from the MSE-loss /
timestep bookkeeping, mirroring the framework.

The output ``Cosmos3PackedSequence`` maps 1:1 onto the
``Cosmos3VFMTransformer.forward`` kwargs via :meth:`to_dit_kwargs`. This module
is pure torch/python; it imports no diffusers/transformers model classes.

Reference of record: ``cosmos_framework`` (NVIDIA), the parity oracle used by
``tests/local_tests/cosmos3/test_cosmos3_packing_parity.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.models.dits.cosmos3 import (
    compute_mrope_position_ids_text,
    compute_mrope_position_ids_vision,
)

__all__ = [
    "Cosmos3VisionItem",
    "Cosmos3SampleInputs",
    "Cosmos3PackedSequence",
    "pack_cosmos3_video_sequence",
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class Cosmos3VisionItem:
    """One vision latent for a sample.

    Args:
        latent: VAE latent ``[C, T, H, W]`` (a leading batch axis of size 1 is
            accepted and squeezed).
        condition_frame_indexes: Latent-frame indices that are *conditioned*
            (clean) rather than noisy. ``[]`` for T2V, ``[0]`` for I2V, and the
            single conditioned frame for T2I.
        fps: Frames-per-second for this clip; only used when
            ``enable_fps_modulation`` is set.
    """

    latent: torch.Tensor
    condition_frame_indexes: list[int] = field(default_factory=list)
    fps: float | None = None


@dataclass
class Cosmos3SoundItem:
    """One sound latent for a sample (t2vs).

    Args:
        latent: AVAE sound latent ``[C, T]`` (channels, temporal frames).
        condition_frame_indexes: Latent-frame indices that are *conditioned*
            (clean). ``[]`` for t2vs (all frames generated).
        fps: Sound latent FPS (``sound_latent_fps``, e.g. 25); only used when
            ``enable_fps_modulation`` is set.
    """

    latent: torch.Tensor
    condition_frame_indexes: list[int] = field(default_factory=list)
    fps: float | None = None


@dataclass
class Cosmos3ActionItem:
    """One action latent for a sample (action-conditioned world model).

    Args:
        latent: Action latent ``[T, action_dim]`` (per-frame action vectors).
        condition_frame_indexes: Frame indices kept clean (conditioning actions).
        domain_id: Embodiment domain id (scalar / ``[1]``) for the
            domain-aware action projection.
        fps: Action FPS; only used when ``enable_fps_modulation`` is set.
    """

    latent: torch.Tensor
    condition_frame_indexes: list[int] = field(default_factory=list)
    domain_id: int = 0
    fps: float | None = None


@dataclass
class Cosmos3SampleInputs:
    """Per-sample packing inputs (text prompt + vision item, +sound, +action)."""

    text_ids: list[int]
    vision: Cosmos3VisionItem
    timestep: float
    sound: Cosmos3SoundItem | None = None
    action: Cosmos3ActionItem | None = None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass
class Cosmos3PackedSequence:
    """Packed-sequence inputs consumed by ``Cosmos3VFMTransformer.forward``.

    Field names mirror the framework ``PackedSequence`` (+ its ``vision``
    ``ModalityData``) so the parity test can compare field-by-field.
    """

    # Sequence structure.
    sample_lens: list[int]
    split_lens: list[int]
    attn_modes: list[str]
    sequence_length: int
    is_image_batch: bool

    # Text modality.
    text_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor  # [3, sequence_length]

    # Vision modality.
    vision_tokens: list[torch.Tensor]
    vision_token_shapes: list[tuple[int, int, int]]
    vision_sequence_indexes: torch.Tensor
    vision_timesteps: torch.Tensor
    vision_mse_loss_indexes: torch.Tensor
    vision_noisy_frame_indexes: list[torch.Tensor]
    vision_condition_mask: list[torch.Tensor]
    fps_vision: torch.Tensor | None = None

    # Sound modality (t2vs); empty/None when no sound.
    sound_tokens: list[torch.Tensor] = field(default_factory=list)
    sound_token_shapes: list[tuple[int, int, int]] = field(default_factory=list)
    sound_sequence_indexes: torch.Tensor | None = None
    sound_timesteps: torch.Tensor | None = None
    sound_mse_loss_indexes: torch.Tensor | None = None
    sound_noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    sound_condition_mask: list[torch.Tensor] = field(default_factory=list)
    fps_sound: torch.Tensor | None = None

    # Action modality (action-conditioned world model); empty/None when no action.
    action_tokens: list[torch.Tensor] = field(default_factory=list)
    action_token_shapes: list[tuple[int, ...]] = field(default_factory=list)
    action_sequence_indexes: torch.Tensor | None = None
    action_timesteps: torch.Tensor | None = None
    action_mse_loss_indexes: torch.Tensor | None = None
    action_noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    action_condition_mask: list[torch.Tensor] = field(default_factory=list)
    action_domain_id: list[torch.Tensor] = field(default_factory=list)

    def to_dit_kwargs(self, device: torch.device | str | None = None) -> dict[str, Any]:
        """Return the kwargs dict for ``Cosmos3VFMTransformer.forward``.

        Packing is device-agnostic (ids/indexes/position-ids are built on CPU).
        When ``device`` is given, every tensor input is moved to it so the DiT
        forward runs on a single device (e.g. the model's GPU at inference).
        """

        def _mv(x: Any) -> Any:
            return x.to(device) if (device is not None and torch.is_tensor(x)) else x

        return dict(
            text_ids=_mv(self.text_ids),
            text_indexes=_mv(self.text_indexes),
            position_ids=_mv(self.position_ids),
            sequence_length=int(self.sequence_length),
            split_lens=list(self.split_lens),
            attn_modes=list(self.attn_modes),
            vision_tokens=[_mv(t) for t in self.vision_tokens],
            vision_token_shapes=list(self.vision_token_shapes),
            vision_sequence_indexes=_mv(self.vision_sequence_indexes),
            vision_timesteps=_mv(self.vision_timesteps),
            vision_mse_loss_indexes=_mv(self.vision_mse_loss_indexes),
            vision_noisy_frame_indexes=[_mv(t) for t in self.vision_noisy_frame_indexes],
            fps_vision=self.fps_vision,
            sound_tokens=[_mv(t) for t in self.sound_tokens],
            sound_token_shapes=list(self.sound_token_shapes),
            sound_sequence_indexes=_mv(self.sound_sequence_indexes),
            sound_timesteps=_mv(self.sound_timesteps),
            sound_mse_loss_indexes=_mv(self.sound_mse_loss_indexes),
            sound_noisy_frame_indexes=[_mv(t) for t in self.sound_noisy_frame_indexes],
            fps_sound=_mv(self.fps_sound),
            action_tokens=[_mv(t) for t in self.action_tokens],
            action_token_shapes=list(self.action_token_shapes),
            action_sequence_indexes=_mv(self.action_sequence_indexes),
            action_timesteps=_mv(self.action_timesteps),
            action_mse_loss_indexes=_mv(self.action_mse_loss_indexes),
            action_noisy_frame_indexes=[_mv(t) for t in self.action_noisy_frame_indexes],
            action_domain_id=[_mv(t) for t in self.action_domain_id],
        )


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def pack_cosmos3_video_sequence(
    samples: list[Cosmos3SampleInputs],
    special_tokens: dict[str, int],
    *,
    latent_patch_size: int = 2,
    include_end_of_generation_token: bool = False,
    temporal_modality_margin: int = 15_000,
    reset_spatial_ids: bool = True,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    initial_mrope_temporal_offset: int | float = 0,
) -> Cosmos3PackedSequence:
    """Pack prompts + vision latents into the Cosmos3 DiT packed-sequence inputs.

    Video subset of ``cosmos_framework`` ``pack_input_sequence`` under
    ``unified_3d_mrope``: each sample is ``[causal text, full vision]``.

    Args:
        samples: Per-sample text prompt token ids + vision item + timestep.
        special_tokens: Must contain ``eos_token_id`` and
            ``start_of_generation`` (and ``end_of_generation`` if
            ``include_end_of_generation_token``). ``bos_token_id`` is honored if
            present (prepended) to match the framework.
        latent_patch_size: Latent patch size used by the DiT.
        include_end_of_generation_token: Append the framework's end-of-generation
            marker after the vision split.
        temporal_modality_margin: Temporal-offset bump applied at the
            text->vision boundary (``unified_3d_mrope_temporal_modality_margin``).
        reset_spatial_ids: Reset vision spatial ids to 0 per segment.
        enable_fps_modulation: Use float, fps-scaled temporal positions.
        base_fps: Base FPS used when ``enable_fps_modulation``.
        temporal_compression_factor: VAE temporal compression factor.
        initial_mrope_temporal_offset: Per-sample starting temporal offset.

    Returns:
        A :class:`Cosmos3PackedSequence`.
    """
    assert "eos_token_id" in special_tokens, "special_tokens must contain eos_token_id"
    assert "start_of_generation" in special_tokens, "special_tokens must contain start_of_generation"
    if latent_patch_size < 1:
        raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}")

    # Build-time accumulators (concatenated across samples).
    sample_lens: list[int] = []
    split_lens: list[int] = []
    attn_modes: list[str] = []

    text_ids: list[int] = []
    text_indexes: list[int] = []
    position_id_blocks: list[torch.Tensor] = []  # each [3, n]

    vision_tokens: list[torch.Tensor] = []
    vision_token_shapes: list[tuple[int, int, int]] = []
    vision_sequence_indexes: list[int] = []
    vision_timesteps: list[float] = []
    vision_mse_loss_indexes: list[int] = []
    vision_noisy_frame_indexes: list[torch.Tensor] = []
    vision_condition_mask: list[torch.Tensor] = []
    fps_values: list[float] = []

    sound_tokens: list[torch.Tensor] = []
    sound_token_shapes: list[tuple[int, int, int]] = []
    sound_sequence_indexes: list[int] = []
    sound_timesteps: list[float] = []
    sound_mse_loss_indexes: list[int] = []
    sound_noisy_frame_indexes: list[torch.Tensor] = []
    sound_condition_mask: list[torch.Tensor] = []
    sound_fps_values: list[float] = []

    action_tokens: list[torch.Tensor] = []
    action_token_shapes: list[tuple[int, ...]] = []
    action_sequence_indexes: list[int] = []
    action_timesteps: list[float] = []
    action_mse_loss_indexes: list[int] = []
    action_noisy_frame_indexes: list[torch.Tensor] = []
    action_condition_mask: list[torch.Tensor] = []
    action_domain_id: list[torch.Tensor] = []

    curr = 0  # running position in the packed sequence
    is_image_batch = True

    for sample in samples:
        temporal_offset: int | float = initial_mrope_temporal_offset
        sample_len = 0

        # ---- 1. Text split (causal) ----
        if "bos_token_id" in special_tokens:
            shifted_text_ids = [special_tokens["bos_token_id"], *sample.text_ids]
        else:
            shifted_text_ids = list(sample.text_ids)
        # The video path always has a following generation modality, so the
        # framework appends eos + start_of_generation.
        shifted_text_ids = [*shifted_text_ids, special_tokens["eos_token_id"], special_tokens["start_of_generation"]]
        text_split_len = len(shifted_text_ids)

        text_ids.extend(shifted_text_ids)
        text_indexes.extend(range(curr, curr + text_split_len))

        text_mrope, temporal_offset = compute_mrope_position_ids_text(
            num_tokens=text_split_len,
            temporal_offset=int(temporal_offset),
        )
        position_id_blocks.append(text_mrope)

        attn_modes.append("causal")
        split_lens.append(text_split_len)
        curr += text_split_len
        sample_len += text_split_len

        # End of text modality: bump temporal offset before vision.
        temporal_offset += temporal_modality_margin
        # Sound shares the vision temporal start (parallel temporal positions).
        vision_start_temporal_offset = temporal_offset

        # ---- 2. Vision split (full) ----
        latent = sample.vision.latent
        latent = latent.squeeze(0) if latent.dim() == 5 else latent  # [C, T, H, W]
        _c, latent_t, latent_h, latent_w = latent.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        vision_tokens.append(sample.vision.latent)
        vision_token_shapes.append((latent_t, patch_h, patch_w))
        vision_sequence_indexes.extend(range(curr, curr + num_vision_tokens))

        condition_set = {idx for idx in sample.vision.condition_frame_indexes if 0 <= idx < latent_t}
        cond_mask = torch.zeros((latent_t, 1, 1), device=latent.device, dtype=latent.dtype)
        for frame_idx in condition_set:
            cond_mask[frame_idx, 0, 0] = 1.0
        vision_condition_mask.append(cond_mask)

        noisy_frames = torch.tensor(
            [idx for idx in range(latent_t) if idx not in condition_set],
            device=latent.device,
            dtype=torch.long,
        )
        vision_noisy_frame_indexes.append(noisy_frames)

        # MSE-loss indices + per-token timesteps cover only the noisy frames.
        frame_token_stride = patch_h * patch_w
        for frame_idx in range(latent_t):
            if frame_idx in condition_set:
                continue
            frame_start = curr + frame_idx * frame_token_stride
            vision_mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))
            vision_timesteps.extend([float(sample.timestep)] * frame_token_stride)

        vision_fps = sample.vision.fps if enable_fps_modulation else None
        if vision_fps is not None:
            fps_values.append(float(vision_fps))
        vision_mrope, temporal_offset = compute_mrope_position_ids_vision(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=temporal_offset,
            fps=vision_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
            enable_fps_modulation=enable_fps_modulation,
        )
        position_id_blocks.append(vision_mrope)

        curr += num_vision_tokens
        sample_len += num_vision_tokens

        # ---- 2a2. Action split: shares the vision "full" split ----
        # Mirrors framework ``_pack_action_tokens``: action latent [T, D] -> T
        # tokens (token shape (T,)), domain-aware, 3D-MRoPE at the vision temporal
        # offset with ``start_frame_offset=1`` (parallel to vision; tcf=1; does
        # not advance the offset).
        action_split_len = 0
        if sample.action is not None:
            action_latent = sample.action.latent  # [T, D]
            action_t = int(action_latent.shape[0])
            action_split_len = action_t

            action_tokens.append(action_latent)
            action_token_shapes.append((action_t, ))
            action_sequence_indexes.extend(range(curr, curr + action_t))
            action_domain_id.append(torch.tensor([int(sample.action.domain_id)], dtype=torch.long))

            a_cond_set = {idx for idx in sample.action.condition_frame_indexes if 0 <= idx < action_t}
            a_cond_mask = torch.zeros((action_t, 1), device=action_latent.device, dtype=action_latent.dtype)
            for fi in a_cond_set:
                a_cond_mask[fi, 0] = 1.0
            action_condition_mask.append(a_cond_mask)

            a_noisy = torch.tensor([idx for idx in range(action_t) if idx not in a_cond_set],
                                   device=action_latent.device,
                                   dtype=torch.long)
            action_noisy_frame_indexes.append(a_noisy)

            for fi in range(action_t):
                if fi in a_cond_set:
                    continue
                action_mse_loss_indexes.append(curr + fi)
                action_timesteps.append(float(sample.timestep))

            action_fps = sample.action.fps if enable_fps_modulation else None
            action_mrope, _ = compute_mrope_position_ids_vision(
                grid_t=action_t,
                grid_h=1,
                grid_w=1,
                temporal_offset=vision_start_temporal_offset,
                fps=action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,  # action is at frame rate
                base_temporal_compression_factor=temporal_compression_factor,
                enable_fps_modulation=enable_fps_modulation,
                start_frame_offset=1,
            )
            position_id_blocks.append(action_mrope)
            curr += action_t
            sample_len += action_t

        # ---- 2b. Sound split (t2vs): shares the vision "full" split ----
        # Mirrors framework ``_pack_sound_tokens``: sound latent [C, T] -> T
        # tokens (token shape (T,1,1)), packed right after vision, with 3D-MRoPE
        # temporal positions starting at the vision temporal offset (parallel to
        # vision, start_frame_offset=0, tcf=1) and NOT advancing it.
        sound_split_len = 0
        if sample.sound is not None:
            sound_latent = sample.sound.latent
            sound_latent = sound_latent.squeeze(0) if sound_latent.dim() == 3 else sound_latent  # [C, T]
            _sc, sound_t = sound_latent.shape
            sound_split_len = sound_t

            sound_tokens.append(sound_latent)
            sound_token_shapes.append((sound_t, 1, 1))
            sound_sequence_indexes.extend(range(curr, curr + sound_t))

            s_cond_set = {idx for idx in sample.sound.condition_frame_indexes if 0 <= idx < sound_t}
            s_cond_mask = torch.zeros((sound_t, 1), device=sound_latent.device, dtype=sound_latent.dtype)
            for fi in s_cond_set:
                s_cond_mask[fi, 0] = 1.0
            sound_condition_mask.append(s_cond_mask)

            s_noisy = torch.tensor([idx for idx in range(sound_t) if idx not in s_cond_set],
                                   device=sound_latent.device,
                                   dtype=torch.long)
            sound_noisy_frame_indexes.append(s_noisy)

            for fi in range(sound_t):
                if fi in s_cond_set:
                    continue
                sound_mse_loss_indexes.append(curr + fi)  # 1 token per sound frame
                sound_timesteps.append(float(sample.timestep))

            sound_fps = sample.sound.fps if enable_fps_modulation else None
            if sound_fps is not None:
                sound_fps_values.append(float(sound_fps))
            sound_mrope, _ = compute_mrope_position_ids_vision(
                grid_t=sound_t,
                grid_h=1,
                grid_w=1,
                temporal_offset=vision_start_temporal_offset,
                fps=sound_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,  # sound latent already at sound_latent_fps
                enable_fps_modulation=enable_fps_modulation,
                start_frame_offset=0,
            )
            position_id_blocks.append(sound_mrope)
            curr += sound_t
            sample_len += sound_t

        # ---- 3. Optional end-of-generation marker ----
        eov_len = 0
        if include_end_of_generation_token:
            assert "end_of_generation" in special_tokens, ("special_tokens must contain end_of_generation when "
                                                           "include_end_of_generation_token=True")
            text_ids.append(special_tokens["end_of_generation"])
            text_indexes.append(curr)
            eov_dtype = torch.float32 if enable_fps_modulation else torch.long
            eov_ids = torch.full((3, 1), temporal_offset, dtype=eov_dtype)
            position_id_blocks.append(eov_ids)
            temporal_offset += 1
            curr += 1
            eov_len = 1
            sample_len += 1

        # Vision + action + sound + any trailing eov marker share one "full" split.
        attn_modes.append("full")
        split_lens.append(num_vision_tokens + action_split_len + sound_split_len + eov_len)
        sample_lens.append(sample_len)

        if latent_t != 1:
            is_image_batch = False

    sequence_length = sum(sample_lens)

    # position_ids: float iff any block is float (fps modulation path).
    any_float = any(b.dtype.is_floating_point for b in position_id_blocks)
    if any_float:
        position_id_blocks = [b.to(torch.float32) for b in position_id_blocks]
    position_ids = torch.cat(position_id_blocks, dim=1)  # [3, sequence_length]

    timesteps_dtype = torch.float32
    return Cosmos3PackedSequence(
        sample_lens=sample_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        sequence_length=sequence_length,
        is_image_batch=is_image_batch,
        text_ids=torch.tensor(text_ids, dtype=torch.long),
        text_indexes=torch.tensor(text_indexes, dtype=torch.long),
        position_ids=position_ids,
        vision_tokens=vision_tokens,
        vision_token_shapes=vision_token_shapes,
        vision_sequence_indexes=torch.tensor(vision_sequence_indexes, dtype=torch.long),
        vision_timesteps=torch.tensor(vision_timesteps, dtype=timesteps_dtype),
        vision_mse_loss_indexes=torch.tensor(vision_mse_loss_indexes, dtype=torch.long),
        vision_noisy_frame_indexes=vision_noisy_frame_indexes,
        vision_condition_mask=vision_condition_mask,
        fps_vision=(torch.tensor(fps_values, dtype=torch.float32) if fps_values else None),
        sound_tokens=sound_tokens,
        sound_token_shapes=sound_token_shapes,
        sound_sequence_indexes=(torch.tensor(sound_sequence_indexes, dtype=torch.long) if sound_tokens else None),
        sound_timesteps=(torch.tensor(sound_timesteps, dtype=timesteps_dtype) if sound_tokens else None),
        sound_mse_loss_indexes=(torch.tensor(sound_mse_loss_indexes, dtype=torch.long) if sound_tokens else None),
        sound_noisy_frame_indexes=sound_noisy_frame_indexes,
        sound_condition_mask=sound_condition_mask,
        fps_sound=(torch.tensor(sound_fps_values, dtype=torch.float32) if sound_fps_values else None),
        action_tokens=action_tokens,
        action_token_shapes=action_token_shapes,
        action_sequence_indexes=(torch.tensor(action_sequence_indexes, dtype=torch.long) if action_tokens else None),
        action_timesteps=(torch.tensor(action_timesteps, dtype=timesteps_dtype) if action_tokens else None),
        action_mse_loss_indexes=(torch.tensor(action_mse_loss_indexes, dtype=torch.long) if action_tokens else None),
        action_noisy_frame_indexes=action_noisy_frame_indexes,
        action_condition_mask=action_condition_mask,
        action_domain_id=action_domain_id,
    )
