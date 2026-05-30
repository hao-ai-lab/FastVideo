# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastvideo.api.sampling_param import SamplingParam

_SIGNATURE_EXCLUDED_FIELDS = frozenset({
    "prompt",
    "prompt_path",
    "output_path",
    "output_video_name",
    "seed",
    "save_video",
    "return_frames",
})

_UNSUPPORTED_DYNAMIC_BATCH_FIELDS = frozenset({
    "image_path",
    "pil_image",
    "video_path",
    "mouse_cond",
    "keyboard_cond",
    "grid_sizes",
    "pose",
    "camera_states",
    "camera_trajectory",
    "action_list",
    "action_speed_list",
    "gt_latents",
    "conditioning_mask",
    "c2ws_plucker_emb",
    "refine_from",
    "stage1_video",
    "trajectory_type",
    "movement_distance",
    "camera_rotation",
    "ltx2_images",
    "ltx2_conditioning_latent_stage1",
    "ltx2_conditioning_latent_stage2",
    "ltx2_video_conditions",
    "init_audio",
    "inpaint_audio",
    "inpaint_mask",
    "continuation_state",
})

_UNSUPPORTED_EXTRA_KEYS = frozenset({
    "ltx2_audio_latents",
    "ltx2_audio_clean_latent",
    "ltx2_audio_denoise_mask",
    "audio_num_frames",
    "video_position_offset_sec",
})


@dataclass(frozen=True)
class BatchCompatibility:
    can_batch: bool
    reason: str | None = None


def resolution_key(request: Any) -> str:
    height = _first_scalar(getattr(request, "height", None))
    width = _first_scalar(getattr(request, "width", None))
    num_frames = _first_scalar(getattr(request, "num_frames", None))
    return f"{height}x{width}x{num_frames}"


def dynamic_batch_signature(
    request: SamplingParam,
    *,
    extra: dict[str, Any] | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Build a hashable compatibility signature for a generation request."""
    signature_items: list[tuple[str, Any]] = []
    for field in dataclasses.fields(request):
        if field.name in _SIGNATURE_EXCLUDED_FIELDS:
            continue
        signature_items.append((field.name, _freeze_signature_value(getattr(request, field.name, None))))
    if extra:
        signature_items.append(("extra", _freeze_signature_value(extra)))
    return tuple(signature_items)


def can_dynamic_batch(
    base: SamplingParam,
    candidate: SamplingParam,
    *,
    base_extra: dict[str, Any] | None = None,
    candidate_extra: dict[str, Any] | None = None,
) -> BatchCompatibility:
    """Return whether two FastVideo generation requests can be merged."""
    base_ready = _request_is_batchable(base, extra=base_extra)
    if not base_ready.can_batch:
        return base_ready
    candidate_ready = _request_is_batchable(candidate, extra=candidate_extra)
    if not candidate_ready.can_batch:
        return candidate_ready

    base_sig = dynamic_batch_signature(base, extra=base_extra)
    candidate_sig = dynamic_batch_signature(candidate, extra=candidate_extra)
    if base_sig == candidate_sig:
        return BatchCompatibility(can_batch=True)

    mismatch = _first_mismatch(base_sig, candidate_sig)
    return BatchCompatibility(can_batch=False, reason=mismatch or "signature_mismatch")


def _request_is_batchable(
    request: SamplingParam,
    *,
    extra: dict[str, Any] | None = None,
) -> BatchCompatibility:
    if not isinstance(request.prompt, str):
        return BatchCompatibility(can_batch=False, reason="prompt_type")
    if request.prompt_path is not None:
        return BatchCompatibility(can_batch=False, reason="prompt_path")
    if request.num_videos_per_prompt != 1:
        return BatchCompatibility(can_batch=False, reason="num_videos_per_prompt")
    if request.return_continuation_state:
        return BatchCompatibility(can_batch=False, reason="return_continuation_state")

    for name in _UNSUPPORTED_DYNAMIC_BATCH_FIELDS:
        value = getattr(request, name, None)
        if _is_present(value):
            return BatchCompatibility(can_batch=False, reason=name)

    if extra:
        unsupported = sorted(set(extra) & _UNSUPPORTED_EXTRA_KEYS)
        if unsupported:
            return BatchCompatibility(can_batch=False, reason=f"extra.{unsupported[0]}")

    return BatchCompatibility(can_batch=True)


def _freeze_signature_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature_value(item))
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, list | tuple):
        return tuple(_freeze_signature_value(item) for item in value)
    return repr(value)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if value is False:
        return False
    return not (isinstance(value, list | tuple | dict | set) and not value)


def _first_scalar(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value


def _first_mismatch(
    base_sig: tuple[tuple[str, Any], ...],
    candidate_sig: tuple[tuple[str, Any], ...],
) -> str | None:
    if len(base_sig) != len(candidate_sig):
        return "sampling_params"
    for (name, base_value), (candidate_name, candidate_value) in zip(base_sig, candidate_sig, strict=True):
        if name != candidate_name:
            return "sampling_params"
        if base_value != candidate_value:
            return f"sampling_params.{name}"
    return None
