# SPDX-License-Identifier: Apache-2.0
"""Track which GenerationRequest fields the user explicitly provided.

This module solves a specific problem: when translating a GenerationRequest into
a legacy SamplingParam, we need to distinguish user-provided values (which
should override model defaults) from schema defaults (which should NOT override
model defaults).

The approach:
1. At bind time, store the original raw dict and a baseline snapshot.
2. Patch __setattr__ on tracked dataclass types to record dirty field paths.
3. At access time, do a lazy 3-way merge: raw + baseline + current state,
   with dirty paths forcing inclusion even when current == baseline.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import dataclasses
from typing import Any, cast
from collections.abc import Callable

from fastvideo.api.schema import (
    ContinuationState,
    GenerationPlan,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    PlannedStage,
    RequestRuntimeConfig,
    RunConfig,
    SamplingConfig,
    ServeConfig,
)

EXPLICIT_REQUEST_ATTR = "_fastvideo_explicit_request"
ORIGINAL_REQUEST_STATE_ATTR = "_fastvideo_original_request_state"
_TRACKING_ROOT_ATTR = "_fastvideo_request_tracking_root"
_TRACKING_PATH_ATTR = "_fastvideo_request_tracking_path"
_TRACKING_PATCHED_ATTR = "_fastvideo_request_tracking_patched"
_DIRTY_PATHS_ATTR = "_fastvideo_dirty_paths"
_TRACKED_REQUEST_TYPES = (
    GenerationRequest,
    InputConfig,
    SamplingConfig,
    RequestRuntimeConfig,
    OutputConfig,
    ContinuationState,
    PlannedStage,
    GenerationPlan,
)


def bind_generation_request_raw(
    request: GenerationRequest,
    raw: Mapping[str, Any] | None,
) -> GenerationRequest:
    _ensure_request_tracking()
    # Disable dirty tracking during bind so tree walk doesn't record paths.
    object.__setattr__(request, _DIRTY_PATHS_ATTR, None)
    object.__setattr__(request, EXPLICIT_REQUEST_ATTR, deepcopy(dict(raw or {})))
    object.__setattr__(request, ORIGINAL_REQUEST_STATE_ATTR, _serialize_config(request))
    _set_tracking_roots(request, request, "")
    # Enable dirty tracking.
    object.__setattr__(request, _DIRTY_PATHS_ATTR, set())
    return request


def bind_run_config_raw(
    config: RunConfig,
    raw: Mapping[str, Any],
) -> RunConfig:
    request_raw = raw.get("request")
    if isinstance(request_raw, Mapping):
        bind_generation_request_raw(config.request, request_raw)
    return config


def bind_serve_config_raw(
    config: ServeConfig,
    raw: Mapping[str, Any],
) -> ServeConfig:
    default_request_raw = raw.get("default_request")
    if isinstance(default_request_raw, Mapping):
        bind_generation_request_raw(config.default_request, default_request_raw)
    elif "default_request" not in raw:
        bind_generation_request_raw(config.default_request, {})
    return config


def refresh_generation_request_raw(request: GenerationRequest, ) -> dict[str, Any] | None:
    raw = getattr(request, EXPLICIT_REQUEST_ATTR, None)
    baseline = getattr(request, ORIGINAL_REQUEST_STATE_ATTR, None)
    if not isinstance(raw, Mapping) or not isinstance(baseline, Mapping):
        return None

    dirty = getattr(request, _DIRTY_PATHS_ATTR, None) or frozenset()
    current = _serialize_config(request)
    merged = deepcopy(dict(raw))
    _merge_request_mutations(merged, dict(baseline), current, dirty)

    object.__setattr__(request, EXPLICIT_REQUEST_ATTR, merged)
    object.__setattr__(request, ORIGINAL_REQUEST_STATE_ATTR, current)
    object.__setattr__(request, _DIRTY_PATHS_ATTR, set())
    return merged


# ---------------------------------------------------------------------------
# 3-way merge: raw + baseline + current, with dirty-path forcing
# ---------------------------------------------------------------------------

_MISSING = object()


def _merge_request_mutations(
    merged: dict[str, Any],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    dirty: frozenset[str] | set[str],
    path_prefix: str = "",
) -> None:
    # Remove keys that were deleted from the current state.
    for key in set(merged) | set(baseline):
        if key not in current:
            merged.pop(key, None)

    for key in current:
        current_path = f"{path_prefix}.{key}" if path_prefix else key
        current_value = current[key]
        baseline_value = baseline.get(key, _MISSING)
        merged_value = merged.get(key, _MISSING)

        # Recurse into nested mappings.
        if isinstance(current_value, Mapping) and isinstance(baseline_value, Mapping):
            nested = (deepcopy(dict(merged_value)) if isinstance(merged_value, Mapping) else {})
            _merge_request_mutations(nested, baseline_value, current_value, dirty, current_path)
            if nested:
                merged[key] = nested
            else:
                merged.pop(key, None)
            continue

        # A field is explicitly set if:
        # - it's new (not in baseline),
        # - it changed from baseline, or
        # - its path was touched by __setattr__ (dirty).
        is_dirty = current_path in dirty
        if baseline_value is _MISSING or current_value != baseline_value or is_dirty:
            merged[key] = deepcopy(current_value)


# ---------------------------------------------------------------------------
# __setattr__ patching for dirty-path recording
# ---------------------------------------------------------------------------


def _ensure_request_tracking() -> None:
    for config_type in _TRACKED_REQUEST_TYPES:
        _patch_tracking_setattr(config_type)


def _patch_tracking_setattr(config_type: type[Any]) -> None:
    if getattr(config_type, _TRACKING_PATCHED_ATTR, False):
        return

    original_setattr = cast(
        Callable[[Any, str, Any], None],
        config_type.__setattr__,
    )
    field_names = {field.name for field in dataclasses.fields(config_type)}

    def _tracking_setattr(self: Any, name: str, value: Any) -> None:
        if name.startswith("_fastvideo_") or name not in field_names:
            original_setattr(self, name, value)
            return

        root = getattr(self, _TRACKING_ROOT_ATTR, None)
        if root is not None:
            dirty = getattr(root, _DIRTY_PATHS_ATTR, None)
            if isinstance(dirty, set):
                prefix = getattr(self, _TRACKING_PATH_ATTR, "")
                path = f"{prefix}.{name}" if prefix else name
                dirty.add(path)

        original_setattr(self, name, value)

    type.__setattr__(config_type, "__setattr__", _tracking_setattr)
    setattr(config_type, _TRACKING_PATCHED_ATTR, True)


# ---------------------------------------------------------------------------
# Tree walk to set tracking root/path on nested dataclasses
# ---------------------------------------------------------------------------


def _set_tracking_roots(
    root: GenerationRequest,
    obj: Any,
    prefix: str,
) -> None:
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return
    object.__setattr__(obj, _TRACKING_ROOT_ATTR, root)
    object.__setattr__(obj, _TRACKING_PATH_ATTR, prefix)
    for field in dataclasses.fields(obj):
        child = getattr(obj, field.name)
        child_path = f"{prefix}.{field.name}" if prefix else field.name
        if dataclasses.is_dataclass(child) and not isinstance(child, type):
            _set_tracking_roots(root, child, child_path)


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _serialize_config(config: Any) -> Any:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {field.name: _serialize_config(getattr(config, field.name)) for field in dataclasses.fields(config)}
    if isinstance(config, list):
        return [_serialize_config(item) for item in config]
    if isinstance(config, dict):
        return {key: _serialize_config(value) for key, value in config.items()}
    return deepcopy(config)


__all__ = [
    "EXPLICIT_REQUEST_ATTR",
    "ORIGINAL_REQUEST_STATE_ATTR",
    "bind_generation_request_raw",
    "bind_run_config_raw",
    "bind_serve_config_raw",
    "refresh_generation_request_raw",
]
