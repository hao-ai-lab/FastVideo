# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
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
_TRACKING_SUSPENDED_ATTR = "_fastvideo_request_tracking_suspended"
_TRACKING_PATCHED_ATTR = "_fastvideo_request_tracking_patched"
_MISSING = object()
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


class _TrackedRequestDict(dict):

    def __init__(
        self,
        initial: Mapping[str, Any] | None = None,
        *,
        root: GenerationRequest,
        path: tuple[Any, ...],
    ) -> None:
        super().__init__()
        object.__setattr__(self, _TRACKING_ROOT_ATTR, root)
        object.__setattr__(self, _TRACKING_PATH_ATTR, path)
        with _suspend_tracking(root):
            for key, value in dict(initial or {}).items():
                dict.__setitem__(
                    self,
                    key,
                    _track_request_value(root, path + (key, ), value),
                )

    def __setitem__(self, key: Any, value: Any) -> None:
        root = getattr(self, _TRACKING_ROOT_ATTR, None)
        path = getattr(self, _TRACKING_PATH_ATTR, ())
        tracked_value = _track_request_value(root, path + (key, ), value)
        dict.__setitem__(self, key, tracked_value)
        _set_explicit_request_value(root, path + (key, ), tracked_value)

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, key)
        _delete_explicit_request_value(
            getattr(self, _TRACKING_ROOT_ATTR, None),
            getattr(self, _TRACKING_PATH_ATTR, ()) + (key, ),
        )

    def clear(self) -> None:
        for key in list(self.keys()):
            del self[key]

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if key in self:
            value = dict.__getitem__(self, key)
            del self[key]
            return value
        if default is not _MISSING:
            return default
        raise KeyError(key)

    def popitem(self) -> tuple[Any, Any]:
        key = next(iter(self))
        value = dict.__getitem__(self, key)
        del self[key]
        return key, value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def rebind(self, root: GenerationRequest, path: tuple[Any, ...]) -> None:
        object.__setattr__(self, _TRACKING_ROOT_ATTR, root)
        object.__setattr__(self, _TRACKING_PATH_ATTR, path)
        with _suspend_tracking(root):
            for key, value in list(self.items()):
                tracked_value = _track_request_value(root, path + (key, ), value)
                if tracked_value is not value:
                    dict.__setitem__(self, key, tracked_value)


def bind_generation_request_raw(
    request: GenerationRequest,
    raw: Mapping[str, Any] | None,
) -> GenerationRequest:
    _ensure_request_tracking()
    setattr(request, EXPLICIT_REQUEST_ATTR, deepcopy(dict(raw or {})))
    setattr(request, ORIGINAL_REQUEST_STATE_ATTR, _serialize_config(request))
    _bind_request_tracking(request)
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

    current = _serialize_config(request)
    merged = deepcopy(dict(raw))
    _merge_request_mutations(
        merged,
        dict(baseline),
        current,
    )
    setattr(request, EXPLICIT_REQUEST_ATTR, merged)
    setattr(request, ORIGINAL_REQUEST_STATE_ATTR, current)
    return merged


def _merge_request_mutations(
    merged: dict[str, Any],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    for key in set(merged) | set(baseline):
        if key not in current:
            merged.pop(key, None)

    for key in current:
        current_value = current[key]
        baseline_value = baseline.get(key, _MISSING)
        merged_value = merged.get(key, _MISSING)

        if isinstance(current_value, Mapping) and isinstance(baseline_value, Mapping):
            nested = (deepcopy(dict(merged_value)) if isinstance(merged_value, Mapping) else {})
            _merge_request_mutations(nested, baseline_value, current_value)
            if nested:
                merged[key] = nested
            else:
                merged.pop(key, None)
            continue

        if baseline_value is _MISSING or current_value != baseline_value:
            merged[key] = deepcopy(current_value)


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
        if root is None or getattr(root, _TRACKING_SUSPENDED_ATTR, False):
            original_setattr(self, name, value)
            return

        path = getattr(self, _TRACKING_PATH_ATTR, ())
        tracked_value = _track_request_value(root, path + (name, ), value)
        original_setattr(self, name, tracked_value)
        _set_explicit_request_value(root, path + (name, ), tracked_value)

    type.__setattr__(config_type, "__setattr__", _tracking_setattr)
    setattr(config_type, _TRACKING_PATCHED_ATTR, True)


def _bind_request_tracking(request: GenerationRequest) -> None:
    object.__setattr__(request, _TRACKING_SUSPENDED_ATTR, True)
    try:
        _track_request_value(request, (), request)
    finally:
        object.__setattr__(request, _TRACKING_SUSPENDED_ATTR, False)


def _track_request_value(
    root: GenerationRequest | None,
    path: tuple[Any, ...],
    value: Any,
) -> Any:
    if root is None:
        return value

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _patch_tracking_setattr(type(value))
        object.__setattr__(value, _TRACKING_ROOT_ATTR, root)
        object.__setattr__(value, _TRACKING_PATH_ATTR, path)
        with _suspend_tracking(root):
            for field in dataclasses.fields(value):
                current = getattr(value, field.name)
                tracked_current = _track_request_value(
                    root,
                    path + (field.name, ),
                    current,
                )
                if tracked_current is not current:
                    object.__setattr__(value, field.name, tracked_current)
        return value

    if isinstance(value, _TrackedRequestDict):
        value.rebind(root, path)
        return value

    if isinstance(value, dict):
        return _TrackedRequestDict(value, root=root, path=path)

    return value


@contextmanager
def _suspend_tracking(root: GenerationRequest | None):
    if root is None:
        yield
        return

    was_suspended = getattr(root, _TRACKING_SUSPENDED_ATTR, False)
    object.__setattr__(root, _TRACKING_SUSPENDED_ATTR, True)
    try:
        yield
    finally:
        object.__setattr__(root, _TRACKING_SUSPENDED_ATTR, was_suspended)


def _set_explicit_request_value(
    root: GenerationRequest | None,
    path: tuple[Any, ...],
    value: Any,
) -> None:
    if root is None or getattr(root, _TRACKING_SUSPENDED_ATTR, False) or not path:
        return

    raw = getattr(root, EXPLICIT_REQUEST_ATTR, None)
    if not isinstance(raw, dict):
        return

    cursor = raw
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = _serialize_config(value)


def _delete_explicit_request_value(
    root: GenerationRequest | None,
    path: tuple[Any, ...],
) -> None:
    if root is None or getattr(root, _TRACKING_SUSPENDED_ATTR, False) or not path:
        return

    raw = getattr(root, EXPLICIT_REQUEST_ATTR, None)
    if not isinstance(raw, dict):
        return

    _delete_explicit_request_path(raw, path)


def _delete_explicit_request_path(
    raw: dict[str, Any],
    path: tuple[Any, ...],
) -> bool:
    key = path[0]
    if len(path) == 1:
        raw.pop(key, None)
        return not raw

    child = raw.get(key)
    if not isinstance(child, dict):
        return not raw

    if _delete_explicit_request_path(child, path[1:]):
        raw.pop(key, None)
    return not raw


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
