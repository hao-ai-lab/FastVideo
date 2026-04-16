# SPDX-License-Identifier: Apache-2.0
"""Track which GenerationRequest fields the user explicitly provided.

When translating a GenerationRequest into a legacy SamplingParam we must
distinguish user-provided values (which should override model defaults)
from schema defaults (which should NOT override model defaults).

The mechanism: a single ``_fastvideo_explicit_paths`` set stored on the
root ``GenerationRequest``. It holds dotted leaf paths (e.g.
``"sampling.guidance_scale"``) the user has touched, either via raw
config at bind time or via attribute assignment at runtime. A patched
``__setattr__`` on the request dataclass types records assignments into
this set.

The set holds leaf paths only. Nested dataclass or mapping assignments
are flattened to their leaves at record time.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, cast

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

EXPLICIT_PATHS_ATTR = "_fastvideo_explicit_paths"

_TRACKING_ROOT_ATTR = "_fastvideo_request_tracking_root"
_TRACKING_PATH_ATTR = "_fastvideo_request_tracking_path"
_TRACKING_PATCHED_ATTR = "_fastvideo_request_tracking_patched"

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
    """Install explicit-path tracking on *request*.

    *raw* is the parsed config dict (YAML/JSON/kwargs); every leaf key
    in it becomes an explicit path. Subsequent attribute assignments on
    *request* or its nested dataclasses are recorded automatically via a
    patched ``__setattr__``.
    """
    _ensure_request_tracking()
    # Disable recording while we walk the tree to install roots.
    object.__setattr__(request, EXPLICIT_PATHS_ATTR, None)
    _set_tracking_roots(request, request, "")
    paths: set[str] = set()
    _record_value_paths(raw or {}, "", paths)
    object.__setattr__(request, EXPLICIT_PATHS_ATTR, paths)
    return request


def bind_run_config_raw(
    config: RunConfig,
    raw: Mapping[str, Any],
) -> RunConfig:
    request_raw = raw.get("request")
    if isinstance(request_raw, Mapping):
        bind_generation_request_raw(config.request, request_raw)
    else:
        bind_generation_request_raw(config.request, {})
    return config


def bind_serve_config_raw(
    config: ServeConfig,
    raw: Mapping[str, Any],
) -> ServeConfig:
    default_request_raw = raw.get("default_request")
    if isinstance(default_request_raw, Mapping):
        bind_generation_request_raw(config.default_request, default_request_raw)
    else:
        bind_generation_request_raw(config.default_request, {})
    return config


def get_explicit_paths(request: GenerationRequest) -> frozenset[str]:
    """Return a snapshot of the explicit paths set on *request*."""
    paths = getattr(request, EXPLICIT_PATHS_ATTR, None)
    if isinstance(paths, set | frozenset):
        return frozenset(paths)
    return frozenset()


def reset_tracking_roots(request: GenerationRequest) -> None:
    """Re-install tracking roots after a deepcopy or manual clone.

    The paths set itself deepcopies correctly; we only need to repoint
    the tracking root on nested dataclasses at the new root.
    """
    _ensure_request_tracking()
    _set_tracking_roots(request, request, "")


# ---------------------------------------------------------------------------
# Path recording
# ---------------------------------------------------------------------------


def _record_value_paths(
    value: Any,
    prefix: str,
    out: set[str],
) -> None:
    """Add every leaf path under *value* to *out*.

    A leaf is any terminal value (non-dataclass, non-mapping, or empty
    mapping/dataclass). ``prefix`` is the dotted path at which *value*
    sits. When called with an empty ``prefix`` (the root), leaves are
    recorded at their own key.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        dc_fields = dataclasses.fields(value)
        if not dc_fields:
            if prefix:
                out.add(prefix)
            return
        for field in dc_fields:
            child = getattr(value, field.name)
            path = f"{prefix}.{field.name}" if prefix else field.name
            _record_value_paths(child, path, out)
        return
    if isinstance(value, Mapping):
        if not value:
            if prefix:
                out.add(prefix)
            return
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            _record_value_paths(child, path, out)
        return
    if prefix:
        out.add(prefix)


# ---------------------------------------------------------------------------
# __setattr__ patching
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

        original_setattr(self, name, value)

        root = getattr(self, _TRACKING_ROOT_ATTR, None)
        if root is None:
            return
        paths = getattr(root, EXPLICIT_PATHS_ATTR, None)
        if not isinstance(paths, set):
            return

        prefix = getattr(self, _TRACKING_PATH_ATTR, "")
        path = f"{prefix}.{name}" if prefix else name
        # Wholesale dataclass replacement: install roots on the new
        # instance so its future mutations are tracked too.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            _set_tracking_roots(root, value, path)
        _record_value_paths(value, path, paths)

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


__all__ = [
    "EXPLICIT_PATHS_ATTR",
    "bind_generation_request_raw",
    "bind_run_config_raw",
    "bind_serve_config_raw",
    "get_explicit_paths",
    "reset_tracking_roots",
]
