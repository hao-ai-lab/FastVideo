# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import dataclasses
from typing import Any

from fastvideo.api.schema import GenerationRequest, RunConfig, ServeConfig

EXPLICIT_REQUEST_ATTR = "_fastvideo_explicit_request"
ORIGINAL_REQUEST_STATE_ATTR = "_fastvideo_original_request_state"
_MISSING = object()


def bind_generation_request_raw(
    request: GenerationRequest,
    raw: Mapping[str, Any] | None,
) -> GenerationRequest:
    setattr(request, EXPLICIT_REQUEST_ATTR, deepcopy(dict(raw or {})))
    setattr(request, ORIGINAL_REQUEST_STATE_ATTR, _serialize_config(request))
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
    for key in current:
        current_value = current[key]
        baseline_value = baseline.get(key, _MISSING)

        if isinstance(current_value, Mapping) and isinstance(baseline_value, Mapping):
            nested = deepcopy(merged.get(key, {})) if isinstance(merged.get(key), Mapping) else {}
            before = deepcopy(nested)
            _merge_request_mutations(nested, baseline_value, current_value)
            if nested != before or key in merged:
                merged[key] = nested
            continue

        if baseline_value is _MISSING or current_value != baseline_value:
            merged[key] = deepcopy(current_value)


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
