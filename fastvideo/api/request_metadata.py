# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from fastvideo.api.schema import GenerationRequest, RunConfig, ServeConfig

EXPLICIT_REQUEST_ATTR = "_fastvideo_explicit_request"


def bind_generation_request_raw(
    request: GenerationRequest,
    raw: Mapping[str, Any] | None,
) -> GenerationRequest:
    setattr(request, EXPLICIT_REQUEST_ATTR, deepcopy(dict(raw or {})))
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


__all__ = [
    "EXPLICIT_REQUEST_ATTR",
    "bind_generation_request_raw",
    "bind_run_config_raw",
    "bind_serve_config_raw",
]
