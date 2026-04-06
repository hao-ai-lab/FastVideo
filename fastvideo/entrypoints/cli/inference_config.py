# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from fastvideo.api.overrides import apply_overrides, parse_cli_overrides
from fastvideo.api.parser import load_raw_config, parse_config
from fastvideo.api.schema import RunConfig, ServeConfig

_GENERATE_OVERRIDE_PREFIXES = ("generator.", "request.")
_SERVE_OVERRIDE_PREFIXES = (
    "generator.",
    "server.",
    "default_request.",
)


def build_generate_run_config(
    args: argparse.Namespace,
    overrides: list[str] | None = None,
) -> RunConfig:
    raw = _load_nested_config(getattr(args, "config", None))
    raw.setdefault("request", {})
    raw = _apply_dotted_overrides(
        raw,
        overrides,
        allowed_prefixes=_GENERATE_OVERRIDE_PREFIXES,
    )
    _ensure_generate_cli_defaults(raw)
    config = parse_config(RunConfig, raw)
    _validate_generate_prompt_sources(config)
    return config


def build_serve_config(
    args: argparse.Namespace,
    overrides: list[str] | None = None,
) -> ServeConfig:
    raw = _load_nested_config(getattr(args, "config", None))
    raw.setdefault("server", {})
    raw.setdefault("default_request", {})
    raw = _apply_dotted_overrides(
        raw,
        overrides,
        allowed_prefixes=_SERVE_OVERRIDE_PREFIXES,
    )
    return parse_config(ServeConfig, raw)


def _load_nested_config(path: str | None) -> dict[str, Any]:
    if not path:
        raise ValueError("Inference CLI requires --config PATH; use a nested config file "
                         "plus optional dotted overrides")

    raw = load_raw_config(path)
    if not isinstance(raw.get("generator"), Mapping):
        raise ValueError("Inference config must use the nested schema with a top-level "
                         "'generator' mapping")
    return deepcopy(dict(raw))


def _apply_dotted_overrides(
    raw: Mapping[str, Any],
    overrides: list[str] | None,
    *,
    allowed_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if not overrides:
        return deepcopy(dict(raw))

    parsed = parse_cli_overrides(overrides)
    for key in parsed:
        if "." not in key:
            raise ValueError("CLI overrides must use dotted config paths like "
                             "--request.sampling.seed 42")
        if not key.startswith(allowed_prefixes):
            allowed = ", ".join(allowed_prefixes)
            raise ValueError(f"Unsupported override path {key!r}. Allowed prefixes: {allowed}")
    return apply_overrides(raw, parsed)


def _ensure_generate_cli_defaults(raw: dict[str, Any]) -> None:
    request = raw.setdefault("request", {})
    output = request.setdefault("output", {})
    output.setdefault("return_frames", False)


def _validate_generate_prompt_sources(config: RunConfig) -> None:
    has_prompt = config.request.prompt is not None
    has_prompt_path = config.request.inputs.prompt_path is not None
    if not (has_prompt or has_prompt_path):
        raise ValueError("Either request.prompt or request.inputs.prompt_path must be provided")
    if has_prompt and has_prompt_path:
        raise ValueError("Cannot provide both request.prompt and request.inputs.prompt_path")


__all__ = [
    "build_generate_run_config",
    "build_serve_config",
]
