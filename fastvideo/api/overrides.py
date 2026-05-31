# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from typing import Any
from collections.abc import Mapping

import yaml

from fastvideo.api.errors import ConfigValidationError


def parse_cli_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ``--dotted.key value`` style overrides into a flat mapping."""
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(overrides):
        token = overrides[index]
        if not token.startswith("--"):
            raise ValueError(f"Expected --dotted.key, got {token!r}")

        key = token[2:]
        if not key:
            raise ValueError("Override key cannot be empty")

        if "=" in key:
            key, raw_value = key.split("=", 1)
        else:
            index += 1
            if index >= len(overrides):
                raise ValueError(f"Missing value for override {token!r}")
            raw_value = overrides[index]

        parsed[_normalize_override_key(key)] = _cast_override_value(raw_value)
        index += 1

    return parsed


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with dotted-key overrides applied."""
    merged = deepcopy(dict(config))
    for dotted_key, value in overrides.items():
        _apply_single_override(merged, dotted_key, value)
    return merged


def normalize_overrides(overrides: list[str] | Mapping[str, Any] | None, ) -> dict[str, Any] | None:
    """Normalize a CLI list or mapping of overrides into a flat dict."""
    if not overrides:
        return None
    if isinstance(overrides, list):
        return parse_cli_overrides(overrides)
    return dict(overrides)


def _apply_single_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override path {dotted_key!r}")

    cursor = config
    for depth, part in enumerate(parts[:-1]):
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        elif not isinstance(existing, dict):
            raise ConfigValidationError(
                ".".join(parts[:depth + 1]),
                "cannot apply nested override through a non-mapping value",
            )
        cursor = existing

    cursor[parts[-1]] = value


def _cast_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    if raw.startswith("[") or raw.startswith("{"):
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            pass

    return raw


def _normalize_override_key(key: str) -> str:
    return key.replace("-", "_")


__all__ = ["apply_overrides", "normalize_overrides", "parse_cli_overrides"]
