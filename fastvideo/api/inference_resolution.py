# SPDX-License-Identifier: Apache-2.0
"""Resolution of an inference ``GeneratorConfig`` into the frozen values that the run starts with.

``resolve_inference_config`` runs ``INFERENCE_RESOLUTION_STEPS`` in order through
:func:`fastvideo.api.resolution.resolve_generator_config`, so every value that a step decides is recorded with the
step's name. ``generator_config_to_fastvideo_args`` flattens the resolved values into ``FastVideoArgs``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from fastvideo.api.resolution import ResolutionStep, ResolvedGeneratorConfig, resolve_generator_config
from fastvideo.api.schema import GeneratorConfig

INFERENCE_RESOLUTION_STEPS: tuple[ResolutionStep, ...] = ()


def resolve_inference_config(config: GeneratorConfig | Mapping[str, Any]) -> ResolvedGeneratorConfig:
    """Resolve an inference config and freeze the result.

    A mapping is the raw nested input, and every leaf that it contains counts as written by the user. A
    ``GeneratorConfig`` object does not record which fields were written, so its fields that differ from the
    schema defaults count as written.
    """
    raw = config if isinstance(config, Mapping) else written_fields(config)
    return resolve_generator_config(raw, INFERENCE_RESOLUTION_STEPS)


def written_fields(config: GeneratorConfig) -> dict[str, Any]:
    """Raw nested mapping of the fields of ``config`` that differ from the schema defaults, plus ``model_path``.

    Parsing the mapping gives back a config equal to ``config``.
    """
    return {
        "model_path": config.model_path,
        **_non_default_fields(config, GeneratorConfig(model_path=config.model_path))
    }


def _non_default_fields(value: Any, default: Any) -> dict[str, Any]:
    """Fields of the dataclass ``value`` that differ from the same fields of ``default``, recursively.

    A nested config that is set while its default is ``None`` is kept, with its own non-default fields, so that
    parsing recreates it.
    """
    written: dict[str, Any] = {}
    for config_field in fields(value):
        current = getattr(value, config_field.name)
        base = getattr(default, config_field.name)
        if is_dataclass(current) and is_dataclass(base):
            nested = _non_default_fields(current, base)
            if nested:
                written[config_field.name] = nested
        elif is_dataclass(current):
            nested_type: Any = type(current)
            written[config_field.name] = _non_default_fields(current, nested_type())
        elif current != base:
            written[config_field.name] = current
    return written


__all__ = [
    "INFERENCE_RESOLUTION_STEPS",
    "resolve_inference_config",
    "written_fields",
]
