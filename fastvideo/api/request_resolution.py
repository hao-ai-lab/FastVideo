# SPDX-License-Identifier: Apache-2.0
"""Resolution of the settings of a ``GenerationRequest`` against the model's sampling defaults.

``resolve_request`` records, for every sampling, runtime, and output field, whether the request set it or the
model's preset defaults (``SamplingParam.from_pretrained``) decided it. ``request_to_sampling_param`` builds the
``SamplingParam`` from the same two sources: the preset defaults, then the fields that the request set.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import Any

from fastvideo.api.resolution import ResolutionStep, ResolutionView, ResolvedRequest, resolve_generation_request
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import GenerationRequest, OutputConfig, RequestRuntimeConfig, SamplingConfig

_DEFAULTED_SECTIONS = (("sampling", SamplingConfig), ("runtime", RequestRuntimeConfig), ("output", OutputConfig))


def sampling_defaults_step(sampling_param: SamplingParam, defaults_name: str) -> ResolutionStep:
    """Build the step that sets each request field the request did not set to the model's default.

    A field is covered when ``sampling_param`` has an attribute of the same name; it is set when the request did
    not write it and the default differs from the schema default. The step's source name is
    ``fill_sampling_defaults[<defaults_name>]``.
    """

    def fill_sampling_defaults(view: ResolutionView) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for section, config_type in _DEFAULTED_SECTIONS:
            for config_field in fields(config_type):
                path = f"{section}.{config_field.name}"
                if view.is_explicit(path) or not hasattr(sampling_param, config_field.name):
                    continue
                default = getattr(sampling_param, config_field.name)
                if default != view.get(path):
                    values[path] = deepcopy(default)
        return values

    fill_sampling_defaults.__qualname__ = f"fill_sampling_defaults[{defaults_name}]"
    return fill_sampling_defaults


def resolve_request(request: GenerationRequest, *, model_path: str) -> ResolvedRequest:
    """Resolve the settings of ``request`` against the sampling defaults of ``model_path``.

    The defaults are the model's preset, as ``SamplingParam.from_pretrained`` builds them; the step's source name
    is ``preset <name>``, or the ``SamplingParam`` class name for a model without a preset.
    """
    from fastvideo.registry import get_preset_selection

    sampling_param = SamplingParam.from_pretrained(model_path)
    try:
        preset_name, _ = get_preset_selection(model_path)
    except (ValueError, RuntimeError):
        preset_name = None
    defaults_name = f"preset {preset_name}" if preset_name else type(sampling_param).__name__
    return resolve_generation_request(request, [sampling_defaults_step(sampling_param, defaults_name)])


__all__ = [
    "resolve_request",
    "sampling_defaults_step",
]
