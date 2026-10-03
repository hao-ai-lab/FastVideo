# SPDX-License-Identifier: Apache-2.0
"""Resolved runtime configs for weight-free stage tests."""

from collections.abc import Mapping
from typing import Any

from fastvideo.api.inference_resolution import fill_runtime_defaults
from fastvideo.api.resolution import ResolvedGeneratorConfig, resolve_generator_config


def make_resolved_config(pipeline_config: Any = None, raw: Mapping[str, Any] | None = None) -> ResolvedGeneratorConfig:
    """Resolve ``raw``, a nested ``GeneratorConfig`` mapping, with ``pipeline_config`` as the model definition.

    Only the runtime-defaults step runs, so no checkpoint or registry lookup happens. ``pipeline_config`` stands in
    for the ``PipelineConfig`` that materialization builds; a test passes a small stand-in that holds only the model
    definition data that the stage under test reads.
    """
    return resolve_generator_config(
        {
            "model_path": "unused/for-stage-tests",
            **(raw or {})
        },
        (fill_runtime_defaults, ),
        materialize=lambda _: pipeline_config,
    )
