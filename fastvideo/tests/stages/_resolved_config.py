# SPDX-License-Identifier: Apache-2.0
"""Resolved runtime configs for weight-free stage tests."""

from collections.abc import Mapping
from typing import Any

from fastvideo.api.inference_resolution import fill_runtime_defaults
from fastvideo.api.resolution import ResolutionView, ResolvedGeneratorConfig, resolve_generator_config
from fastvideo.api.schema import GenericModelOptions


def fill_generic_model_block(view: ResolutionView) -> dict[str, Any]:
    """Set the ``generic`` block when ``raw`` writes no ``pipeline.model``; a written family block is kept as is."""
    return {} if view.get("pipeline.model") is not None else {"pipeline.model": GenericModelOptions()}


def make_resolved_config(pipeline_config: Any = None, raw: Mapping[str, Any] | None = None) -> ResolvedGeneratorConfig:
    """Resolve ``raw``, a nested ``GeneratorConfig`` mapping, with ``pipeline_config`` as the model definition.

    Only the ``pipeline.model`` fill and the runtime-defaults step run, so no checkpoint or registry lookup happens;
    the family of a written ``pipeline.model`` block is not checked against a model. ``pipeline_config`` stands in
    for the ``PipelineConfig`` that materialization builds; a test passes a small stand-in that holds only the model
    definition data that the stage under test reads.
    """
    return resolve_generator_config(
        {
            "model_path": "unused/for-stage-tests",
            **(raw or {})
        },
        (fill_generic_model_block, fill_runtime_defaults),
        materialize=lambda _: pipeline_config,
    )
