# SPDX-License-Identifier: Apache-2.0
"""Resolution of an inference ``GeneratorConfig`` into the frozen values that the run starts with.

``resolve_inference_config`` runs the steps from ``inference_resolution_steps`` in order through
:func:`fastvideo.api.resolution.resolve_generator_config`, so every value that a step decides is recorded with the
step's name. ``generator_config_to_fastvideo_args`` flattens the resolved values into ``FastVideoArgs``.

The steps run in this order. A fill step sets a field only while it is ``None``, so an earlier step takes
precedence over a later one: user input, then environment variables, then model defaults.

1. Environment variables fill typed fields.
2. Model defaults from the model's ``PipelineConfig`` fill the typed fields that are still unset.
3. Derived values replace placeholders.

``FastVideoArgs.__post_init__`` and ``check_fastvideo_args`` still apply the same rules to a ``FastVideoArgs`` that
is built directly; for a resolved config they find the values already decided and change nothing.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any

import fastvideo.envs as envs
from fastvideo.api.parser import parse_config
from fastvideo.api.resolution import ResolutionStep, ResolutionView, ResolvedGeneratorConfig, resolve_generator_config
from fastvideo.api.schema import GeneratorConfig


def fill_attention_backend_from_env(view: ResolutionView) -> dict[str, Any]:
    """``FASTVIDEO_ATTENTION_BACKEND`` sets ``engine.attention.backend`` while the field is unset.

    An unsupported backend name raises ``ValueError``.
    """
    from fastvideo.attention.selector import get_env_variable_attn_backend

    if view.get("engine.attention.backend") is not None:
        return {}
    backend = get_env_variable_attn_backend()
    return {} if backend is None else {"engine.attention.backend": backend.name}


def fill_regional_compile_from_env(view: ResolutionView) -> dict[str, Any]:
    """``FASTVIDEO_INFERENCE_TORCH_COMPILE`` turns ``engine.compile.regional`` on unless it is already on."""
    if view.get("engine.compile.regional") or not envs.FASTVIDEO_INFERENCE_TORCH_COMPILE.get():
        return {}
    return {"engine.compile.regional": True}


def fill_vae_parallel_from_env(view: ResolutionView) -> dict[str, Any]:
    """The ``FASTVIDEO_VAE_PARALLEL_*`` variables set the MiniMax-H3 sequence-parallel VAE options.

    Each switch turns on unless it is already on. The decode strategy fills while it is unset, and is ``gather``
    when the variable is unset too.
    """
    values: dict[str, Any] = {}
    if not view.get("pipeline.minimax_h3.vae_parallel_decode") and envs.FASTVIDEO_VAE_PARALLEL_DECODE.get():
        values["pipeline.minimax_h3.vae_parallel_decode"] = True
    if not view.get("pipeline.minimax_h3.vae_parallel_encode") and envs.FASTVIDEO_VAE_PARALLEL_ENCODE.get():
        values["pipeline.minimax_h3.vae_parallel_encode"] = True
    if view.get("pipeline.minimax_h3.vae_parallel_decode_strategy") is None:
        strategy = envs.FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY.get() or "gather"
        values["pipeline.minimax_h3.vae_parallel_decode_strategy"] = strategy
    return values


def pipeline_config_defaults_step(config: GeneratorConfig) -> ResolutionStep:
    """Build the step that fills unset typed fields with the values of the model's ``PipelineConfig``.

    The ``PipelineConfig`` is the one that flattening starts from: the registry class for ``model_path``, updated
    by a ``pipeline_config`` entry in ``pipeline.experimental`` or else by ``pipeline.components.pipeline_config_path``.
    A typed field is filled when its flat name is an attribute of that ``PipelineConfig``, its value is ``None``,
    and the attribute is not ``None``. The step's source name carries the ``PipelineConfig`` class name.
    """
    from fastvideo.api.compat import _FLAT_NAME_FIELDS
    from fastvideo.configs.pipelines.base import PipelineConfig

    pipeline_config_source = config.pipeline.experimental.get("pipeline_config",
                                                              config.pipeline.components.pipeline_config_path)
    kwargs: dict[str, Any] = {"model_path": config.model_path}
    if pipeline_config_source is not None:
        kwargs["pipeline_config"] = deepcopy(pipeline_config_source)
    defaults = PipelineConfig.from_kwargs(kwargs)

    def fill_pipeline_config_defaults(view: ResolutionView) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for flat_name, (dotted_path, _) in _FLAT_NAME_FIELDS.items():
            default = getattr(defaults, flat_name, None)
            if default is not None and view.get(dotted_path) is None:
                values[dotted_path] = deepcopy(default)
        return values

    fill_pipeline_config_defaults.__qualname__ = f"fill_pipeline_config_defaults[{type(defaults).__name__}]"
    return fill_pipeline_config_defaults


def derive_parallel_sizes(view: ResolutionView) -> dict[str, Any]:
    """Replace the -1 placeholders: ``tp_size`` becomes 1, and ``sp_size`` and ``hsdp_shard_dim`` become ``num_gpus``."""
    num_gpus = view.get("engine.num_gpus")
    placeholders = {
        "engine.parallelism.tp_size": 1,
        "engine.parallelism.sp_size": num_gpus,
        "engine.parallelism.hsdp_shard_dim": num_gpus,
    }
    return {path: value for path, value in placeholders.items() if view.get(path) == -1}


ENVIRONMENT_STEPS: tuple[ResolutionStep, ...] = (
    fill_attention_backend_from_env,
    fill_regional_compile_from_env,
    fill_vae_parallel_from_env,
)


def inference_resolution_steps(config: GeneratorConfig) -> tuple[ResolutionStep, ...]:
    """The resolution steps for ``config``, in the order that they run."""
    return (*ENVIRONMENT_STEPS, pipeline_config_defaults_step(config), derive_parallel_sizes)


def resolve_inference_config(config: GeneratorConfig | Mapping[str, Any]) -> ResolvedGeneratorConfig:
    """Resolve an inference config and freeze the result.

    A mapping is the raw nested input, and every leaf that it contains counts as written by the user. A
    ``GeneratorConfig`` object does not record which fields were written, so its fields that differ from the
    schema defaults count as written.
    """
    raw = config if isinstance(config, Mapping) else written_fields(config)
    typed_config = config if isinstance(config, GeneratorConfig) else parse_config(GeneratorConfig, config)
    return resolve_generator_config(raw, inference_resolution_steps(typed_config))


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
    "ENVIRONMENT_STEPS",
    "derive_parallel_sizes",
    "fill_attention_backend_from_env",
    "fill_regional_compile_from_env",
    "fill_vae_parallel_from_env",
    "inference_resolution_steps",
    "pipeline_config_defaults_step",
    "resolve_inference_config",
    "written_fields",
]
