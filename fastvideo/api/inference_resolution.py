# SPDX-License-Identifier: Apache-2.0
"""Resolution of an inference ``GeneratorConfig`` into the frozen runtime config that the run starts with.

``resolve_inference_config`` runs the steps from ``inference_resolution_steps`` in order through
:func:`fastvideo.api.resolution.resolve_generator_config`, so every value that a step decides is recorded with the
step's name, and then materializes the model's ``PipelineConfig`` once and attaches it as ``pipeline_config``.

The steps run in this order. A fill step sets a field only while it is ``None``, so an earlier step takes
precedence over a later one: user input, then environment variables, then model defaults.

1. Environment variables fill typed fields.
2. Model defaults from the model's ``PipelineConfig`` fill the typed fields that are still unset.
3. The flat keys under ``pipeline.preset_overrides`` and ``pipeline.experimental`` set the typed fields of their
   names; they win over the typed input, as they do when the ``PipelineConfig`` is built.
4. Derived values replace placeholders and load the files that a path names.
5. Validation steps raise on inconsistent values and decide nothing.

``fastvideo.api.training_schema`` resolves the training and preprocessing roots with the same steps plus their own.
``FastVideoArgs.__post_init__`` and ``check_fastvideo_args`` still apply these rules to a ``FastVideoArgs`` that is
built directly.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, get_args

import fastvideo.envs as envs
from fastvideo.api.parser import parse_config
from fastvideo.api.resolution import ResolutionStep, ResolutionView, ResolvedGeneratorConfig, resolve_generator_config
from fastvideo.api.schema import ExecutionMode, GeneratorConfig, WorkloadType
from fastvideo.logger import init_logger

logger = init_logger(__name__)


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
    """``FASTVIDEO_INFERENCE_TORCH_COMPILE`` turns ``engine.compile.regional`` on while the field is unset."""
    if view.get("engine.compile.regional") is not None or not envs.FASTVIDEO_INFERENCE_TORCH_COMPILE.get():
        return {}
    return {"engine.compile.regional": True}


def fill_vae_parallel_from_env(view: ResolutionView) -> dict[str, Any]:
    """The ``FASTVIDEO_VAE_PARALLEL_*`` variables set the MiniMax-H3 sequence-parallel VAE options.

    Each switch turns on while it is unset. The decode strategy fills while it is unset, and is ``gather`` when
    the variable is unset too.
    """
    values: dict[str, Any] = {}
    if view.get("pipeline.minimax_h3.vae_parallel_decode") is None and envs.FASTVIDEO_VAE_PARALLEL_DECODE.get():
        values["pipeline.minimax_h3.vae_parallel_decode"] = True
    if view.get("pipeline.minimax_h3.vae_parallel_encode") is None and envs.FASTVIDEO_VAE_PARALLEL_ENCODE.get():
        values["pipeline.minimax_h3.vae_parallel_encode"] = True
    if view.get("pipeline.minimax_h3.vae_parallel_decode_strategy") is None:
        strategy = envs.FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY.get() or "gather"
        values["pipeline.minimax_h3.vae_parallel_decode_strategy"] = strategy
    return values


def _pipeline_config_source(config: GeneratorConfig) -> Any:
    """The ``pipeline_config`` keyword that the flat keywords of ``config`` carry: a JSON path, a dict, or an object.

    ``pipeline.experimental`` wins over ``pipeline.preset_overrides``, which wins over
    ``pipeline.components.pipeline_config_path``.
    """
    source = config.pipeline.components.pipeline_config_path
    for overrides in (config.pipeline.preset_overrides, config.pipeline.experimental):
        if overrides.get("pipeline_config") is not None:
            source = overrides["pipeline_config"]
    return source


def build_model_pipeline_config(config: GeneratorConfig) -> Any:
    """Build the model's ``PipelineConfig`` before any typed value is applied.

    It is the registry class for ``model_path``, updated from the ``pipeline_config`` source of
    :func:`_pipeline_config_source`. The defaults step reads it, and materialization then applies the resolved values
    to the same instance.
    """
    from fastvideo.configs.pipelines.base import PipelineConfig

    source = _pipeline_config_source(config)
    kwargs: dict[str, Any] = {"model_path": config.model_path}
    if source is not None:
        kwargs["pipeline_config"] = deepcopy(source)
    return PipelineConfig.from_kwargs(kwargs)


def pipeline_config_defaults_step(config: GeneratorConfig, defaults: Any = None) -> ResolutionStep:
    """Build the step that fills unset typed fields with the values of the model's ``PipelineConfig``.

    ``defaults`` is the instance from :func:`build_model_pipeline_config`; it is built from ``config`` when it is
    not given. A typed field is filled when its flat name is an attribute of that ``PipelineConfig``, its value is
    ``None``, and the attribute is not ``None``. The step's source name carries the ``PipelineConfig`` class name.
    """
    from fastvideo.api.compat import _FLAT_NAME_FIELDS

    if defaults is None:
        defaults = build_model_pipeline_config(config)

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


_LTX2_VAE_TILE_FIELDS = (
    "pipeline.ltx2.vae_spatial_tile_size_in_pixels",
    "pipeline.ltx2.vae_spatial_tile_overlap_in_pixels",
    "pipeline.ltx2.vae_temporal_tile_size_in_frames",
    "pipeline.ltx2.vae_temporal_tile_overlap_in_frames",
)


def derive_vae_tiling_from_ltx2_tile_sizes(view: ResolutionView) -> dict[str, Any]:
    """An LTX-2 VAE tile size turns ``pipeline.vae_tiling`` on while the field is unset."""
    if view.get("pipeline.vae_tiling") is not None or all(view.get(path) is None for path in _LTX2_VAE_TILE_FIELDS):
        return {}
    return {"pipeline.vae_tiling": True}


def vae_tiling_default_step(defaults: Any) -> ResolutionStep:
    """Build the step that fills ``pipeline.vae_tiling`` from the model's ``PipelineConfig.vae_tiling`` while unset.

    It runs after :func:`derive_vae_tiling_from_ltx2_tile_sizes`, so an LTX-2 tile size still turns tiling on.
    """

    def fill_vae_tiling_default(view: ResolutionView) -> dict[str, Any]:
        default = getattr(defaults, "vae_tiling", None)
        if default is None or view.get("pipeline.vae_tiling") is not None:
            return {}
        return {"pipeline.vae_tiling": default}

    fill_vae_tiling_default.__qualname__ = f"fill_vae_tiling_default[{type(defaults).__name__}]"
    return fill_vae_tiling_default


def copy_refine_preset_overrides(view: ResolutionView) -> dict[str, Any]:
    """``pipeline.preset_overrides.refine`` sets the LTX-2 refine fields of the same names.

    The keys are the refine preset and stage override fields (``enabled``, ``add_noise``, ``num_inference_steps``,
    ``guidance_scale``, ``image_crf``, ``video_position_offset_sec``); a ``None`` value leaves its field unchanged.
    """
    from fastvideo.api.compat import _LTX2_REFINE_FLAT_KEYS

    refine = view.get_plain("pipeline.preset_overrides").get("refine")
    if not isinstance(refine, Mapping):
        return {}
    return {
        f"pipeline.ltx2.refine.{key}": refine[key]
        for key in sorted(_LTX2_REFINE_FLAT_KEYS) if refine.get(key) is not None
    }


def _flat_value_for_field(annotation: Any, value: Any) -> Any:
    """A flat keyword value in the type of its typed field: a string becomes the member of an enum field."""
    enum_types = [arg for arg in (annotation, *get_args(annotation)) if isinstance(arg, type) and issubclass(arg, Enum)]
    if enum_types and isinstance(value, str) and not isinstance(value, enum_types[0]):
        return enum_types[0](value)
    return deepcopy(value)


def route_flat_override_keys(view: ResolutionView) -> dict[str, Any]:
    """The flat keys under ``pipeline.preset_overrides`` and then ``pipeline.experimental`` set their typed fields.

    A key that is the flat name of a typed field sets that field; ``dit_config.<key>`` and ``vae_config.<key>`` set
    ``pipeline.dit`` and ``pipeline.vae`` entries; a string ``pipeline_config`` sets
    ``pipeline.components.pipeline_config_path``. A generic ``refine_*`` key then sets its LTX-2 refine field, so it
    wins over the ``ltx2_refine_*`` key. ``None`` values and keys without a typed field change nothing.
    """
    from fastvideo.api.compat import _COMPONENT_OVERRIDE_PREFIXES, _FLAT_NAME_FIELDS
    from fastvideo.api.flat_name_fallback import GENERIC_REFINE_PATHS, generic_refine_aliases

    preset_overrides = view.get_plain("pipeline.preset_overrides")
    experimental = view.get_plain("pipeline.experimental")
    flat_inputs = {key: value for key, value in preset_overrides.items() if key != "refine"}
    flat_inputs.update(experimental)
    values: dict[str, Any] = {}
    for key, value in flat_inputs.items():
        if value is None:
            continue
        prefixes = [prefix for prefix in _COMPONENT_OVERRIDE_PREFIXES if key.startswith(prefix)]
        if prefixes:
            values[f"pipeline.{_COMPONENT_OVERRIDE_PREFIXES[prefixes[0]]}.{key[len(prefixes[0]):]}"] = value
        elif key == "pipeline_config":
            if isinstance(value, str):
                values["pipeline.components.pipeline_config_path"] = value
        elif key in _FLAT_NAME_FIELDS:
            dotted_path, annotation = _FLAT_NAME_FIELDS[key]
            values[dotted_path] = _flat_value_for_field(annotation, value)
    for name, value in generic_refine_aliases(preset_overrides, experimental).items():
        if value is not None:
            values[GENERIC_REFINE_PATHS[name]] = value
    return values


def load_moba_config(view: ResolutionView) -> dict[str, Any]:
    """``engine.attention.moba_config_path`` names a JSON file whose contents become ``engine.attention.moba_config``.

    A missing or malformed file raises.
    """
    path = view.get("engine.attention.moba_config_path")
    if not path:
        return {}
    try:
        with open(path) as handle:
            moba_config = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logger.error("Failed to load V-MoBA config from %s: %s", path, error)
        raise
    logger.info("Loaded V-MoBA config from %s", path)
    return {"engine.attention.moba_config": moba_config}


def derive_num_gpus_from_parallel_sizes(view: ResolutionView) -> dict[str, Any]:
    """``engine.num_gpus`` rises to the larger of ``tp_size`` and ``sp_size`` when it is smaller."""
    num_gpus = view.get("engine.num_gpus")
    required = max(view.get("engine.parallelism.tp_size"), view.get("engine.parallelism.sp_size"))
    return {"engine.num_gpus": required} if num_gpus < required else {}


def validate_lora_strength(view: ResolutionView) -> dict[str, Any]:
    """``pipeline.components.lora_strength`` must be a finite number."""
    lora_strength = view.get("pipeline.components.lora_strength")
    if not math.isfinite(lora_strength):
        raise ValueError(f"lora_strength must be finite, got {lora_strength}")
    return {}


def validate_attention_backend(view: ResolutionView) -> dict[str, Any]:
    """A set ``engine.attention.backend`` must name a supported backend, so a typo fails before any model loads."""
    backend = view.get("engine.attention.backend")
    if backend is not None:
        from fastvideo.attention.selector import coerce_attn_backend

        coerce_attn_backend(backend)
    return {}


# Mirrors fastvideo.models.vaes.minimax_h3_parallel.DECODE_GATHER_STRATEGIES, so resolution imports no model
# module; fastvideo/tests/api/test_resolved_runtime_config.py pins the two in sync.
VAE_PARALLEL_DECODE_STRATEGIES = ("gather", "all_gather")


def validate_vae_parallel_decode_strategy(view: ResolutionView) -> dict[str, Any]:
    """``pipeline.minimax_h3.vae_parallel_decode_strategy`` must be ``gather`` or ``all_gather``, from any source."""
    strategy = view.get("pipeline.minimax_h3.vae_parallel_decode_strategy")
    if strategy not in VAE_PARALLEL_DECODE_STRATEGIES:
        raise ValueError(f"vae_parallel_decode_strategy must be one of {VAE_PARALLEL_DECODE_STRATEGIES}, "
                         f"got {strategy!r}.")
    return {}


def validate_parallel_sizes(view: ResolutionView) -> dict[str, Any]:
    """``engine.num_gpus`` must be at least, and divisible by, ``sp_size``, ``hsdp_replicate_dim``, and
    ``hsdp_shard_dim``."""
    num_gpus = view.get("engine.num_gpus")
    for name in ("sp_size", "hsdp_replicate_dim", "hsdp_shard_dim"):
        size = view.get(f"engine.parallelism.{name}")
        if not (size <= num_gpus and num_gpus % size == 0):
            raise ValueError(f"num_gpus must >= and be divisible by {name}")
    return {}


def warn_deprecated_environment_variables(view: ResolutionView) -> dict[str, Any]:
    """Log a warning for each deprecated FastVideo environment variable that is set."""
    envs.warn_deprecated_variables()
    return {}


# Values that runtime code uses for typed fields that no earlier step decided. ``pipeline.ltx2.refine.*`` and
# ``engine.offload.lazy_module_load`` are not listed: ``None`` there means "decide later" (the LTX-2 checkpoint defaults
# and the device policy).
RUNTIME_DEFAULTS: dict[str, Any] = {
    "engine.attention.vsa_sparsity": 0.0,
    "engine.attention.vsa_tile_size": 256,
    "engine.attention.moba_config": {},
    "engine.compile.text_encoder_enabled": False,
    "engine.compile.vae_enabled": False,
    "engine.compile.audio_vae_enabled": False,
    "engine.compile.regional": False,
    "pipeline.workload_type": WorkloadType.T2V,
    "pipeline.minimax_h3.taeh3_chunk_size": 5,
    "pipeline.minimax_h3.vae_parallel_decode": False,
    "pipeline.minimax_h3.vae_parallel_encode": False,
    "pipeline.minimax_h3.video_decode_backend": "h3-vae",
    "pipeline.ltx2.legacy_native_noise_order": False,
    "pipeline.ltx2.use_distilled_sigmas": True,
}


def fill_runtime_defaults(view: ResolutionView) -> dict[str, Any]:
    """Give each field in ``RUNTIME_DEFAULTS`` its runtime default while it is still unset; runs after every other step."""
    return {path: deepcopy(default) for path, default in RUNTIME_DEFAULTS.items() if view.get(path) is None}


ENVIRONMENT_STEPS: tuple[ResolutionStep, ...] = (
    fill_attention_backend_from_env,
    fill_regional_compile_from_env,
    fill_vae_parallel_from_env,
)
# Steps between the model defaults and the derived values: inputs written under flat keyword names.
FLAT_INPUT_STEPS: tuple[ResolutionStep, ...] = (copy_refine_preset_overrides, route_flat_override_keys)
# Steps that check the values decided so far against each other.
VALIDATION_STEPS: tuple[ResolutionStep, ...] = (
    validate_lora_strength,
    validate_attention_backend,
    validate_vae_parallel_decode_strategy,
    warn_deprecated_environment_variables,
)


def generator_resolution_steps(
        config: GeneratorConfig,
        defaults: Any = None,
        *,
        before_placeholders: tuple[ResolutionStep, ...] = (),
        after: tuple[ResolutionStep, ...] = (),
) -> tuple[ResolutionStep, ...]:
    """The resolution steps of any generator root, in the order that they run.

    ``before_placeholders`` runs after the flat inputs and before ``derive_parallel_sizes``, for checks that need the
    -1 placeholders; ``after`` runs last. ``validate_parallel_sizes`` checks the sizes before
    ``derive_num_gpus_from_parallel_sizes`` raises ``num_gpus``.
    """
    if defaults is None:
        defaults = build_model_pipeline_config(config)
    return (
        *ENVIRONMENT_STEPS,
        pipeline_config_defaults_step(config, defaults),
        *FLAT_INPUT_STEPS,
        *before_placeholders,
        derive_parallel_sizes,
        derive_vae_tiling_from_ltx2_tile_sizes,
        vae_tiling_default_step(defaults),
        load_moba_config,
        *VALIDATION_STEPS,
        validate_parallel_sizes,
        derive_num_gpus_from_parallel_sizes,
        *after,
        fill_runtime_defaults,
    )


def inference_resolution_steps(config: GeneratorConfig, defaults: Any = None) -> tuple[ResolutionStep, ...]:
    """The resolution steps for an inference ``config``, in the order that they run."""
    return generator_resolution_steps(config, defaults)


def _apply_ltx2_vae_overrides(pipeline_config: Any, flat: Mapping[str, Any]) -> None:
    """Apply the LTX-2 VAE tiling keywords to ``pipeline_config``, as ``FastVideoArgs`` did after construction.

    ``ltx2_vae_tiling`` sets ``vae_tiling``; otherwise any tile size turns it on. Each tile size is written onto
    ``vae_config`` when the VAE config has that attribute.
    """
    tile_sizes = {
        "ltx2_spatial_tile_size_in_pixels": flat.get("ltx2_vae_spatial_tile_size_in_pixels"),
        "ltx2_spatial_tile_overlap_in_pixels": flat.get("ltx2_vae_spatial_tile_overlap_in_pixels"),
        "ltx2_temporal_tile_size_in_frames": flat.get("ltx2_vae_temporal_tile_size_in_frames"),
        "ltx2_temporal_tile_overlap_in_frames": flat.get("ltx2_vae_temporal_tile_overlap_in_frames"),
    }
    if flat.get("ltx2_vae_tiling") is not None and hasattr(pipeline_config, "vae_tiling"):
        pipeline_config.vae_tiling = flat["ltx2_vae_tiling"]
    elif any(value is not None for value in tile_sizes.values()) and hasattr(pipeline_config, "vae_tiling"):
        pipeline_config.vae_tiling = True
    vae_config = pipeline_config.vae_config
    for attribute, value in tile_sizes.items():
        if value is not None and hasattr(vae_config, attribute):
            setattr(vae_config, attribute, value)


def _apply_transformer_quant(pipeline_config: Any, transformer_quant: Any) -> None:
    """Pin a transformer quantization config on ``dit_config.quant_config`` unless one is already set there.

    A registry name such as ``nvfp4_qat_train`` becomes its ``QuantizationConfig`` instance first.
    """
    dit_config = getattr(pipeline_config, "dit_config", None)
    if transformer_quant is None or dit_config is None:
        return
    if isinstance(transformer_quant, str):
        from fastvideo.layers.quantization import get_quantization_config
        transformer_quant = get_quantization_config(transformer_quant)()
    if getattr(dit_config, "quant_config", None) is None:
        dit_config.quant_config = transformer_quant


def materialize_pipeline_config(resolved: ResolvedGeneratorConfig, pipeline_config: Any) -> Any:
    """Apply the resolved values to the model's ``PipelineConfig``, validate it, and freeze it.

    ``pipeline_config`` is the instance from :func:`build_model_pipeline_config`. The flat keywords of
    :func:`fastvideo.api.compat.generator_kwargs` update it with the rules of ``PipelineConfig.update_config_from_dict``
    (``dit_config.<key>`` and ``vae_config.<key>`` included). The keywords that it leaves are the ones that apply the
    LTX-2 VAE tiling and the transformer quantization. A preprocessing run loads the VAE encoder.
    """
    from fastvideo.api.compat import generator_kwargs

    flat = generator_kwargs(resolved)
    source = flat.get("pipeline_config")
    if isinstance(source, str):
        flat["pipeline_config_path"] = source
    flat["model_path"] = resolved.model_path
    pipeline_config.update_config_from_dict(flat)
    _apply_ltx2_vae_overrides(pipeline_config, flat)
    _apply_transformer_quant(pipeline_config, flat.get("transformer_quant"))
    pipeline_config.check_pipeline_config()
    if resolved.mode == ExecutionMode.PREPROCESS and not pipeline_config.vae_config.load_encoder:
        pipeline_config.vae_config.load_encoder = True
    pipeline_config.freeze()
    return pipeline_config


def resolve_config(
    config: GeneratorConfig | Mapping[str, Any],
    config_class: type[GeneratorConfig],
    build_steps: Callable[[GeneratorConfig, Any], tuple[ResolutionStep, ...]],
) -> ResolvedGeneratorConfig:
    """Resolve a generator root, materialize its ``PipelineConfig``, and freeze the result.

    A mapping is the raw nested input, and every leaf that it contains counts as written by the user. A config
    object does not record which fields were written, so its fields that differ from the schema defaults count as
    written. ``build_steps(typed_config, model_pipeline_config)`` returns the step list. The model's
    ``PipelineConfig`` is built once: the defaults step reads it and materialization finishes it.
    """
    raw = config if isinstance(config, Mapping) else written_fields(config)
    typed_config = config if isinstance(config, config_class) else parse_config(config_class, config)
    model_pipeline_config = build_model_pipeline_config(typed_config)
    return resolve_generator_config(
        raw,
        build_steps(typed_config, model_pipeline_config),
        config_class=config_class,
        materialize=lambda resolved: materialize_pipeline_config(resolved, model_pipeline_config),
    )


def resolve_inference_config(config: GeneratorConfig | Mapping[str, Any]) -> ResolvedGeneratorConfig:
    """Resolve an inference config and freeze the result, with its ``PipelineConfig`` as ``pipeline_config``.

    A mapping is the raw nested input, and every leaf that it contains counts as written by the user. A
    ``GeneratorConfig`` object does not record which fields were written, so its fields that differ from the
    schema defaults count as written.
    """
    return resolve_config(config, GeneratorConfig, inference_resolution_steps)


def written_fields(config: GeneratorConfig) -> dict[str, Any]:
    """Raw nested mapping of the fields of ``config`` that differ from the schema defaults, plus ``model_path``.

    The defaults are those of the class of ``config``. Parsing the mapping gives back a config equal to ``config``.
    """
    return {"model_path": config.model_path, **_non_default_fields(config, type(config)(model_path=config.model_path))}


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
    "FLAT_INPUT_STEPS",
    "VALIDATION_STEPS",
    "build_model_pipeline_config",
    "copy_refine_preset_overrides",
    "derive_num_gpus_from_parallel_sizes",
    "derive_parallel_sizes",
    "derive_vae_tiling_from_ltx2_tile_sizes",
    "fill_attention_backend_from_env",
    "fill_regional_compile_from_env",
    "fill_vae_parallel_from_env",
    "generator_resolution_steps",
    "inference_resolution_steps",
    "load_moba_config",
    "materialize_pipeline_config",
    "pipeline_config_defaults_step",
    "resolve_config",
    "resolve_inference_config",
    "route_flat_override_keys",
    "validate_attention_backend",
    "validate_lora_strength",
    "validate_parallel_sizes",
    "validate_vae_parallel_decode_strategy",
    "warn_deprecated_environment_variables",
    "written_fields",
]
