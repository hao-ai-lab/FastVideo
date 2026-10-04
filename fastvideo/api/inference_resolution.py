# SPDX-License-Identifier: Apache-2.0
"""Resolution of an inference ``GeneratorConfig`` into the frozen runtime config that the run starts with.

``resolve_inference_config`` runs the steps from ``inference_resolution_steps`` in order through
:func:`fastvideo.api.resolution.resolve_generator_config`, so every value that a step decides is recorded with the
step's name, and then materializes the model's ``PipelineConfig`` once and attaches it as ``pipeline_config``.

The steps run in this order. A fill step sets a field only while it is ``None``, so an earlier step takes
precedence over a later one: user input, then environment variables, then model defaults.

1. Environment variables fill typed fields.
2. Model defaults from the model's ``PipelineConfig`` fill the typed fields of ``PIPELINE_CONFIG_MIRRORS`` that are
   still unset.
3. ``pipeline.preset_overrides.refine`` sets the LTX-2 refine fields, and the checkpoint's bundled files fill the
   LTX-2 refine fields and the MiniMax-H3 DMD schedule that are still unset (``fastvideo.api.checkpoint_defaults``).
4. Derived values replace placeholders and load the files that a path names.
5. The device policy settles the offload settings for the memory class of the local device
   (``fastvideo.api.device_policy``).
6. Validation steps raise on inconsistent values and decide nothing.
7. ``fill_runtime_defaults`` gives every field that is still unset its runtime default.

``fastvideo.api.training_schema`` resolves the training and preprocessing roots with the same steps plus their own.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any

import fastvideo.envs as envs
from fastvideo.api.checkpoint_defaults import dmd_schedule_checkpoint_step, ltx2_refine_checkpoint_step
from fastvideo.api.device_policy import DEVICE_POLICY_STEPS
from fastvideo.api.parser import parse_config
from fastvideo.api.resolution import (ResolutionStep, ResolutionView, ResolvedGeneratorConfig, resolve_generator_config,
                                      thaw)
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


# Typed path -> the ``PipelineConfig`` attribute that mirrors it. The model's ``PipelineConfig`` subclasses declare
# their defaults for these settings as attributes: the defaults step fills an unset typed field from its attribute,
# and materialization writes the resolved typed value back to the attribute. Runtime code reads the typed path.
PIPELINE_CONFIG_MIRRORS: dict[str, str] = {
    "model_path": "model_path",
    "pipeline.components.pipeline_config_path": "pipeline_config_path",
    "engine.disable_autocast": "disable_autocast",
    "engine.precision.dit": "dit_precision",
    "engine.precision.vae": "vae_precision",
    "engine.precision.vae_decode": "vae_decode_precision",
    "engine.precision.image_encoder": "image_encoder_precision",
    "engine.precision.text_encoders": "text_encoder_precisions",
    "pipeline.flow_shift": "flow_shift",
    "pipeline.embedded_cfg_scale": "embedded_cfg_scale",
    "pipeline.dmd_denoising_steps": "dmd_denoising_steps",
    "pipeline.boundary_ratio": "boundary_ratio",
    "pipeline.vae_tiling": "vae_tiling",
    "pipeline.vae_sp": "vae_sp",
    "pipeline.longcat.enable_bsa": "enable_bsa",
    "pipeline.longcat.bsa_sparsity": "bsa_sparsity",
    "pipeline.longcat.bsa_cdf_threshold": "bsa_cdf_threshold",
    "pipeline.longcat.bsa_chunk_q": "bsa_chunk_q",
    "pipeline.longcat.bsa_chunk_k": "bsa_chunk_k",
}
# Mirrored paths that the defaults step leaves unset: an LTX-2 tile size can turn ``pipeline.vae_tiling`` on before
# :func:`vae_tiling_default_step` fills it from the model default.
_FILLED_AFTER_DERIVATION = frozenset({"pipeline.vae_tiling"})


def _pipeline_config_source(config: GeneratorConfig) -> Any:
    """The source that the model's ``PipelineConfig`` is built from: a JSON path, a mapping, or a ``PipelineConfig``.

    ``pipeline.experimental.pipeline_config`` wins over ``pipeline.components.pipeline_config_path``.
    """
    source = config.pipeline.experimental.get("pipeline_config")
    return config.pipeline.components.pipeline_config_path if source is None else source


def build_model_pipeline_config(config: GeneratorConfig) -> Any:
    """Build the model's ``PipelineConfig`` before any typed value is applied.

    It is the registry class for ``model_path``, updated from the source of :func:`_pipeline_config_source`. The
    defaults step reads it, and materialization then applies the resolved values to the same instance.
    """
    from fastvideo.configs.pipelines.base import PipelineConfig

    return PipelineConfig.from_source(config.model_path, deepcopy(_pipeline_config_source(config)))


def pipeline_config_defaults_step(config: GeneratorConfig, defaults: Any = None) -> ResolutionStep:
    """Build the step that fills unset typed fields with the values of the model's ``PipelineConfig``.

    ``defaults`` is the instance from :func:`build_model_pipeline_config`; it is built from ``config`` when it is
    not given. A typed field of ``PIPELINE_CONFIG_MIRRORS`` is filled when its value is ``None`` and its attribute on
    that ``PipelineConfig`` is not ``None``. The step's source name carries the ``PipelineConfig`` class name.
    """
    if defaults is None:
        defaults = build_model_pipeline_config(config)

    def fill_pipeline_config_defaults(view: ResolutionView) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for dotted_path, attribute in PIPELINE_CONFIG_MIRRORS.items():
            default = getattr(defaults, attribute, None)
            if dotted_path not in _FILLED_AFTER_DERIVATION and default is not None and view.get(dotted_path) is None:
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
    from fastvideo.pipelines.basic.ltx2.stage_overrides import (refine_preset_override_fields,
                                                                refine_stage_override_fields)

    refine = view.get_plain("pipeline.preset_overrides").get("refine")
    if not isinstance(refine, Mapping):
        return {}
    keys = refine_preset_override_fields() | refine_stage_override_fields()
    return {f"pipeline.ltx2.refine.{key}": refine[key] for key in sorted(keys) if refine.get(key) is not None}


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


# Typed fields that no runtime code reads. Resolution rejects a value at one of them, so the setting is not dropped.
UNSUPPORTED_PATHS = ("pipeline.preset", "pipeline.preset_version", "pipeline.components.vae_weights")


def validate_unsupported_fields(view: ResolutionView) -> dict[str, Any]:
    """The fields of ``UNSUPPORTED_PATHS`` must be unset."""
    unsupported = [path for path in UNSUPPORTED_PATHS if view.get(path) is not None]
    if unsupported:
        raise NotImplementedError(f"resolution does not support {', '.join(unsupported)}; leave these fields unset")
    return {}


def validate_experimental_keys(view: ResolutionView) -> dict[str, Any]:
    """A ``pipeline.experimental`` key must not name a ``PipelineConfig`` attribute that has a typed path.

    Materialization writes those attributes from their typed paths (``PIPELINE_CONFIG_MIRRORS``), so the key is set at
    the typed path instead.
    """
    typed_paths = {attribute: path for path, attribute in PIPELINE_CONFIG_MIRRORS.items()}
    misplaced = sorted(set(view.get("pipeline.experimental")) & set(typed_paths))
    if misplaced:
        moves = ", ".join(f"pipeline.experimental.{key} -> {typed_paths[key]}" for key in misplaced)
        raise ValueError(f"these settings have typed config paths; set them there: {moves}")
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


def apply_nvfp4_fa4_env(view: ResolutionView) -> dict[str, Any]:
    """``engine.attention.nvfp4_fa4`` exports the environment that the NVFP4 FlashAttention-4 path reads.

    Sets ``FASTVIDEO_NVFP4_FA4=1`` and the ``CUTE_DSL_ENABLE_TVM_FFI`` default of the CuTe DSL kernels when the field
    is true; decides nothing.
    """
    if view.get("engine.attention.nvfp4_fa4"):
        envs.FASTVIDEO_NVFP4_FA4.set(True)
        envs.setdefault_external("CUTE_DSL_ENABLE_TVM_FFI", "1")
    return {}


def torch_compile_kwargs(resolved_config: ResolvedGeneratorConfig) -> dict[str, Any]:
    """The keyword arguments for ``torch.compile`` that ``engine.compile`` describes.

    ``backend``, ``fullgraph``, ``mode``, and ``dynamic`` are included when they are set; ``extras`` is merged on top.
    """
    compile_config = resolved_config.engine.compile
    kwargs: dict[str, Any] = {}
    for key in ("backend", "fullgraph", "mode", "dynamic"):
        value = getattr(compile_config, key)
        if value is not None:
            kwargs[key] = value
    kwargs.update(thaw(compile_config.extras))
    return kwargs


# Values that runtime code uses for typed fields that no earlier step decided. The LTX-2 refine switches are the
# stage-2 defaults of the LTX-2 pipeline; the checkpoint's model_index.json fills them first when it declares them.
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
    "pipeline.ltx2.refine.enabled": False,
    "pipeline.ltx2.refine.add_noise": True,
    "pipeline.ltx2.refine.guidance_scale": 1.0,
    "pipeline.ltx2.refine.num_inference_steps": 3,
}


def fill_runtime_defaults(view: ResolutionView) -> dict[str, Any]:
    """Give each field in ``RUNTIME_DEFAULTS`` its runtime default while it is still unset; runs after every other step."""
    return {path: deepcopy(default) for path, default in RUNTIME_DEFAULTS.items() if view.get(path) is None}


ENVIRONMENT_STEPS: tuple[ResolutionStep, ...] = (
    fill_attention_backend_from_env,
    fill_regional_compile_from_env,
    fill_vae_parallel_from_env,
)
# Steps that check the values decided so far against each other.
VALIDATION_STEPS: tuple[ResolutionStep, ...] = (
    validate_unsupported_fields,
    validate_experimental_keys,
    validate_lora_strength,
    validate_attention_backend,
    validate_vae_parallel_decode_strategy,
    warn_deprecated_environment_variables,
    apply_nvfp4_fa4_env,
)


def generator_resolution_steps(
        config: GeneratorConfig,
        defaults: Any = None,
        *,
        before_placeholders: tuple[ResolutionStep, ...] = (),
        after: tuple[ResolutionStep, ...] = (),
) -> tuple[ResolutionStep, ...]:
    """The resolution steps of any generator root, in the order that they run.

    ``before_placeholders`` runs after the checkpoint fills and before ``derive_parallel_sizes``, for checks that need
    the -1 placeholders; ``after`` runs last. The device-policy steps run after every fill and derivation and before
    the validations. ``validate_parallel_sizes`` checks the sizes before ``derive_num_gpus_from_parallel_sizes`` raises
    ``num_gpus``.
    """
    if defaults is None:
        defaults = build_model_pipeline_config(config)
    return (
        *ENVIRONMENT_STEPS,
        pipeline_config_defaults_step(config, defaults),
        copy_refine_preset_overrides,
        ltx2_refine_checkpoint_step(defaults),
        dmd_schedule_checkpoint_step(defaults),
        *before_placeholders,
        derive_parallel_sizes,
        derive_vae_tiling_from_ltx2_tile_sizes,
        vae_tiling_default_step(defaults),
        load_moba_config,
        *DEVICE_POLICY_STEPS,
        *VALIDATION_STEPS,
        validate_parallel_sizes,
        derive_num_gpus_from_parallel_sizes,
        *after,
        fill_runtime_defaults,
    )


def inference_resolution_steps(config: GeneratorConfig, defaults: Any = None) -> tuple[ResolutionStep, ...]:
    """The resolution steps for an inference ``config``, in the order that they run."""
    return generator_resolution_steps(config, defaults)


# LTX-2 VAE tile size path -> the ``vae_config`` attribute that the LTX-2 VAE reads.
_LTX2_VAE_TILE_ATTRIBUTES = {
    "pipeline.ltx2.vae_spatial_tile_size_in_pixels": "ltx2_spatial_tile_size_in_pixels",
    "pipeline.ltx2.vae_spatial_tile_overlap_in_pixels": "ltx2_spatial_tile_overlap_in_pixels",
    "pipeline.ltx2.vae_temporal_tile_size_in_frames": "ltx2_temporal_tile_size_in_frames",
    "pipeline.ltx2.vae_temporal_tile_overlap_in_frames": "ltx2_temporal_tile_overlap_in_frames",
}


def _config_value(config: GeneratorConfig, path: str) -> Any:
    """Value at a dotted field path of a typed config; ``None`` through an optional section that is ``None``."""
    node: Any = config
    for part in path.split("."):
        if node is None:
            return None
        node = getattr(node, part)
    return node


def _set_present_attributes(target: Any, values: Mapping[str, Any]) -> None:
    """Set each value that is not ``None`` on ``target`` when ``target`` has an attribute of that name."""
    for name, value in values.items():
        if value is not None and hasattr(target, name):
            setattr(target, name, value)


def _apply_transformer_quant(pipeline_config: Any, transformer_quant: str | None) -> None:
    """Pin the quantization config that ``transformer_quant`` names on ``dit_config.quant_config`` unless one is set.

    A registry name such as ``nvfp4_qat_train`` becomes its ``QuantizationConfig`` instance.
    """
    dit_config = getattr(pipeline_config, "dit_config", None)
    if transformer_quant is None or dit_config is None:
        return
    from fastvideo.layers.quantization import get_quantization_config

    if getattr(dit_config, "quant_config", None) is None:
        dit_config.quant_config = get_quantization_config(transformer_quant)()


def materialize_pipeline_config(resolved: ResolvedGeneratorConfig, pipeline_config: Any) -> Any:
    """Apply the resolved values to the model's ``PipelineConfig``, validate it, and freeze it.

    ``pipeline_config`` is the instance from :func:`build_model_pipeline_config`. In order, materialization sets the
    ``pipeline.experimental`` keys that are ``PipelineConfig`` attributes (model-only fields), the attributes of
    ``PIPELINE_CONFIG_MIRRORS`` from their typed values, and the ``pipeline.vae`` and ``pipeline.dit`` entries on
    ``vae_config`` and ``dit_config``; a ``None`` value leaves its attribute unchanged. It then writes the LTX-2 VAE
    tile sizes onto ``vae_config``, pins the ``engine.quantization.transformer_quant`` config on ``dit_config``, runs
    ``check_pipeline_config``, and loads the VAE encoder for a preprocessing run.
    """
    config = resolved.to_config()
    _set_present_attributes(pipeline_config, {
        key: value
        for key, value in config.pipeline.experimental.items() if key != "pipeline_config"
    })
    mirrored = {attribute: _config_value(config, path) for path, attribute in PIPELINE_CONFIG_MIRRORS.items()}
    if mirrored["text_encoder_precisions"] is not None:
        mirrored["text_encoder_precisions"] = tuple(mirrored["text_encoder_precisions"])
    _set_present_attributes(pipeline_config, mirrored)
    _set_present_attributes(pipeline_config.vae_config, config.pipeline.vae)
    _set_present_attributes(pipeline_config.dit_config, config.pipeline.dit)
    _set_present_attributes(pipeline_config.vae_config, {
        attribute: _config_value(config, path)
        for path, attribute in _LTX2_VAE_TILE_ATTRIBUTES.items()
    })
    _apply_transformer_quant(pipeline_config, _config_value(config, "engine.quantization.transformer_quant"))
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
    "torch_compile_kwargs",
    "ENVIRONMENT_STEPS",
    "PIPELINE_CONFIG_MIRRORS",
    "VALIDATION_STEPS",
    "apply_nvfp4_fa4_env",
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
    "validate_attention_backend",
    "validate_experimental_keys",
    "validate_lora_strength",
    "validate_parallel_sizes",
    "validate_unsupported_fields",
    "validate_vae_parallel_decode_strategy",
    "warn_deprecated_environment_variables",
    "written_fields",
]
