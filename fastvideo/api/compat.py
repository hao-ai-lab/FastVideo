# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.api.overrides import apply_overrides, normalize_overrides
from fastvideo.api.parser import config_to_dict, load_raw_config, parse_config
from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.request_metadata import (
    EXPLICIT_PATHS_ATTR,
    bind_generation_request_raw,
    get_explicit_paths,
    reset_tracking_roots,
)
from fastvideo.api.schema import (
    FLAT_NAME,
    CompileConfig,
    ContinuationState,
    GenerationRequest,
    GeneratorConfig,
)
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.ltx2.stage_overrides import (
    refine_preset_override_fields,
    refine_stage_override_fields,
)

_MISSING = object()
REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS = (
    "ltx2_audio_latents",
    "ltx2_audio_clean_latent",
    "ltx2_audio_denoise_mask",
    "audio_num_frames",
    "video_position_offset_sec",
    "vsa_mode",
    "vsa_dense_first_n_steps",
    "vsa_dense_layers",
)
# The VideoGenerator.from_pretrained keywords besides model_path, config, and log_queue. Every other setting goes
# through VideoGenerator.from_config at its typed path.
FROM_PRETRAINED_KWARGS = frozenset({
    "num_gpus",
    "revision",
    "trust_remote_code",
    "distributed_executor_backend",
    "tp_size",
    "sp_size",
    "hsdp_replicate_dim",
    "hsdp_shard_dim",
    "dist_timeout",
    "use_fsdp_inference",
    "disable_autocast",
    "enable_stage_verification",
    "dit_cpu_offload",
    "dit_layerwise_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
    "pin_cpu_memory",
    "enable_torch_compile",
    "torch_compile_kwargs",
    "lora_path",
    "lora_strength",
    "output_type",
    "nvfp4_fa4",
})
# torch.compile kwargs that map to first-class CompileConfig fields.
_COMPILE_TYPED_KEYS = ("backend", "fullgraph", "mode", "dynamic")
# LTX-2 refine flat kwargs (init + per-request) known to FastVideoArgs.
_LTX2_REFINE_FLAT_KEYS = (refine_preset_override_fields() | refine_stage_override_fields())


def normalize_generator_config(config: GeneratorConfig | Mapping[str, Any], ) -> GeneratorConfig:
    if isinstance(config, GeneratorConfig):
        return config
    return parse_config(GeneratorConfig, config)


def load_generator_config_from_file(
    path: str | Path,
    overrides: list[str] | Mapping[str, Any] | None = None,
) -> GeneratorConfig:
    raw = load_raw_config(path)
    normalized_overrides = normalize_overrides(overrides)

    if _looks_like_run_or_serve_config(raw):
        if normalized_overrides:
            raw = apply_overrides(raw, normalized_overrides)
        return parse_config(GeneratorConfig, raw["generator"])

    if normalized_overrides:
        adjusted = normalized_overrides
        if all(key.startswith("generator.") for key in adjusted):
            adjusted = {key[len("generator."):]: value for key, value in adjusted.items()}
        raw = apply_overrides(raw, adjusted)

    return parse_config(GeneratorConfig, raw)


def from_pretrained_kwargs_to_config(
    model_path: str,
    kwargs: Mapping[str, Any],
) -> GeneratorConfig:
    """Build a ``GeneratorConfig`` from ``VideoGenerator.from_pretrained`` keyword arguments.

    A keyword that a schema field declares as its flat name sets that field. ``torch_compile_kwargs`` is split across
    ``engine.compile``, and the keywords without a typed field are kept in ``pipeline.experimental``. A keyword outside
    ``FROM_PRETRAINED_KWARGS`` raises ``TypeError`` that names the typed path to use with ``from_config``.
    """
    unsupported = sorted(set(kwargs) - FROM_PRETRAINED_KWARGS)
    if unsupported:
        paths = ", ".join(f"{key} -> {_typed_path_of_keyword(key, kwargs[key])}" for key in unsupported)
        raise TypeError("VideoGenerator.from_pretrained(...) does not accept these keywords; pass them to "
                        f"VideoGenerator.from_config(...) at these config paths: {paths}")

    raw: dict[str, Any] = {"model_path": model_path}
    for key, value in kwargs.items():
        if key == "torch_compile_kwargs":
            remaining: dict[str, Any] = (dict(deepcopy(value)) if isinstance(value, Mapping) else {})
            for first_class in _COMPILE_TYPED_KEYS:
                if first_class in remaining:
                    _set_dotted_path(raw, ["engine", "compile", first_class], remaining.pop(first_class))
            if remaining:
                _set_dotted_path(raw, ["engine", "compile", "extras"], remaining)
        elif key in _FLAT_NAME_FIELDS:
            _set_dotted_path(raw, _FLAT_NAME_FIELDS[key][0].split("."), value)
        else:
            _set_dotted_path(raw, ["pipeline", "experimental", key], deepcopy(value))
    return parse_config(GeneratorConfig, raw)


def _typed_path_of_keyword(key: str, value: Any) -> str:
    """The ``GeneratorConfig`` path that holds the setting of a flat ``FastVideoArgs`` keyword."""
    if key in _FLAT_NAME_FIELDS and not (key == "pipeline_config" and not isinstance(value, str)):
        return _FLAT_NAME_FIELDS[key][0]
    for prefix, section in _COMPONENT_OVERRIDE_PREFIXES.items():
        if key.startswith(prefix):
            return f"pipeline.{section}.{key[len(prefix):]}"
    return f"pipeline.experimental.{key}"


def generator_config_to_fastvideo_args(
    config: GeneratorConfig | Mapping[str, Any] | ResolvedGeneratorConfig, ) -> FastVideoArgs:
    """Resolve a ``GeneratorConfig`` and flatten it into a ``FastVideoArgs``.

    A config that is not resolved yet goes through :func:`resolve_inference_config` first. The keywords are the ones
    that :func:`generator_kwargs` builds.
    """
    resolved = config if isinstance(config, ResolvedGeneratorConfig) else resolve_inference_config(config)
    return FastVideoArgs.from_kwargs(**generator_kwargs(resolved), resolved_config=resolved)


def generator_kwargs(resolved: ResolvedGeneratorConfig) -> dict[str, Any]:
    """Flatten a resolved config into the flat keywords that build its ``PipelineConfig``.

    Every field of the resolved config that declares a flat name and holds a value other than ``None`` becomes the
    keyword of that name. The paths in ``_SPECIALLY_MAPPED_PATHS`` are converted below; ``pipeline.preset_overrides``
    and then ``pipeline.experimental`` are applied last, so their keys win over the typed fields.
    """
    normalized = resolved.to_config()
    unsupported = []
    if normalized.pipeline.preset is not None:
        unsupported.append("pipeline.preset")
    if normalized.pipeline.preset_version is not None:
        unsupported.append("pipeline.preset_version")
    if normalized.pipeline.components.vae_weights is not None:
        unsupported.append("pipeline.components.vae_weights")
    if unsupported:
        joined = ", ".join(unsupported)
        raise NotImplementedError(f"VideoGenerator compatibility adapter does not support {joined} yet")

    kwargs: dict[str, Any] = {}
    for flat_name, (dotted_path, _) in _FLAT_NAME_FIELDS.items():
        value = _read_dotted_path(normalized, dotted_path.split("."))
        if value is not _MISSING and value is not None:
            kwargs[flat_name] = deepcopy(value)
    kwargs["torch_compile_kwargs"] = _compile_config_to_torch_kwargs(normalized.engine.compile)

    quantization = normalized.engine.quantization
    if quantization is not None and quantization.transformer_quant is not None:
        # Resolve the typed quant name to a concrete ``QuantizationConfig``
        # instance and pin it on ``dit_config.quant_config``. The legacy
        # path expected callers to do this themselves via
        # ``pipeline_config.dit_config.quant_config = NVFP4Config()``; the
        # typed surface accepts a string and does the wiring here so
        # downstream code can rely on a single source of truth.
        from fastvideo.layers.quantization import get_quantization_config
        _resolved_quant_cls = get_quantization_config(quantization.transformer_quant)
        kwargs["transformer_quant"] = _resolved_quant_cls()

    for prefix, section in _COMPONENT_OVERRIDE_PREFIXES.items():
        for field_name, value in getattr(normalized.pipeline, section).items():
            kwargs[f"{prefix}{field_name}"] = deepcopy(value)

    preset_overrides = deepcopy(normalized.pipeline.preset_overrides)
    refine = preset_overrides.pop("refine", None)
    if isinstance(refine, Mapping):
        for key in _LTX2_REFINE_FLAT_KEYS:
            if key in refine:
                kwargs[f"ltx2_refine_{key}"] = refine[key]
        if "enabled" in refine:
            kwargs["refine_enabled"] = refine["enabled"]
    kwargs.update(preset_overrides)
    kwargs.update(deepcopy(normalized.pipeline.experimental))
    return kwargs


def normalize_generation_request(request: GenerationRequest | Mapping[str, Any], ) -> GenerationRequest:
    normalized = (request if isinstance(request, GenerationRequest) else parse_config(GenerationRequest, request))

    if not hasattr(normalized, EXPLICIT_PATHS_ATTR):
        # Request wasn't bound through the parser (e.g. constructed
        # directly). Treat every currently-set field as explicit.
        bind_generation_request_raw(normalized, _serialize_generation_request(normalized))
    return normalized


def request_to_sampling_param(
    request: GenerationRequest,
    *,
    model_path: str,
) -> SamplingParam:
    if request.plan is not None:
        raise NotImplementedError("GenerationRequest.plan is not wired into VideoGenerator yet")

    sampling_param = SamplingParam.from_pretrained(model_path)
    if request.state is not None:
        _validate_continuation_state(request.state)
        sampling_param.continuation_state = request.state
    if request.output.return_state:
        sampling_param.return_continuation_state = True
    updates = explicit_request_updates(request)

    for key, value in updates.items():
        if hasattr(sampling_param, key):
            setattr(sampling_param, key, deepcopy(value))
        elif key in REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS:
            continue
        elif value == _SCHEMA_DEFAULT_UPDATES.get(key, _MISSING):
            # Schema-default field that isn't on SamplingParam; tolerated
            # because direct GenerationRequest(...) construction has no
            # way to distinguish "user set" from "schema default".
            continue
        else:
            raise ValueError(f"Request field {key!r} is not supported by sampling params for {model_path}")

    sampling_param.__post_init__()
    sampling_param.check_sampling_param()
    return sampling_param


def expand_request_prompt_batch(request: GenerationRequest, ) -> list[GenerationRequest]:
    if not isinstance(request.prompt, list):
        return [request]

    requests: list[GenerationRequest] = []
    for index, prompt in enumerate(request.prompt):
        single_request = deepcopy(request)
        # deepcopy preserves the tracking-root cycle, but re-pin roots
        # defensively so that subsequent setattrs record on the copy.
        reset_tracking_roots(single_request)
        single_request.prompt = prompt
        _fan_out_batched_input_value(request, single_request, "image_path", index)
        _fan_out_batched_input_value(request, single_request, "video_path", index)
        requests.append(single_request)
    return requests


def _looks_like_run_or_serve_config(raw: Mapping[str, Any]) -> bool:
    return isinstance(raw.get("generator"), Mapping)


def _schema_fields(config_type: type, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(dotted path, field, annotation)`` for every field under ``config_type`` that is not a nested config.

    A field whose type is a dataclass, or an optional dataclass, is a nested config and is walked into.
    """
    type_hints = get_type_hints(config_type)
    for config_field in fields(config_type):
        dotted_path = f"{prefix}{config_field.name}"
        annotation = type_hints[config_field.name]
        nested = [arg for arg in (annotation, *get_args(annotation)) if isinstance(arg, type) and is_dataclass(arg)]
        if nested:
            yield from _schema_fields(nested[0], f"{dotted_path}.")
        else:
            yield dotted_path, config_field, annotation


# Flat keyword name -> (dotted path, annotation) for every GeneratorConfig field that declares a flat name.
_FLAT_NAME_FIELDS: dict[str, tuple[str, Any]] = {
    config_field.metadata[FLAT_NAME]: (dotted_path, annotation)
    for dotted_path, config_field, annotation in _schema_fields(GeneratorConfig) if FLAT_NAME in config_field.metadata
}
# GeneratorConfig fields without a flat name, and how generator_config_to_fastvideo_args carries each one.
_SPECIALLY_MAPPED_PATHS: dict[str, str] = {
    **{
        f"engine.compile.{key}": "merged into torch_compile_kwargs"
        for key in (*_COMPILE_TYPED_KEYS, "extras")
    },
    "engine.quantization.transformer_quant": "resolved to a QuantizationConfig instance",
    "pipeline.preset": "not supported",
    "pipeline.preset_version": "not supported",
    "pipeline.components.vae_weights": "not supported",
    "pipeline.ltx2.refine.image_crf": "no flat keyword; copied from pipeline.preset_overrides.refine",
    "pipeline.ltx2.refine.video_position_offset_sec": "no flat keyword; copied from pipeline.preset_overrides.refine",
    "pipeline.dit": "keys passed as dit_config.<key>",
    "pipeline.vae": "keys passed as vae_config.<key>",
    "pipeline.preset_overrides": "keys passed as flat keywords; refine keys renamed to ltx2_refine_*",
    "pipeline.experimental": "keys passed as flat keywords",
}
# Flat keyword prefix for a component config override -> the PipelineSelection dict that holds the overrides.
_COMPONENT_OVERRIDE_PREFIXES = {
    "dit_config.": "dit",
    "vae_config.": "vae",
}


def _compile_config_to_torch_kwargs(compile_config: CompileConfig, ) -> dict[str, Any]:
    """Flatten typed ``CompileConfig`` back to a ``torch_compile_kwargs``
    dict that the legacy ``FastVideoArgs`` path still expects.

    Typed first-class fields (:attr:`backend`, :attr:`fullgraph`,
    :attr:`mode`, :attr:`dynamic`) are only emitted when the user set
    them explicitly (non-``None``). ``extras`` is merged on top for any
    uncommon kwargs.
    """
    out: dict[str, Any] = {}
    for key in _COMPILE_TYPED_KEYS:
        value = getattr(compile_config, key)
        if value is not None:
            out[key] = value
    if compile_config.extras:
        out.update(deepcopy(compile_config.extras))
    return out


def request_to_batch_extra(request: GenerationRequest) -> dict[str, Any]:
    """Extract typed-request extensions consumed through ``ForwardBatch.extra``."""
    return {
        key: deepcopy(value)
        for key, value in explicit_request_updates(request).items() if key in REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS
    }


def explicit_request_raw(request: GenerationRequest) -> dict[str, Any]:
    """The fields that the caller or operator wrote in ``request``, as a nested raw request mapping.

    Like :func:`explicit_request_updates`, it uses the paths tracked during parsing, but it keeps the sections
    (``sampling``, ``stage_overrides``, and so on) instead of flattening them.
    """
    assert hasattr(request, EXPLICIT_PATHS_ATTR), ("GenerationRequest reached explicit_request_raw without tracking; "
                                                   "route it through normalize_generation_request or parse_config")
    return _build_sparse_raw_from_paths(request, get_explicit_paths(request))


def explicit_request_updates(request: GenerationRequest) -> dict[str, Any]:
    """Project a ``GenerationRequest`` down to *explicitly set* fields only.

    Returns a flat kwargs dict suitable for merging into a generator call.
    The projection uses ``_fastvideo_explicit_paths`` (populated during
    ``parse_config`` / raw binding) so schema defaults on the dataclass
    are **not** emitted — only paths the caller/operator actually wrote.

    This is what makes ``ServeConfig.default_request`` work as an
    operator-pinned baseline rather than a full override: a YAML with just
    ``sampling.seed: 42`` yields ``{"seed": 42}``, not the full sampling
    config with its 15 schema defaults.

    Precondition: the request must carry ``_fastvideo_explicit_paths`` —
    populated by :func:`fastvideo.api.parser.parse_config` or
    :func:`fastvideo.api.compat.normalize_generation_request`. Calling on
    a raw ``GenerationRequest()`` asserts.
    """
    assert hasattr(request,
                   EXPLICIT_PATHS_ATTR), ("GenerationRequest reached explicit_request_updates without tracking; "
                                          "every entry point must route through normalize_generation_request "
                                          "or parse_config first")
    paths = get_explicit_paths(request)
    raw = _build_sparse_raw_from_paths(request, paths)
    return _extract_request_updates(raw)


def _build_sparse_raw_from_paths(
    request: GenerationRequest,
    paths: frozenset[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        parts = path.split(".")
        value = _read_dotted_path(request, parts)
        if value is _MISSING:
            continue
        _set_dotted_path(result, parts, deepcopy(value))
    return result


def _read_dotted_path(obj: Any, parts: list[str]) -> Any:
    for part in parts:
        if is_dataclass(obj) and not isinstance(obj, type):
            if not hasattr(obj, part):
                return _MISSING
            obj = getattr(obj, part)
        elif isinstance(obj, Mapping):
            if part not in obj:
                return _MISSING
            obj = obj[part]
        else:
            return _MISSING
    return obj


def _set_dotted_path(
    target: dict[str, Any],
    parts: list[str],
    value: Any,
) -> None:
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _extract_request_updates(raw: Mapping[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "negative_prompt" in raw:
        updates["negative_prompt"] = deepcopy(raw["negative_prompt"])

    for section_name in ("inputs", "sampling", "runtime", "output"):
        section = raw.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key, value in section.items():
            updates[key] = deepcopy(value)

    stage_overrides = raw.get("stage_overrides")
    if stage_overrides:
        updates.update(_flatten_stage_overrides(stage_overrides))

    extensions = raw.get("extensions")
    if isinstance(extensions, Mapping):
        for key, value in extensions.items():
            updates[key] = deepcopy(value)

    return updates


def _flatten_stage_overrides(stage_overrides: Any) -> dict[str, Any]:
    if not isinstance(stage_overrides, Mapping):
        raise ValueError("GenerationRequest.stage_overrides must be a mapping")

    flattened: dict[str, Any] = {}
    for stage_name, overrides in stage_overrides.items():
        if not isinstance(overrides, Mapping):
            raise ValueError(f"GenerationRequest.stage_overrides.{stage_name} must be a mapping")
        for key, value in overrides.items():
            if key in flattened and flattened[key] != value:
                raise ValueError(f"Conflicting stage override for {key!r} across stages")
            flattened[key] = deepcopy(value)
    return flattened


def _serialize_generation_request(request: GenerationRequest) -> dict[str, Any]:
    return deepcopy(config_to_dict(request))


_SCHEMA_DEFAULT_UPDATES = _extract_request_updates(config_to_dict(GenerationRequest()))

_KNOWN_CONTINUATION_KINDS: set[str] = set()


def register_continuation_kind(kind: str) -> None:
    """Register a :class:`ContinuationState.kind` as recognized.

    PR 7 wires the envelope through; per-kind payload deserializers live
    with each model family (e.g. ``fastvideo.pipelines.basic.ltx2.
    continuation.LTX2ContinuationState``). The registry lets the
    public-API compat layer validate the kind early, before the state
    reaches the pipeline.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("ContinuationState kind must be a non-empty string")
    _KNOWN_CONTINUATION_KINDS.add(kind)


def _validate_continuation_state(state: ContinuationState) -> None:
    if not isinstance(state.kind, str) or not state.kind:
        raise ValueError("GenerationRequest.state.kind must be a non-empty string; got "
                         f"{state.kind!r}")
    if not isinstance(state.payload, Mapping):
        raise ValueError(f"GenerationRequest.state.payload must be a mapping; got "
                         f"{type(state.payload).__name__}")
    if state.kind not in _KNOWN_CONTINUATION_KINDS:
        known = sorted(_KNOWN_CONTINUATION_KINDS)
        raise ValueError(f"Unknown ContinuationState kind {state.kind!r}; registered "
                         f"kinds: {known}. Import the model family that owns this kind "
                         "(e.g. `import fastvideo.pipelines.basic.ltx2.continuation`) "
                         "to register it, or drop the state field.")


def _fan_out_batched_input_value(
    source_request: GenerationRequest,
    target_request: GenerationRequest,
    field_name: str,
    index: int,
) -> None:
    value = getattr(source_request.inputs, field_name)
    if not isinstance(value, list):
        return
    _validate_batched_input_length(source_request.prompt, value, field_name)
    setattr(target_request.inputs, field_name, deepcopy(value[index]))


def _validate_batched_input_length(
    prompts: str | list[str] | None,
    values: list[Any],
    field_name: str,
) -> None:
    if not isinstance(prompts, list):
        return
    if len(values) != len(prompts):
        raise ValueError(f"GenerationRequest.inputs.{field_name} must have the same length as request.prompt")


__all__ = [
    "FROM_PRETRAINED_KWARGS",
    "REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS",
    "explicit_request_raw",
    "explicit_request_updates",
    "from_pretrained_kwargs_to_config",
    "generator_config_to_fastvideo_args",
    "generator_kwargs",
    "load_generator_config_from_file",
    "normalize_generation_request",
    "normalize_generator_config",
    "register_continuation_kind",
    "request_to_batch_extra",
    "request_to_sampling_param",
]
