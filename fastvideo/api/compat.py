# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

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
    InputConfig,
    OutputConfig,
    RequestRuntimeConfig,
    SamplingConfig,
)
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.ltx2.stage_overrides import (
    refine_preset_override_fields,
    refine_stage_override_fields,
)
from fastvideo.utils import shallow_asdict

_INPUT_FIELD_NAMES = {field.name for field in fields(InputConfig)}
_SAMPLING_FIELD_NAMES = {field.name for field in fields(SamplingConfig)}
_RUNTIME_FIELD_NAMES = {field.name for field in fields(RequestRuntimeConfig)}
_OUTPUT_FIELD_NAMES = {field.name for field in fields(OutputConfig)}
_MISSING = object()
_LEGACY_REQUEST_ALIASES = {
    "neg_prompt": "negative_prompt",
}
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


def legacy_from_pretrained_to_config(
    model_path: str,
    kwargs: Mapping[str, Any],
) -> GeneratorConfig:
    """Build a ``GeneratorConfig`` from ``VideoGenerator.from_pretrained`` keyword arguments.

    A keyword that a schema field declares as its flat name sets that field. The keywords in the branches below
    need a conversion instead, and any other keyword is kept in ``pipeline.experimental``.
    """
    raw: dict[str, Any] = {"model_path": model_path}
    experimental: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key == "torch_compile_kwargs":
            remaining: dict[str, Any] = (dict(deepcopy(value)) if isinstance(value, Mapping) else {})
            for first_class in _COMPILE_TYPED_KEYS:
                if first_class in remaining:
                    _set_dotted_path(raw, ["engine", "compile", first_class], remaining.pop(first_class))
            if remaining:
                _set_dotted_path(raw, ["engine", "compile", "extras"], remaining)
        elif key == "pipeline_config" and not isinstance(value, str):
            experimental[key] = deepcopy(value)
        elif key.startswith(tuple(_COMPONENT_OVERRIDE_PREFIXES)):
            component, field_name = key.split(".", 1)
            _set_dotted_path(raw, ["pipeline", _COMPONENT_OVERRIDE_PREFIXES[component + "."], field_name], value)
        elif key in _LTX2_REFINE_PRESET_KEYWORDS:
            _set_dotted_path(raw, ["pipeline", "preset_overrides", "refine", key[len("ltx2_refine_"):]], value)
        elif key in _EMPTY_MEANS_UNSET_KEYWORDS:
            # An empty string means "no file"; keep typed None.
            _set_dotted_path(raw, _EMPTY_MEANS_UNSET_KEYWORDS[key].split("."), value or None)
        elif key in _FLAT_NAME_FIELDS:
            dotted_path, annotation = _FLAT_NAME_FIELDS[key]
            if get_origin(annotation) is dict:
                value = dict(deepcopy(value)) if isinstance(value, Mapping) else {}
            _set_dotted_path(raw, dotted_path.split("."), value)
        else:
            experimental[key] = deepcopy(value)

    if experimental:
        _set_dotted_path(raw, ["pipeline", "experimental"], experimental)
    return parse_config(GeneratorConfig, raw)


def generator_config_to_fastvideo_args(
    config: GeneratorConfig | Mapping[str, Any] | ResolvedGeneratorConfig, ) -> FastVideoArgs:
    """Resolve a ``GeneratorConfig`` and flatten it into the ``FastVideoArgs`` that runtime code reads.

    A config that is not resolved yet goes through :func:`resolve_inference_config` first. Every field of the
    resolved config that declares a flat name and holds a value other than ``None`` becomes the keyword of that name.
    The paths in ``_SPECIALLY_MAPPED_PATHS`` are converted below; ``pipeline.preset_overrides`` and then
    ``pipeline.experimental`` are applied last, so their keys win over the typed fields.
    """
    resolved = config if isinstance(config, ResolvedGeneratorConfig) else resolve_inference_config(config)
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
    return FastVideoArgs.from_kwargs(**kwargs, resolved_config=resolved)


def normalize_generation_request(request: GenerationRequest | Mapping[str, Any], ) -> GenerationRequest:
    normalized = (request if isinstance(request, GenerationRequest) else parse_config(GenerationRequest, request))

    if not hasattr(normalized, EXPLICIT_PATHS_ATTR):
        # Request wasn't bound through the parser (e.g. constructed
        # directly). Treat every currently-set field as explicit.
        bind_generation_request_raw(normalized, _serialize_generation_request(normalized))
    return normalized


def legacy_generate_call_to_request(
    prompt: str | None,
    sampling_param: SamplingParam | None,
    *,
    mouse_cond: Any | None = None,
    keyboard_cond: Any | None = None,
    grid_sizes: Any | None = None,
    legacy_kwargs: Mapping[str, Any] | None = None,
) -> GenerationRequest:
    raw = _sampling_param_to_request_raw(sampling_param)
    if prompt is not None:
        raw["prompt"] = prompt

    for key, value in (legacy_kwargs or {}).items():
        _apply_request_field(raw, key, value)

    if mouse_cond is not None:
        raw.setdefault("inputs", {})["mouse_cond"] = mouse_cond
    if keyboard_cond is not None:
        raw.setdefault("inputs", {})["keyboard_cond"] = keyboard_cond
    if grid_sizes is not None:
        raw.setdefault("inputs", {})["grid_sizes"] = grid_sizes

    normalized = parse_config(GenerationRequest, raw)
    bind_generation_request_raw(normalized, raw)
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
# from_pretrained keywords whose empty-string value means "unset", and the field each one sets.
_EMPTY_MEANS_UNSET_KEYWORDS = {
    "ltx2_refine_upsampler_path": "pipeline.components.upsampler_weights",
}
# from_pretrained keywords that set pipeline.preset_overrides.refine.<key without the ltx2_refine_ prefix>.
_LTX2_REFINE_PRESET_KEYWORDS = frozenset({
    "ltx2_refine_enabled",
    "ltx2_refine_add_noise",
    "ltx2_refine_num_inference_steps",
    "ltx2_refine_guidance_scale",
})


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


def _sampling_param_to_request_raw(sampling_param: SamplingParam | None, ) -> dict[str, Any]:
    if sampling_param is None:
        return {}

    raw: dict[str, Any] = {}
    for key, value in shallow_asdict(sampling_param).items():
        if key == "prompt":
            continue
        _apply_request_field(raw, key, deepcopy(value))
    return raw


def _apply_request_field(
    raw: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    key = _LEGACY_REQUEST_ALIASES.get(key, key)
    if key == "negative_prompt":
        raw["negative_prompt"] = value
        return
    if key in _INPUT_FIELD_NAMES:
        raw.setdefault("inputs", {})[key] = value
        return
    if key in _SAMPLING_FIELD_NAMES:
        raw.setdefault("sampling", {})[key] = value
        return
    if key in _RUNTIME_FIELD_NAMES:
        raw.setdefault("runtime", {})[key] = value
        return
    if key in _OUTPUT_FIELD_NAMES:
        raw.setdefault("output", {})[key] = value
        return
    raw.setdefault("extensions", {})[key] = value


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
    "REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS",
    "explicit_request_raw",
    "explicit_request_updates",
    "generator_config_to_fastvideo_args",
    "legacy_from_pretrained_to_config",
    "legacy_generate_call_to_request",
    "load_generator_config_from_file",
    "normalize_generation_request",
    "normalize_generator_config",
    "register_continuation_kind",
    "request_to_batch_extra",
    "request_to_sampling_param",
]
