# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from fastvideo.api.overrides import apply_overrides, normalize_overrides
from fastvideo.api.parser import config_to_dict, load_raw_config, parse_config
from fastvideo.api.request_metadata import (
    EXPLICIT_PATHS_ATTR,
    bind_generation_request_raw,
    get_explicit_paths,
    reset_tracking_roots,
)
from fastvideo.api.schema import (
    CompileConfig,
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
_REQUEST_PIPELINE_OVERRIDE_FIELDS = frozenset({
    "embedded_cfg_scale",
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


def legacy_from_pretrained_to_config(
    model_path: str,
    kwargs: Mapping[str, Any],
) -> GeneratorConfig:
    raw: dict[str, Any] = {"model_path": model_path}
    engine: dict[str, Any] = {}
    parallelism: dict[str, Any] = {}
    offload: dict[str, Any] = {}
    compile_config: dict[str, Any] = {}
    pipeline: dict[str, Any] = {}
    components: dict[str, Any] = {}
    quantization: dict[str, Any] = {}
    experimental: dict[str, Any] = {}
    preset_overrides: dict[str, Any] = {}
    preset_refine: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key == "revision":
            raw["revision"] = value
        elif key == "trust_remote_code":
            raw["trust_remote_code"] = value
        elif key == "num_gpus":
            engine["num_gpus"] = value
        elif key == "distributed_executor_backend":
            engine["execution_backend"] = value
        elif key in {"tp_size", "sp_size", "hsdp_replicate_dim", "hsdp_shard_dim", "dist_timeout"}:
            parallelism[key] = value
        elif key == "dit_cpu_offload":
            offload["dit"] = value
        elif key == "dit_layerwise_offload":
            offload["dit_layerwise"] = value
        elif key == "text_encoder_cpu_offload":
            offload["text_encoder"] = value
        elif key == "image_encoder_cpu_offload":
            offload["image_encoder"] = value
        elif key == "vae_cpu_offload":
            offload["vae"] = value
        elif key == "pin_cpu_memory":
            offload["pin_cpu_memory"] = value
        elif key == "enable_torch_compile":
            compile_config["enabled"] = value
        elif key == "enable_torch_compile_text_encoder":
            compile_config["text_encoder_enabled"] = value
        elif key == "torch_compile_kwargs":
            remaining: dict[str, Any] = (dict(deepcopy(value)) if isinstance(value, Mapping) else {})
            for first_class in _COMPILE_TYPED_KEYS:
                if first_class in remaining:
                    compile_config[first_class] = remaining.pop(first_class)
            if remaining:
                compile_config["extras"] = remaining
        elif key == "ltx2_vae_tiling":
            pipeline["vae_tiling"] = value
        elif key == "config_model_path":
            components["config_root"] = value
        elif key == "ltx2_refine_enabled":
            preset_refine["enabled"] = value
        elif key == "ltx2_refine_upsampler_path":
            # Empty string means "no upsampler"; keep typed None.
            components["upsampler_weights"] = value or None
        elif key == "ltx2_refine_lora_path":
            # Empty string means "no refine LoRA"; keep typed None.
            components["lora_path"] = value or None
        elif key == "ltx2_refine_add_noise":
            preset_refine["add_noise"] = value
        elif key == "ltx2_refine_num_inference_steps":
            preset_refine["num_inference_steps"] = value
        elif key == "ltx2_refine_guidance_scale":
            preset_refine["guidance_scale"] = value
        elif key in {"enable_stage_verification", "use_fsdp_inference", "disable_autocast"}:
            engine[key] = value
        elif key == "override_text_encoder_quant":
            quantization["text_encoder_quant"] = value
        elif key == "workload_type":
            pipeline["workload_type"] = value
        elif key == "lora_path":
            components["lora_path"] = value
        elif key == "override_pipeline_cls_name":
            components["override_pipeline_cls_name"] = value
        elif key == "override_transformer_cls_name":
            components["override_transformer_cls_name"] = value
        elif key == "pipeline_config":
            if isinstance(value, str):
                components["pipeline_config_path"] = value
            else:
                experimental[key] = deepcopy(value)
        elif key == "override_text_encoder_safetensors":
            components["text_encoder_weights"] = value
        elif key == "init_weights_from_safetensors":
            components["transformer_weights"] = value
        elif key == "init_weights_from_safetensors_2":
            components["transformer_2_weights"] = value
        else:
            experimental[key] = deepcopy(value)

    if parallelism:
        engine["parallelism"] = parallelism
    if offload:
        engine["offload"] = offload
    if compile_config:
        engine["compile"] = compile_config
    if quantization:
        engine["quantization"] = quantization
    if engine:
        raw["engine"] = engine

    if components:
        pipeline["components"] = components
    if preset_refine:
        preset_overrides["refine"] = preset_refine
    if preset_overrides:
        pipeline["preset_overrides"] = preset_overrides
    if experimental:
        pipeline["experimental"] = experimental
    if pipeline:
        raw["pipeline"] = pipeline

    return parse_config(GeneratorConfig, raw)


def generator_config_to_fastvideo_args(config: GeneratorConfig | Mapping[str, Any], ) -> FastVideoArgs:
    normalized = normalize_generator_config(config)
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

    engine = normalized.engine
    kwargs: dict[str, Any] = {
        "model_path": normalized.model_path,
        "revision": normalized.revision,
        "trust_remote_code": normalized.trust_remote_code,
        "num_gpus": engine.num_gpus,
        "distributed_executor_backend": engine.execution_backend,
        "tp_size": engine.parallelism.tp_size,
        "sp_size": engine.parallelism.sp_size,
        "hsdp_replicate_dim": engine.parallelism.hsdp_replicate_dim,
        "hsdp_shard_dim": engine.parallelism.hsdp_shard_dim,
        "dist_timeout": engine.parallelism.dist_timeout,
        "dit_cpu_offload": engine.offload.dit,
        "dit_layerwise_offload": engine.offload.dit_layerwise,
        "text_encoder_cpu_offload": engine.offload.text_encoder,
        "image_encoder_cpu_offload": engine.offload.image_encoder,
        "vae_cpu_offload": engine.offload.vae,
        "pin_cpu_memory": engine.offload.pin_cpu_memory,
        "enable_torch_compile": engine.compile.enabled,
        "torch_compile_kwargs": _compile_config_to_torch_kwargs(engine.compile),
        "enable_stage_verification": engine.enable_stage_verification,
        "use_fsdp_inference": engine.use_fsdp_inference,
        "disable_autocast": engine.disable_autocast,
    }
    if normalized.pipeline.workload_type is not None:
        kwargs["workload_type"] = normalized.pipeline.workload_type
    if normalized.pipeline.vae_tiling is not None:
        kwargs["ltx2_vae_tiling"] = normalized.pipeline.vae_tiling
    if engine.compile.text_encoder_enabled is not None:
        # ``FastVideoArgs.from_kwargs`` filters to declared fields, so
        # this is a no-op on the current legacy path. Emit anyway so the
        # realtime runtime (PR 7.6) — which reads from the kwargs dict
        # before FastVideoArgs filtering — can pick it up once wired.
        kwargs["enable_torch_compile_text_encoder"] = (engine.compile.text_encoder_enabled)

    quantization = engine.quantization
    if quantization is not None and quantization.text_encoder_quant is not None:
        kwargs["override_text_encoder_quant"] = quantization.text_encoder_quant
    if quantization is not None and quantization.transformer_quant is not None:
        kwargs["transformer_quant"] = quantization.transformer_quant

    components = normalized.pipeline.components
    if components.pipeline_config_path is not None:
        kwargs["pipeline_config"] = components.pipeline_config_path
    if components.lora_path is not None:
        kwargs["lora_path"] = components.lora_path
    if components.override_pipeline_cls_name is not None:
        kwargs["override_pipeline_cls_name"] = components.override_pipeline_cls_name
    if components.override_transformer_cls_name is not None:
        kwargs["override_transformer_cls_name"] = components.override_transformer_cls_name
    if components.text_encoder_weights is not None:
        kwargs["override_text_encoder_safetensors"] = components.text_encoder_weights
    if components.transformer_weights is not None:
        kwargs["init_weights_from_safetensors"] = components.transformer_weights
    if components.transformer_2_weights is not None:
        kwargs["init_weights_from_safetensors_2"] = components.transformer_2_weights
    if components.config_root is not None:
        kwargs["config_model_path"] = components.config_root
    if components.upsampler_weights is not None:
        kwargs["ltx2_refine_upsampler_path"] = components.upsampler_weights

    preset_overrides = deepcopy(normalized.pipeline.preset_overrides)
    refine = preset_overrides.pop("refine", None)
    if isinstance(refine, Mapping):
        for key in _LTX2_REFINE_FLAT_KEYS:
            if key in refine:
                kwargs[f"ltx2_refine_{key}"] = refine[key]
    kwargs.update(preset_overrides)
    kwargs.update(deepcopy(normalized.pipeline.experimental))
    return FastVideoArgs.from_kwargs(**kwargs)


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
    if request.state is not None:
        raise NotImplementedError("GenerationRequest.state is not wired into VideoGenerator yet")

    sampling_param = SamplingParam.from_pretrained(model_path)
    updates = explicit_request_updates(request)

    for key, value in updates.items():
        if hasattr(sampling_param, key):
            setattr(sampling_param, key, deepcopy(value))
        elif key in _REQUEST_PIPELINE_OVERRIDE_FIELDS:
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


def request_to_pipeline_overrides(request: GenerationRequest) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in explicit_request_updates(request).items():
        if key in _REQUEST_PIPELINE_OVERRIDE_FIELDS:
            overrides[key] = deepcopy(value)
    return overrides


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
    "explicit_request_updates",
    "generator_config_to_fastvideo_args",
    "legacy_from_pretrained_to_config",
    "legacy_generate_call_to_request",
    "load_generator_config_from_file",
    "normalize_generation_request",
    "normalize_generator_config",
    "request_to_pipeline_overrides",
    "request_to_sampling_param",
]
