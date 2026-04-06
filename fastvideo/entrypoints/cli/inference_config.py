# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from typing import Any

from fastvideo.api.overrides import apply_overrides, parse_cli_overrides
from fastvideo.api.parser import load_raw_config, parse_config
from fastvideo.api.schema import (
    InputConfig,
    OutputConfig,
    RequestRuntimeConfig,
    RunConfig,
    SamplingConfig,
    ServeConfig,
)

_INPUT_FIELD_NAMES = {field.name for field in fields(InputConfig)}
_SAMPLING_FIELD_NAMES = {field.name for field in fields(SamplingConfig)}
_RUNTIME_FIELD_NAMES = {field.name for field in fields(RequestRuntimeConfig)}
_OUTPUT_FIELD_NAMES = {field.name for field in fields(OutputConfig)}

_GENERATE_EXCLUDED_FIELDS = {
    "_provided",
    "_unknown",
    "config",
    "dispatch_function",
    "subparser",
}
_SERVE_EXCLUDED_FIELDS = _GENERATE_EXCLUDED_FIELDS | {
    "host",
    "output_dir",
    "port",
}

_GENERATOR_OVERRIDE_PATHS = {
    "model_path": "generator.model_path",
    "revision": "generator.revision",
    "trust_remote_code": "generator.trust_remote_code",
    "num_gpus": "generator.engine.num_gpus",
    "distributed_executor_backend": "generator.engine.execution_backend",
    "tp_size": "generator.engine.parallelism.tp_size",
    "sp_size": "generator.engine.parallelism.sp_size",
    "hsdp_replicate_dim": "generator.engine.parallelism.hsdp_replicate_dim",
    "hsdp_shard_dim": "generator.engine.parallelism.hsdp_shard_dim",
    "dist_timeout": "generator.engine.parallelism.dist_timeout",
    "dit_cpu_offload": "generator.engine.offload.dit",
    "dit_layerwise_offload": "generator.engine.offload.dit_layerwise",
    "text_encoder_cpu_offload": "generator.engine.offload.text_encoder",
    "image_encoder_cpu_offload": "generator.engine.offload.image_encoder",
    "vae_cpu_offload": "generator.engine.offload.vae",
    "pin_cpu_memory": "generator.engine.offload.pin_cpu_memory",
    "enable_torch_compile": "generator.engine.compile.enabled",
    "torch_compile_kwargs": "generator.engine.compile.kwargs",
    "enable_stage_verification": "generator.engine.enable_stage_verification",
    "use_fsdp_inference": "generator.engine.use_fsdp_inference",
    "disable_autocast": "generator.engine.disable_autocast",
    "override_text_encoder_quant": "generator.engine.quantization.text_encoder_quant",
    "workload_type": "generator.pipeline.workload_type",
    "pipeline_config": "generator.pipeline.components.pipeline_config_path",
    "lora_path": "generator.pipeline.components.lora_path",
    "override_pipeline_cls_name": "generator.pipeline.components.override_pipeline_cls_name",
    "override_transformer_cls_name": "generator.pipeline.components.override_transformer_cls_name",
    "override_text_encoder_safetensors": "generator.pipeline.components.text_encoder_weights",
    "init_weights_from_safetensors": "generator.pipeline.components.transformer_weights",
    "init_weights_from_safetensors_2": "generator.pipeline.components.transformer_2_weights",
}
_SERVE_OVERRIDE_PATHS = {
    "host": "server.host",
    "port": "server.port",
    "output_dir": "server.output_dir",
}


def build_generate_run_config(
    args: argparse.Namespace,
    overrides: list[str] | None = None,
) -> RunConfig:
    raw = _load_generate_raw_config(getattr(args, "config", None))
    raw = _apply_cli_overrides(
        raw,
        _translate_generate_flat_values(_namespace_provided_values(args, excluded=_GENERATE_EXCLUDED_FIELDS)),
        overrides,
    )
    _ensure_generate_cli_defaults(raw)
    config = parse_config(RunConfig, raw)
    _validate_generate_prompt_sources(config)
    return config


def build_serve_config(
    args: argparse.Namespace,
    overrides: list[str] | None = None,
) -> ServeConfig:
    raw = _load_serve_raw_config(getattr(args, "config", None))
    raw = _apply_cli_overrides(
        raw,
        _translate_serve_flat_values(_namespace_provided_values(args, excluded=_SERVE_EXCLUDED_FIELDS)),
        overrides,
    )
    raw = _apply_cli_overrides(
        raw,
        _translate_server_values(_namespace_provided_values(args, excluded=_GENERATE_EXCLUDED_FIELDS)),
        None,
    )
    return parse_config(ServeConfig, raw)


def _load_generate_raw_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "generator": {},
            "request": {},
        }

    raw = load_raw_config(path)
    if _looks_like_run_config(raw):
        loaded = dict(raw)
        loaded.setdefault("request", {})
        return loaded
    return _translate_generate_flat_config(raw)


def _load_serve_raw_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "generator": {},
            "server": {},
            "default_request": {},
        }

    raw = load_raw_config(path)
    if _looks_like_serve_config(raw):
        loaded = dict(raw)
        loaded.setdefault("server", {})
        loaded.setdefault("default_request", {})
        return loaded
    return _translate_serve_flat_config(raw)


def _looks_like_run_config(raw: Mapping[str, Any]) -> bool:
    return isinstance(raw.get("generator"), Mapping)


def _looks_like_serve_config(raw: Mapping[str, Any]) -> bool:
    return isinstance(raw.get("generator"), Mapping)


def _translate_generate_flat_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {
        "generator": {},
        "request": {},
    }
    translated = apply_overrides(translated, _translate_generate_flat_values(raw))
    return translated


def _translate_serve_flat_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {
        "generator": {},
        "server": {},
        "default_request": {},
    }
    translated = apply_overrides(translated, _translate_serve_flat_values(raw))
    translated = apply_overrides(translated, _translate_server_values(raw))
    return translated


def _apply_cli_overrides(
    raw: Mapping[str, Any],
    translated_overrides: Mapping[str, Any] | None,
    dotted_overrides: list[str] | None,
) -> dict[str, Any]:
    merged = deepcopy(dict(raw))
    if translated_overrides:
        merged = apply_overrides(merged, translated_overrides)
    if dotted_overrides:
        merged = apply_overrides(merged, parse_cli_overrides(dotted_overrides))
    return merged


def _namespace_provided_values(
    args: argparse.Namespace,
    *,
    excluded: set[str],
) -> dict[str, Any]:
    provided: set[str] = getattr(args, "_provided", set())
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in excluded or key not in provided or value is None:
            continue
        values[key] = value
    return values


def _translate_generate_flat_values(values: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in values.items():
        if key == "prompt":
            translated["request.prompt"] = value
        elif key in {"prompt_txt", "prompt_path"}:
            translated["request.inputs.prompt_path"] = value
        else:
            path = _request_override_path(key)
            if path is not None:
                translated[path] = value
                continue
            path = _GENERATOR_OVERRIDE_PATHS.get(key)
            if path is not None:
                translated[path] = value
                continue
            translated[f"generator.pipeline.experimental.{key}"] = value
    return translated


def _translate_serve_flat_values(values: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in values.items():
        if key in _SERVE_OVERRIDE_PATHS:
            continue
        path = _GENERATOR_OVERRIDE_PATHS.get(key)
        if path is not None:
            translated[path] = value
            continue
        translated[f"generator.pipeline.experimental.{key}"] = value
    return translated


def _translate_server_values(values: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in values.items():
        path = _SERVE_OVERRIDE_PATHS.get(key)
        if path is not None:
            translated[path] = value
    return translated


def _request_override_path(key: str) -> str | None:
    if key == "negative_prompt":
        return "request.negative_prompt"
    if key in _INPUT_FIELD_NAMES:
        return f"request.inputs.{key}"
    if key in _SAMPLING_FIELD_NAMES:
        return f"request.sampling.{key}"
    if key in _RUNTIME_FIELD_NAMES:
        return f"request.runtime.{key}"
    if key in _OUTPUT_FIELD_NAMES:
        return f"request.output.{key}"
    return None


def _ensure_generate_cli_defaults(raw: dict[str, Any]) -> None:
    request = raw.setdefault("request", {})
    output = request.setdefault("output", {})
    output.setdefault("return_frames", False)


def _validate_generate_prompt_sources(config: RunConfig) -> None:
    has_prompt = config.request.prompt is not None
    has_prompt_path = config.request.inputs.prompt_path is not None
    if not (has_prompt or has_prompt_path):
        raise ValueError("Either request.prompt or request.inputs.prompt_path must be provided")
    if has_prompt and has_prompt_path:
        raise ValueError("Cannot provide both request.prompt and request.inputs.prompt_path")


__all__ = [
    "build_generate_run_config",
    "build_serve_config",
]
