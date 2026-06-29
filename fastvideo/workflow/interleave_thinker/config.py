# SPDX-License-Identifier: Apache-2.0
"""Typed config for native interleaved generation workflows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastvideo.api.overrides import apply_overrides, parse_cli_overrides as parse_dotted_overrides
from fastvideo.api.parser import load_raw_config, parse_config
from fastvideo.api.request_metadata import bind_generation_request_raw
from fastvideo.api.schema import GenerationRequest, GeneratorConfig


@dataclass
class InterleaveRunStateConfig:
    instruction: str | None = None
    initial_image_path: str | None = None
    output_dir: str = "outputs/interleave_run"
    trace_path: str | None = None
    include_images_in_trace: bool = False


@dataclass
class InterleaveImageBackendConfig:
    kind: Literal["fastvideo", "nano_banana"] = "fastvideo"
    output_dir: str | None = None
    model: str = "gemini-3.1-flash-image"
    api_key: str | None = None
    base_url: str | None = None
    aspect_ratio: str | None = None
    image_size: str | None = None
    max_attempts: int = 3
    retry_delay_s: float = 2.0


@dataclass
class InterleavePlannerConfig:
    kind: Literal["single_prompt", "interleave_thinker"] = "single_prompt"
    init_from: str | None = None
    processor_from: str | None = None
    load_backend: bool = True
    trainable: bool = False
    image_dir: str = ""
    torch_dtype: str = "auto"
    device_map: Any | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    use_cache: bool = False
    max_prompt_length: int = 16384
    max_response_length: int = 4096
    lora: dict[str, Any] | None = None
    num_generations: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int | None = None
    max_attempts_per_step: int = 2


@dataclass
class InterleaveCriticConfig:
    kind: Literal["none", "accept_all", "interleave_thinker"] = "accept_all"
    init_from: str | None = None
    processor_from: str | None = None
    load_backend: bool = True
    trainable: bool = False
    image_dir: str = ""
    torch_dtype: str = "auto"
    device_map: Any | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    use_cache: bool = False
    max_prompt_length: int = 16384
    max_response_length: int = 4096
    lora: dict[str, Any] | None = None
    num_generations: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int | None = None


@dataclass
class InterleaveRunConfig:
    interleave: InterleaveRunStateConfig = field(default_factory=InterleaveRunStateConfig)
    image_backend: InterleaveImageBackendConfig = field(default_factory=InterleaveImageBackendConfig)
    planner: InterleavePlannerConfig = field(default_factory=InterleavePlannerConfig)
    critic: InterleaveCriticConfig = field(default_factory=InterleaveCriticConfig)
    request: GenerationRequest = field(default_factory=GenerationRequest)
    generator: GeneratorConfig | None = None


_INTERLEAVE_RUN_OVERRIDE_PREFIXES = (
    "interleave.",
    "image_backend.",
    "planner.",
    "critic.",
    "request.",
    "generator.",
)


def load_interleave_run_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    prompt: str | None = None,
    input_image: str | None = None,
    output_dir: str | None = None,
    trace_path: str | None = None,
    require_instruction: bool = True,
) -> InterleaveRunConfig:
    raw = load_raw_config(path)
    raw = _apply_interleave_runtime_fields(
        raw,
        prompt=prompt,
        input_image=input_image,
        output_dir=output_dir,
        trace_path=trace_path,
    )
    raw = _apply_interleave_overrides(raw, overrides)
    config = parse_config(InterleaveRunConfig, raw)
    bind_generation_request_raw(
        config.request,
        raw.get("request") if isinstance(raw.get("request"), Mapping) else {},
    )
    validate_interleave_run_config(
        config,
        require_instruction=require_instruction,
    )
    return config


def resolve_interleave_instruction(config: InterleaveRunConfig) -> str:
    if config.interleave.instruction:
        return config.interleave.instruction
    if isinstance(config.request.prompt, str) and config.request.prompt:
        return config.request.prompt
    if isinstance(config.request.prompt, list) and len(config.request.prompt) == 1:
        prompt = config.request.prompt[0]
        if isinstance(prompt, str) and prompt:
            return prompt
    raise ValueError("Interleave config requires interleave.instruction or a single request.prompt")


def validate_interleave_run_config(
    config: InterleaveRunConfig,
    *,
    require_instruction: bool = True,
) -> None:
    if require_instruction:
        resolve_interleave_instruction(config)
    if config.image_backend.kind == "fastvideo" and config.generator is None:
        raise ValueError("Interleave config with image_backend.kind=fastvideo requires a generator config")
    if config.planner.kind == "interleave_thinker" and config.planner.max_new_tokens is not None:
        _require_positive_int(config.planner.max_new_tokens, "planner.max_new_tokens")
    _require_positive_int(config.planner.max_attempts_per_step, "planner.max_attempts_per_step")
    if config.critic.kind == "interleave_thinker" and config.critic.max_new_tokens is not None:
        _require_positive_int(config.critic.max_new_tokens, "critic.max_new_tokens")


def _apply_interleave_runtime_fields(
    raw: Mapping[str, Any],
    *,
    prompt: str | None,
    input_image: str | None,
    output_dir: str | None,
    trace_path: str | None,
) -> dict[str, Any]:
    merged = deepcopy(dict(raw))
    interleave = merged.setdefault("interleave", {})
    if not isinstance(interleave, dict):
        raise ValueError("interleave must be a mapping")
    if prompt is not None:
        interleave["instruction"] = prompt
    if input_image is not None:
        interleave["initial_image_path"] = input_image
    if output_dir is not None:
        interleave["output_dir"] = output_dir
    if trace_path is not None:
        interleave["trace_path"] = trace_path
    return merged


def _apply_interleave_overrides(
    raw: Mapping[str, Any],
    overrides: list[str] | None,
) -> dict[str, Any]:
    if not overrides:
        return deepcopy(dict(raw))

    parsed = parse_dotted_overrides(overrides)
    for key in parsed:
        if "." not in key:
            raise ValueError("Overrides must use dotted config paths like --request.sampling.seed 42")
        if not key.startswith(_INTERLEAVE_RUN_OVERRIDE_PREFIXES):
            allowed = ", ".join(_INTERLEAVE_RUN_OVERRIDE_PREFIXES)
            raise ValueError(f"Unsupported override path {key!r}. Allowed prefixes: {allowed}")
    return apply_overrides(raw, parsed)


def _require_positive_int(value: int, path: str) -> None:
    if value <= 0:
        raise ValueError(f"{path} must be > 0; got {value}")


__all__ = [
    "InterleaveCriticConfig",
    "InterleaveImageBackendConfig",
    "InterleavePlannerConfig",
    "InterleaveRunConfig",
    "InterleaveRunStateConfig",
    "load_interleave_run_config",
    "resolve_interleave_instruction",
    "validate_interleave_run_config",
]
