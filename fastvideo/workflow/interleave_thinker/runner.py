# SPDX-License-Identifier: Apache-2.0
"""Config-driven native InterleaveThinker orchestration runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastvideo.workflow.interleave_thinker.config import (
    InterleaveCriticConfig,
    InterleaveImageBackendConfig,
    InterleavePlannerConfig,
    InterleaveRunConfig,
    resolve_interleave_instruction,
)
from fastvideo.workflow.interleave_thinker.generator import (
    FastVideoImageGeneratorBackend,
    ImageGeneratorBackend,
    NanoBananaImageGeneratorBackend,
)
from fastvideo.workflow.interleave_thinker.orchestrator import (
    AcceptAllCritic,
    CriticProvider,
    InterleaveOrchestrator,
    PlannerProvider,
    SinglePromptPlanner,
)
from fastvideo.workflow.interleave_thinker.providers import (
    InterleaveThinkerCriticProvider,
    InterleaveThinkerPlannerProvider,
)
from fastvideo.workflow.interleave_thinker.schema import InterleaveTrace
from fastvideo.workflow.interleave_thinker.trace import save_trace
from fastvideo.train.models.interleave_thinker import (
    InterleaveThinkerCriticModel,
    InterleaveThinkerPlannerModel,
)


@dataclass(frozen=True)
class InterleaveRunResult:
    trace: InterleaveTrace
    trace_path: str


def run_interleave_config(
    config: InterleaveRunConfig,
    *,
    image_backend: ImageGeneratorBackend | None = None,
) -> InterleaveRunResult:
    """Run one native interleaved generation trace from a typed config."""

    cleanup: Callable[[], None] = lambda: None
    if image_backend is None:
        image_backend, cleanup = build_image_backend(config)

    try:
        orchestrator = InterleaveOrchestrator(
            planner=build_planner(config.planner),
            generator=image_backend,
            critic=build_critic(config.critic),
        )
        trace = orchestrator.run(
            resolve_interleave_instruction(config),
            initial_image_path=config.interleave.initial_image_path,
            metadata={
                "image_backend": config.image_backend.kind,
                "planner": config.planner.kind,
                "critic": config.critic.kind,
            },
        )
        trace_path = resolve_trace_path(config)
        save_trace(
            trace,
            trace_path,
            include_images=config.interleave.include_images_in_trace,
        )
        return InterleaveRunResult(
            trace=trace,
            trace_path=str(trace_path),
        )
    finally:
        cleanup()


def resolve_trace_path(config: InterleaveRunConfig) -> Path:
    if config.interleave.trace_path:
        return Path(config.interleave.trace_path)
    return Path(config.interleave.output_dir) / "trace.json"


def build_planner(config: InterleavePlannerConfig) -> PlannerProvider:
    if config.kind == "single_prompt":
        return SinglePromptPlanner(max_attempts=config.max_attempts_per_step)

    model = InterleaveThinkerPlannerModel(**_actor_model_kwargs(config), )
    return InterleaveThinkerPlannerProvider(
        model,
        num_generations=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens or config.max_response_length,
        max_attempts_per_step=config.max_attempts_per_step,
    )


def build_critic(config: InterleaveCriticConfig) -> CriticProvider | None:
    if config.kind == "none":
        return None
    if config.kind == "accept_all":
        return AcceptAllCritic()

    model = InterleaveThinkerCriticModel(**_actor_model_kwargs(config), )
    return InterleaveThinkerCriticProvider(
        model,
        num_generations=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens or config.max_response_length,
    )


def build_image_backend(config: InterleaveRunConfig) -> tuple[ImageGeneratorBackend, Callable[[], None]]:
    image_config = config.image_backend
    output_dir = image_config.output_dir or config.interleave.output_dir
    if image_config.kind == "nano_banana":
        return (
            NanoBananaImageGeneratorBackend(
                model=image_config.model,
                api_key=image_config.api_key,
                base_url=image_config.base_url,
                output_dir=output_dir,
                aspect_ratio=image_config.aspect_ratio,
                image_size=image_config.image_size,
                max_attempts=image_config.max_attempts,
                retry_delay_s=image_config.retry_delay_s,
            ),
            lambda: None,
        )
    return _build_fastvideo_image_backend(config, image_config, output_dir)


def _build_fastvideo_image_backend(
    config: InterleaveRunConfig,
    image_config: InterleaveImageBackendConfig,
    output_dir: str,
) -> tuple[ImageGeneratorBackend, Callable[[], None]]:
    del image_config
    if config.generator is None:
        raise ValueError("FastVideo image backend requires config.generator")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_config(config.generator)

    def cleanup() -> None:
        generator.shutdown()

    return (
        FastVideoImageGeneratorBackend(
            generator,
            output_dir=output_dir,
            default_request=config.request,
        ),
        cleanup,
    )


def _actor_model_kwargs(config: InterleavePlannerConfig | InterleaveCriticConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "load_backend": config.load_backend,
        "trainable": config.trainable,
        "image_dir": config.image_dir,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": config.trust_remote_code,
        "use_cache": config.use_cache,
        "max_prompt_length": config.max_prompt_length,
        "max_response_length": config.max_response_length,
        "lora": config.lora,
    }
    if config.init_from is not None:
        kwargs["init_from"] = config.init_from
    if config.processor_from is not None:
        kwargs["processor_from"] = config.processor_from
    return kwargs


__all__ = [
    "InterleaveRunResult",
    "build_critic",
    "build_image_backend",
    "build_planner",
    "resolve_trace_path",
    "run_interleave_config",
]
