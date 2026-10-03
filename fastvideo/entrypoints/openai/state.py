"""Global server state shared across API modules.

Keeping state in a dedicated module gives every API module the same
generator and resolved config. All modules that need them should import from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.api.resolution import ResolvedGeneratorConfig
    from fastvideo.api.schema import GenerationRequest
    from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine, ServingGenerator

DEFAULT_OUTPUT_DIR = "outputs"

_generator: ServingGenerator | None = None
_serving_engine: OpenAIServingEngine | None = None
_resolved_config: ResolvedGeneratorConfig | None = None
_output_dir: str = DEFAULT_OUTPUT_DIR
_served_model_name: str | None = None
_default_request: GenerationRequest | None = None


def get_generator() -> ServingGenerator:
    """Return the global VideoGenerator instance (set during startup)."""
    assert _generator is not None, "Server not initialized — generator is None"
    return _generator


def get_serving_engine() -> OpenAIServingEngine:
    """Return the shared model-agnostic OpenAI serving engine."""
    assert _serving_engine is not None, "Server not initialized — serving engine is None"
    return _serving_engine


def get_resolved_config() -> ResolvedGeneratorConfig:
    """Return the resolved runtime config of the served generator (set during startup)."""
    assert _resolved_config is not None, "Server not initialized — resolved config is None"
    return _resolved_config


def get_output_dir() -> str:
    """Return the configured output directory."""
    return _output_dir


def get_served_model_name() -> str:
    """Return the public model id advertised by the OpenAI server."""
    resolved_config = get_resolved_config()
    components = resolved_config.pipeline.components
    if components.lora_path:
        return components.lora_nickname
    return _served_model_name or resolved_config.model_path


def get_default_request() -> GenerationRequest | None:
    """Return the ServeConfig.default_request set at startup, if any."""
    return _default_request


def set_state(
    generator: ServingGenerator,
    serving_engine: OpenAIServingEngine,
    resolved_config: ResolvedGeneratorConfig,
    output_dir: str,
    default_request: GenerationRequest | None = None,
    served_model_name: str | None = None,
) -> None:
    """Set all server state at once (called from lifespan)."""
    global _generator, _serving_engine, _resolved_config, _output_dir, _served_model_name, _default_request
    _generator = generator
    _serving_engine = serving_engine
    _resolved_config = resolved_config
    _output_dir = output_dir
    _served_model_name = served_model_name
    _default_request = default_request


def clear_state() -> None:
    """Clear server state on shutdown."""
    global _generator, _serving_engine, _resolved_config, _served_model_name, _default_request
    _generator = None
    _serving_engine = None
    _resolved_config = None
    _served_model_name = None
    _default_request = None
