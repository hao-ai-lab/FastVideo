"""Global server state shared across API modules.

Keeping state in a dedicated module prevents the classic '__main__ vs package
module' duplication that occurs when api_server.py is run with ``python -m``.
All modules that need the generator or server args should import from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.api.schema import GenerationRequest
    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.fastvideo_args import FastVideoArgs

DEFAULT_OUTPUT_DIR = "outputs"

_generator: VideoGenerator | None = None
_fastvideo_args: FastVideoArgs | None = None
_output_dir: str = DEFAULT_OUTPUT_DIR
_default_request: GenerationRequest | None = None


def get_generator() -> VideoGenerator:
    """Return the global VideoGenerator instance (set during startup)."""
    assert _generator is not None, "Server not initialized — generator is None"
    return _generator


def get_server_args() -> FastVideoArgs:
    """Return the global FastVideoArgs (set during startup)."""
    assert _fastvideo_args is not None, "Server not initialized — args is None"
    return _fastvideo_args


def get_output_dir() -> str:
    """Return the configured output directory."""
    return _output_dir


def get_default_request() -> GenerationRequest | None:
    """Return the ServeConfig.default_request set at startup, if any."""
    return _default_request


def set_state(
    generator: VideoGenerator,
    fastvideo_args: FastVideoArgs,
    output_dir: str,
    default_request: GenerationRequest | None = None,
) -> None:
    """Set all server state at once (called from lifespan)."""
    global _generator, _fastvideo_args, _output_dir, _default_request
    _generator = generator
    _fastvideo_args = fastvideo_args
    _output_dir = output_dir
    _default_request = default_request


def clear_state() -> None:
    """Clear server state on shutdown."""
    global _generator, _fastvideo_args, _default_request
    _generator = None
    _fastvideo_args = None
    _default_request = None
