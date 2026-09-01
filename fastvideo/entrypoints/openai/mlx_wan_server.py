# SPDX-License-Identifier: Apache-2.0
"""Serve native FastMetal (Wan2.1) MLX through the shared video-job API and playground."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import uvicorn
import yaml

from fastvideo.api.compat import explicit_request_updates, normalize_generation_request
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest

MODEL: Literal["FastVideo/FastMetal-1.3B-QAD"] = "FastVideo/FastMetal-1.3B-QAD"
# The DMD-distilled step ladder the validated 1.3B recipe uses (fixed count,
# same reason H3 MLX serving pins num_inference_steps to its own ladder size).
_DMD_STEP_COUNT = 3


class MLXWanGeneratorConfig(BaseModel):
    """Where the two FastMetal checkpoint halves live on disk."""
    model_config = ConfigDict(extra="forbid")
    model_path: Literal["FastVideo/FastMetal-1.3B-QAD"] = MODEL
    model_root: str
    mlx_checkpoint: str


class MLXWanServerConfig(BaseModel):
    """Host/port/output shape for an MLX serve YAML's ``server:`` block.

    Generic across MLX-served models -- if a second MLX server config needs
    the same shape, promote this (and its H3 counterpart) to one shared
    module instead of a third copy.
    """
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    output_dir: str = "outputs/mlx_wan"
    served_model_name: str = Field(default="fastwan", min_length=1)


class MLXWanServeConfig(BaseModel):
    """Top-level ``mlx_wan_*.yaml`` shape read by --config."""
    model_config = ConfigDict(extra="forbid")
    runtime: Literal["mlx"]
    generator: MLXWanGeneratorConfig
    server: MLXWanServerConfig = Field(default_factory=MLXWanServerConfig)
    default_request: dict[str, Any]


def validate_wan_video_request(request: VideoGenerationRequest) -> None:
    """Reject unsupported inputs before fetching media or creating a job."""
    allowed = {
        "model",
        "prompt",
        "seed",
        "size",
        "width",
        "height",
        "fps",
        "num_frames",
        "seconds",
        "video_params",
        "task",
        "guidance_scale",
        "num_inference_steps",
    }
    unsupported = request.model_fields_set - allowed
    if unsupported:
        raise ValueError("Wan MLX serving does not support: " + ", ".join(sorted(unsupported)))
    if request.task not in (None, "t2v"):
        raise ValueError("Wan MLX serving supports task=t2v only.")
    if request.guidance_scale not in (None, 1.0):
        raise ValueError("FastMetal MLX is DMD-distilled and requires guidance_scale=1.")
    if request.num_inference_steps not in (None, _DMD_STEP_COUNT):
        raise ValueError(f"Wan MLX serving uses a fixed {_DMD_STEP_COUNT}-step DMD ladder; "
                         f"num_inference_steps must be {_DMD_STEP_COUNT}.")
    if request.seed is not None and not 0 <= request.seed <= 2**32 - 1:
        raise ValueError("Wan MLX seed must be between 0 and 4294967295.")


class MLXWanGenerator:
    """Keep one FastMetal pipeline on one MLX thread across requests."""

    def __init__(self, config: MLXWanGeneratorConfig) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-mlx")
        try:
            self._pipeline = self._worker.submit(self._load, config).result()
        except BaseException:
            self._worker.shutdown(wait=True)
            raise

    @staticmethod
    def _load(config: MLXWanGeneratorConfig):
        """Load the pipeline; must run on the MLX worker thread."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Wan MLX serving requires an Apple Silicon Mac.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Install ffmpeg before starting the Wan MLX server.")
        from fastvideo.mlx_runtime.wan_pipeline import MLXWanPipeline

        return MLXWanPipeline(
            model_root=Path(config.model_root).expanduser(),
            mlx_checkpoint=Path(config.mlx_checkpoint).expanduser(),
        )

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Run one generation on the MLX worker thread; block until it finishes."""
        return self._worker.submit(self._generate, request).result()

    def _generate(self, request: GenerationRequest) -> dict[str, Any]:
        """The actual pipeline call; must run on the MLX worker thread."""
        started = time.perf_counter()
        result = self._pipeline.generate(
            request.prompt,
            output_path=request.output.output_path,
            width=request.sampling.width,
            height=request.sampling.height,
            num_frames=request.sampling.num_frames,
            seed=request.sampling.seed,
            fps=request.sampling.fps,
        )
        return {"video_path": str(result.video_path), "generation_time": time.perf_counter() - started}

    def shutdown(self) -> None:
        """Release the pipeline and stop the MLX worker thread."""

        def release():
            self._pipeline = None
            from fastvideo.mlx_runtime.memory import cleanup_mlx

            cleanup_mlx()

        try:
            self._worker.submit(release).result()
        finally:
            self._worker.shutdown(wait=True)


def load_config(path: str) -> MLXWanServeConfig:
    """Parse a Wan MLX serve YAML into its typed config."""
    with open(path, encoding="utf-8") as source:
        return MLXWanServeConfig.model_validate(yaml.safe_load(source))


def create_mlx_wan_app(config: MLXWanServeConfig):
    """Build the FastAPI app for a validated Wan MLX serve config."""
    request = normalize_generation_request(config.default_request)
    explicit = explicit_request_updates(request)
    supported = {"width", "height", "num_frames", "fps", "seed", "num_inference_steps", "guidance_scale"}
    if set(explicit) - supported:
        raise ValueError("Wan MLX default_request contains unsupported fields: " +
                         ", ".join(sorted(set(explicit) - supported)))
    required = {"width", "height", "num_frames", "fps"}
    if required - set(explicit):
        raise ValueError("Wan MLX default_request must set: " + ", ".join(sorted(required - set(explicit))))
    validate_wan_video_request(VideoGenerationRequest(prompt="validate config", **explicit))
    # Transport admission uses the registered Wan family, not CUDA engine options.
    args = SimpleNamespace(model_path=MODEL,
                           lora_path=None,
                           lora_nickname="default",
                           lora_strength=1.0,
                           override_pipeline_cls_name=None)
    from fastvideo.entrypoints.openai.request_adapter import build_generation_request

    build_generation_request("config-check",
                             VideoGenerationRequest(prompt="validate config"),
                             args,
                             served_model_name=config.server.served_model_name,
                             output_dir=config.server.output_dir,
                             default_request=request)
    return create_app(
        args,
        config.server.output_dir,
        request,
        config.server.served_model_name,
        generator_factory=lambda: MLXWanGenerator(config.generator),
        video_request_validator=validate_wan_video_request,
        runtime="mlx",
    )


def main() -> None:
    """CLI entrypoint: python -m fastvideo.entrypoints.openai.mlx_wan_server --config ..."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        required=True,
                        help="Wan MLX serving YAML; paths are relative to the working directory")
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(create_mlx_wan_app(config), host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
