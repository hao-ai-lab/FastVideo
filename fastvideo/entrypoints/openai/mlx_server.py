# SPDX-License-Identifier: Apache-2.0
"""Serve native FastH3 MLX through the shared video-job API and playground."""

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

MODEL: Literal["FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"] = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"


class MLXGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: Literal["FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"] = MODEL
    model_root: str
    mlx_checkpoint: str
    prompt_cache_dir: str = "outputs/h3_prompt_cache"
    vae_dtype: Literal["fp32", "fp16", "bf16"] = "fp32"


class MLXServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    output_dir: str = "outputs/mlx_fasth3"
    served_model_name: str = Field(default="fasth3", min_length=1)


class MLXServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: Literal["mlx"]
    generator: MLXGeneratorConfig
    server: MLXServerConfig = Field(default_factory=MLXServerConfig)
    default_request: dict[str, Any]


def validate_mlx_video_request(request: VideoGenerationRequest) -> None:
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
        "negative_prompt",
    }
    unsupported = request.model_fields_set - allowed
    if unsupported:
        raise ValueError("H3 MLX serving does not support: " + ", ".join(sorted(unsupported)))
    if request.task not in (None, "t2va"):
        raise ValueError("H3 MLX serving supports task=t2va only.")
    if request.guidance_scale not in (None, 1.0):
        raise ValueError("FastH3 MLX requires guidance_scale=1.")
    if request.negative_prompt not in (None, ""):
        raise ValueError("FastH3 MLX does not use a negative prompt.")
    if request.num_inference_steps not in (None, 5):
        raise ValueError(
            "FastH3 MLX serving uses five sigma points and four DiT forwards; num_inference_steps must be 5.")
    if request.seed is not None and not 0 <= request.seed <= 2**32 - 1:
        raise ValueError("H3 MLX seed must be between 0 and 4294967295.")


class MLXH3Generator:
    """Keep one pipeline on one MLX thread; preserve its phase-memory policy."""

    def __init__(self, config: MLXGeneratorConfig) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-mlx")
        try:
            self._pipeline = self._worker.submit(self._load, config).result()
        except BaseException:
            self._worker.shutdown(wait=True)
            raise

    @staticmethod
    def _load(config: MLXGeneratorConfig):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("H3 MLX serving requires an Apple Silicon Mac.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Install ffmpeg before starting the H3 MLX server.")
        from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

        return MiniMaxH3MLXPipeline(
            model_root=Path(config.model_root).expanduser(),
            mlx_dit_checkpoint=Path(config.mlx_checkpoint).expanduser(),
            prompt_cache_dir=Path(config.prompt_cache_dir).expanduser(),
            vae_dtype=config.vae_dtype,
        )

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        return self._worker.submit(self._generate, request).result()

    def _generate(self, request: GenerationRequest) -> dict[str, Any]:
        started = time.perf_counter()
        result = self._pipeline.generate(
            request.prompt,
            output_path=request.output.output_path,
            width=request.sampling.width,
            height=request.sampling.height,
            num_frames=request.sampling.num_frames,
            seed=request.sampling.seed,
            num_steps=4,
        )
        # Do not retain decoded frames/waveforms or label phase peaks as total RAM.
        return {"video_path": str(result.video_path), "generation_time": time.perf_counter() - started}

    def shutdown(self) -> None:

        def release():
            self._pipeline = None
            from fastvideo.mlx_runtime.minimax_h3_pipeline import _cleanup_mlx

            _cleanup_mlx()

        try:
            self._worker.submit(release).result()
        finally:
            self._worker.shutdown(wait=True)


def load_config(path: str) -> MLXServeConfig:
    with open(path, encoding="utf-8") as source:
        return MLXServeConfig.model_validate(yaml.safe_load(source))


def create_mlx_app(config: MLXServeConfig):
    request = normalize_generation_request(config.default_request)
    explicit = explicit_request_updates(request)
    supported = {
        "width", "height", "num_frames", "fps", "seed", "num_inference_steps", "guidance_scale", "negative_prompt"
    }
    if set(explicit) - supported:
        raise ValueError("MLX default_request contains unsupported fields: " +
                         ", ".join(sorted(set(explicit) - supported)))
    required = {"width", "height", "num_frames", "fps", "seed", "num_inference_steps", "guidance_scale"}
    if required - set(explicit):
        raise ValueError("MLX default_request must set: " + ", ".join(sorted(required - set(explicit))))
    validate_mlx_video_request(VideoGenerationRequest(prompt="validate config", **explicit))
    # Transport admission uses the registered H3 family, not CUDA engine options.
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
        generator_factory=lambda: MLXH3Generator(config.generator),
        video_request_validator=validate_mlx_video_request,
        runtime="mlx",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        required=True,
                        help="H3 MLX serving YAML; paths are relative to the working directory")
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(create_mlx_app(config), host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
