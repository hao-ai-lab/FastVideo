# SPDX-License-Identifier: Apache-2.0
"""Serve native FastMetal Wan2.1 and Wan2.2 MLX through the shared video-job API."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from fastvideo.api.compat import explicit_request_updates, normalize_generation_request
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.api_server import create_app
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.mlx_common import MLXServerConfig, MLXWorkerGenerator, run_mlx_server
from fastvideo.entrypoints.openai.request_adapter import resolve_video_sampling_fields
from fastvideo.mlx_runtime.wan_helpers import WAN_DMD_STEPS, WAN_TEMPORAL_COMPRESSION, plan_wan_generation

# The adapter's own fps fallback, used when a caller validates a request outside
# a configured server (create_mlx_wan_app binds the served fps instead).
_FALLBACK_FPS = 24

# Maps a served model id to the pipeline class that generates it: 1.3B/14B
# share Wan2.1's architecture (MLXWanPipeline), 5B is Wan2.2-TI2V instead
# (MLXWan22Pipeline, 48-channel latents, a different DiT and sampler).
_PIPELINE_CLASS_NAMES = {
    "FastVideo/FastMetal-1.3B-QAD": "MLXWanPipeline",
    "FastVideo/FastMetal-14B-QAD": "MLXWanPipeline",
    "FastVideo/FastMetal-5B-QAD": "MLXWan22Pipeline",
}


class MLXWanGeneratorConfig(BaseModel):
    """Where the two FastMetal checkpoint halves live on disk."""
    model_config = ConfigDict(extra="forbid")
    model_path: Literal["FastVideo/FastMetal-1.3B-QAD", "FastVideo/FastMetal-14B-QAD", "FastVideo/FastMetal-5B-QAD"]
    model_root: str
    mlx_checkpoint: str
    prompt_cache_dir: str | None = "outputs/wan_prompt_cache"


class MLXWanServerConfig(MLXServerConfig):
    """Wan output location and model alias."""
    output_dir: str = "outputs/mlx_wan"
    served_model_name: str = Field(default="fastwan", min_length=1)


class MLXWanServeConfig(BaseModel):
    """Top-level ``mlx_wan_*.yaml`` shape read by --config."""
    model_config = ConfigDict(extra="forbid")
    runtime: Literal["mlx"]
    generator: MLXWanGeneratorConfig
    server: MLXWanServerConfig = Field(default_factory=MLXWanServerConfig)
    default_request: dict[str, Any]


def _aligned_num_frames(num_frames: int) -> int:
    """Round up to the next frame count Wan accepts (1 modulo the VAE temporal stride)."""
    remainder = (num_frames - 1) % WAN_TEMPORAL_COMPRESSION
    if remainder == 0:
        return num_frames
    return num_frames + (WAN_TEMPORAL_COMPRESSION - remainder)


def validate_wan_video_request(
    request: VideoGenerationRequest,
    *,
    default_fps: int = _FALLBACK_FPS,
    default_request: GenerationRequest | None = None,
    model_path: str = "FastVideo/FastMetal-1.3B-QAD",
) -> None:
    """Reject unsupported inputs before fetching media or creating a job.

    Also normalizes the two request shapes the shared adapter would otherwise
    reject after the job is already admitted: an explicit ``task`` and a
    ``seconds`` duration that does not land on Wan's frame grid.
    """
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
    if request.task is not None:
        request.task = None
        request.model_fields_set.discard("task")
    if request.guidance_scale not in (None, 1.0):
        raise ValueError("FastMetal MLX is DMD-distilled and requires guidance_scale=1.")
    if request.num_inference_steps not in (None, len(WAN_DMD_STEPS)):
        raise ValueError(f"Wan MLX serving uses a fixed {len(WAN_DMD_STEPS)}-step DMD ladder; "
                         f"num_inference_steps must be {len(WAN_DMD_STEPS)}.")
    if request.seed is not None and not 0 <= request.seed <= 2**32 - 1:
        raise ValueError("Wan MLX seed must be between 0 and 4294967295.")
    resolved = resolve_video_sampling_fields(request, default_request=default_request, default_fps=default_fps)
    body_set = request.model_fields_set
    nested_set = request.video_params.model_fields_set if request.video_params is not None else set()
    frames_explicit = ("num_frames" in body_set
                       and request.num_frames is not None) or ("video_params" in body_set and "num_frames" in nested_set
                                                               and request.video_params.num_frames is not None)
    if "seconds" in body_set and request.seconds is not None and not frames_explicit:
        request.num_frames = _aligned_num_frames(resolved["num_frames"])
        resolved["num_frames"] = request.num_frames
    wan22 = model_path == "FastVideo/FastMetal-5B-QAD"
    plan_wan_generation(
        height=resolved.get("height", 704 if wan22 else 480),
        width=resolved.get("width", 1280 if wan22 else 832),
        num_frames=resolved.get("num_frames", 81),
        wan22=wan22,
    )


class MLXWanGenerator(MLXWorkerGenerator):
    """Keep one FastMetal pipeline on one MLX thread across requests."""
    thread_name = "wan-mlx"

    @staticmethod
    def _load(config: MLXWanGeneratorConfig):
        """Load the pipeline; must run on the MLX worker thread."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Wan MLX serving requires an Apple Silicon Mac.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Install ffmpeg before starting the Wan MLX server.")
        import fastvideo.mlx_runtime.wan_pipeline as wan_pipeline_module

        pipeline_cls = getattr(wan_pipeline_module, _PIPELINE_CLASS_NAMES[config.model_path])
        return pipeline_cls(
            model_root=Path(config.model_root).expanduser(),
            mlx_checkpoint=Path(config.mlx_checkpoint).expanduser(),
            prompt_cache_dir=getattr(config, "prompt_cache_dir", "outputs/wan_prompt_cache"),
        )

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
    validate_wan_video_request(VideoGenerationRequest(prompt="validate config", **explicit),
                               model_path=config.generator.model_path)
    # Transport admission uses the registered Wan family, not CUDA engine options.
    args = SimpleNamespace(model_path=config.generator.model_path,
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
        video_request_validator=partial(validate_wan_video_request,
                                        default_request=request,
                                        model_path=config.generator.model_path),
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
    run_mlx_server(config)


if __name__ == "__main__":
    main()
