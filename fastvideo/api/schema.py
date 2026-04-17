# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "outputs/"


@dataclass
class ParallelismConfig:
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None


@dataclass
class OffloadConfig:
    dit: bool = True
    dit_layerwise: bool = True
    text_encoder: bool = True
    image_encoder: bool = True
    vae: bool = True
    pin_cpu_memory: bool = True


@dataclass
class CompileConfig:
    enabled: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizationConfig:
    text_encoder_quant: str | None = None
    transformer_quant: str | None = None


@dataclass
class EngineConfig:
    num_gpus: int = 1
    execution_backend: Literal["mp", "ray"] = "mp"
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    enable_stage_verification: bool = True
    use_fsdp_inference: bool = False
    disable_autocast: bool = False
    quantization: QuantizationConfig | None = None


@dataclass
class ComponentConfig:
    config_root: str | None = None
    pipeline_config_path: str | None = None
    text_encoder_weights: str | None = None
    transformer_weights: str | None = None
    transformer_2_weights: str | None = None
    vae_weights: str | None = None
    upsampler_weights: str | None = None
    lora_path: str | None = None
    override_pipeline_cls_name: str | None = None
    override_transformer_cls_name: str | None = None


@dataclass
class PipelineSelection:
    workload_type: Literal["t2v", "i2v", "t2i", "i2i"] | None = None
    preset: str | None = None
    preset_version: int | None = None
    components: ComponentConfig = field(default_factory=ComponentConfig)
    preset_overrides: dict[str, Any] = field(default_factory=dict)
    experimental: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    model_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    engine: EngineConfig = field(default_factory=EngineConfig)
    pipeline: PipelineSelection = field(default_factory=PipelineSelection)


@dataclass
class InputConfig:
    prompt_path: str | None = None
    image_path: str | list[str] | None = None
    video_path: str | list[str] | None = None
    pil_image: Any | None = None
    pose: str | None = None
    mouse_cond: Any | None = None
    keyboard_cond: Any | None = None
    grid_sizes: Any | None = None
    c2ws_plucker_emb: Any | None = None
    refine_from: str | None = None
    stage1_video: Any | None = None


@dataclass
class SamplingConfig:
    num_videos_per_prompt: int = 1
    seed: int = 1024
    num_frames: int = 125
    height: int = 720
    width: int = 1280
    height_sr: int = 1072
    width_sr: int = 1920
    fps: int = 24
    num_inference_steps: int = 50
    num_inference_steps_sr: int = 50
    guidance_scale: float = 1.0
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    boundary_ratio: float | None = None
    sigmas: list[float] | None = None


@dataclass
class RequestRuntimeConfig:
    enable_teacache: bool = False
    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False


@dataclass
class OutputConfig:
    output_path: str = "outputs/"
    output_video_name: str | None = None
    save_video: bool = True
    return_frames: bool = True
    return_state: bool = False


@dataclass
class ContinuationState:
    kind: str
    payload: dict[str, Any]


@dataclass
class PlannedStage:
    name: str
    kind: str
    source: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationPlan:
    stages: list[PlannedStage]
    final_stage: str | None = None


@dataclass
class GenerationRequest:
    prompt: str | list[str] | None = None
    negative_prompt: str | None = None
    inputs: InputConfig = field(default_factory=InputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RequestRuntimeConfig = field(default_factory=RequestRuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    stage_overrides: dict[str, Any] = field(default_factory=dict)
    state: ContinuationState | None = None
    plan: GenerationPlan | None = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    generator: GeneratorConfig
    request: GenerationRequest


@dataclass
class WarmupConfig:
    enabled: bool = True
    prompt: str = ("A cinematic drone shot over coastal cliffs at sunrise, "
                   "golden light, gentle ocean waves, ultra detailed")
    timeout_seconds: int = 2400


@dataclass
class GpuPoolConfig:
    num_workers: int | None = None
    enable_audio_reencode: bool = True
    conditioning_num_frames: int = 9
    conditioning_end_offset: int = 0


@dataclass
class PromptEnhancerConfig:
    provider: Literal["cerebras_ifm", "cerebras", "groq"] = "cerebras_ifm"
    model: str = "gpt-oss-120b"
    timeout_ms: int = 20000
    system_prompt_dir: str | None = None


@dataclass
class PromptSafetyConfig:
    enabled: bool = False
    classifier_path: str | None = None


@dataclass
class StreamingConfig:
    session_timeout_seconds: int = 300
    generation_segment_cap: int = 6
    stream_mode: Literal["av_fmp4", "legacy_jpeg"] = "av_fmp4"
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    pool: GpuPoolConfig = field(default_factory=GpuPoolConfig)
    prompt: PromptEnhancerConfig | None = None
    safety: PromptSafetyConfig | None = None


@dataclass
class ServeConfig:
    """Typed serve config loaded from ``fastvideo serve --config``.

    ``default_request`` is a full :class:`GenerationRequest` — the same type
    clients POST to ``/v1/videos``. At request time the server merges it into
    the incoming body as the operator-pinned baseline.

    Important nuance: only fields the operator **explicitly wrote** in the
    serve YAML/JSON count as defaults. Although the in-memory object is
    fully populated (schema defaults fill every unset field), the merge
    walks ``_fastvideo_explicit_paths`` — populated during parse — so
    unset fields are *not* forced onto requests. Per-request precedence:

        body (client-explicit) > default_request (operator-explicit)
                               > hardcoded fallback (e.g. ``fps=24``)

    See :func:`fastvideo.api.compat.explicit_request_updates` for the
    projection and ``entrypoints/openai/video_api.py::_build_generation_kwargs``
    for the merge.
    """
    generator: GeneratorConfig
    server: ServerConfig = field(default_factory=ServerConfig)
    default_request: GenerationRequest = field(default_factory=GenerationRequest)
    streaming: StreamingConfig | None = None


__all__ = [
    "CompileConfig",
    "ComponentConfig",
    "ContinuationState",
    "EngineConfig",
    "GenerationPlan",
    "GenerationRequest",
    "GeneratorConfig",
    "GpuPoolConfig",
    "InputConfig",
    "OffloadConfig",
    "OutputConfig",
    "ParallelismConfig",
    "PipelineSelection",
    "PlannedStage",
    "PromptEnhancerConfig",
    "PromptSafetyConfig",
    "QuantizationConfig",
    "RequestRuntimeConfig",
    "RunConfig",
    "SamplingConfig",
    "ServeConfig",
    "ServerConfig",
    "StreamingConfig",
    "WarmupConfig",
]
