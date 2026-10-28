# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_MEDIA_REFERENCE_MAX_BYTES = 2 * 1024**3


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "outputs/"
    served_model_name: str | None = None
    media_reference_max_bytes: int = DEFAULT_MEDIA_REFERENCE_MAX_BYTES
    # Directories whose files HTTP callers may reference by local path. Empty rejects local paths.
    media_roots: list[str] = field(default_factory=list)
    # Allows media URLs that resolve to loopback, private, or link-local addresses.
    allow_private_media_urls: bool = False

    def __post_init__(self) -> None:
        if self.media_reference_max_bytes <= 0:
            raise ValueError("server.media_reference_max_bytes must be positive")
        if any(not isinstance(root, str) or not root.strip() for root in self.media_roots):
            raise ValueError("server.media_roots entries must be non-empty paths")


@dataclass
class ParallelismConfig:
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # positive process-group timeout in seconds


@dataclass
class OffloadConfig:
    dit: bool = True
    dit_layerwise: bool = True
    text_encoder: bool = True
    image_encoder: bool = True
    vae: bool = True
    pin_cpu_memory: bool = True
    # Not a CPU offload: loads each heavy component on first use and frees it
    # after the last stage that needs it, so peak memory is the largest
    # overlapping set rather than the sum. Grouped here because it is the same
    # decision the offload knobs answer, which is how much of the model has to
    # be resident at once. ``None`` auto-enables on unified-memory devices.
    lazy_module_load: bool | None = None


@dataclass
class CompileConfig:
    """Typed ``torch.compile`` configuration.

    ``backend``/``fullgraph``/``mode``/``dynamic`` are the four most
    common ``torch.compile`` knobs. ``extras`` holds any remaining
    ``torch.compile`` kwargs (e.g. ``options``, ``disable``).

    The ``enabled`` switch covers the DiT transformer path (including
    ``transformer_2`` and the LTX-2 stage-2 ``transformer_refine``).
    Per-component flags below are independent overlays — set to ``True``
    to compile that component, ``None`` to leave it eager. Each
    ``*_kwargs`` dict overrides the master ``backend``/``fullgraph``/
    ``mode``/``dynamic``/``extras`` for that component when non-empty;
    leaving it empty inherits the master kwargs.
    """

    enabled: bool = False
    backend: str | None = None
    fullgraph: bool | None = None
    mode: str | None = None
    dynamic: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    text_encoder_enabled: bool | None = None
    vae_enabled: bool | None = None
    audio_vae_enabled: bool | None = None

    dit_kwargs: dict[str, Any] = field(default_factory=dict)
    text_encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    vae_kwargs: dict[str, Any] = field(default_factory=dict)
    audio_vae_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizationConfig:
    text_encoder_quant: str | None = None
    transformer_quant: str | None = None
    # Forwarded onto ``NVFP4Config`` when ``transformer_quant`` is ``NVFP4``.
    # ``h3_dit`` selects the packed MiniMax-H3 attention+FFN export.
    # ``h3_dit_ffn`` selects a packed FFN-only export (attention stays dense).
    # ``h3_dit_vsa`` is ``h3_dit`` plus the VSA compression gates.
    layer_profile: str | None = None
    # NVFP4 can purge the original dense weights after conversion. Keep them
    # when inference will later merge a transformer LoRA adapter.
    transformer_retain_original_weights: bool | None = None


@dataclass
class EngineConfig:
    num_gpus: int = 1
    execution_backend: Literal["mp", "uni", "ray", "external_launcher"] = "mp"
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
    lora_nickname: str = "default"
    lora_strength: float = 1.0
    override_pipeline_cls_name: str | None = None
    override_transformer_cls_name: str | None = None


@dataclass
class PipelineSelection:
    workload_type: Literal["t2v", "i2v", "t2i", "i2i", "v2a", "t2a"] | None = None
    preset: str | None = None
    preset_version: int | None = None
    components: ComponentConfig = field(default_factory=ComponentConfig)
    vae_tiling: bool | None = None
    """Tile-based VAE decode. ``None`` keeps the model's default."""
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
    pose_video_path: str | list[str] | None = None
    face_video_path: str | list[str] | None = None
    background_video_path: str | list[str] | None = None
    mask_video_path: str | list[str] | None = None
    animate_mode: str | None = None
    audio_path: str | list[str] | None = None
    pil_image: Any | None = None
    last_image: Any | None = None
    references: list[Any] | None = None
    latents: Any | None = None
    audio_latents: Any | None = None
    pose: str | None = None
    mouse_cond: Any | None = None
    keyboard_cond: Any | None = None
    grid_sizes: Any | None = None
    c2ws_plucker_emb: Any | None = None
    action_path: str | None = None


@dataclass
class SamplingConfig:
    num_videos_per_prompt: int = 1
    seed: int = 1024
    max_sequence_length: int | None = None
    num_frames: int = 125
    height: int = 720
    width: int = 1280
    height_sr: int = 1072
    width_sr: int = 1920
    fps: int = 24
    audio_start_in_s: float | None = None
    audio_end_in_s: float | None = None
    num_inference_steps: int = 50
    num_inference_steps_sr: int = 50
    guidance_scale: float = 1.0
    batch_cfg: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    use_embedded_guidance: bool | None = None
    boundary_ratio: float | None = None
    sigmas: list[float] | None = None

    # Helios autoregressive spatial-pyramid sampling.
    pyramid_num_inference_steps_list: list[int] | None = None
    history_sizes: list[int] | None = None
    num_latent_frames_per_chunk: int = 9
    keep_first_frame: bool = True
    is_skip_first_chunk: bool = False
    use_zero_init: bool = True
    zero_steps: int = 1
    is_amplify_first_chunk: bool = False


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
    return_samples: bool = field(default=False, kw_only=True)


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
    enabled: bool = False
    provider: Literal["cerebras", "groq"] = "cerebras"
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
    prompt: PromptEnhancerConfig = field(default_factory=PromptEnhancerConfig)
    safety: PromptSafetyConfig = field(default_factory=PromptSafetyConfig)


@dataclass
class LoRAAdapterConfig:
    """One community LoRA that a server may apply to a request.

    ``id`` is the only name a client sees. ``path`` is the local file or directory, or the Hugging Face repo id,
    that ``VideoGenerator.set_lora_adapter`` loads. Strength 0 is exactly the base model.
    """
    id: str
    path: str
    default_strength: float = 0.8
    min_strength: float = 0.0
    max_strength: float = 2.0
    # Optional metadata that model_info passes through to clients.
    description: str = ""
    license: str = ""
    source_url: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("lora.adapters[].id must not be empty")
        if not self.path.strip():
            raise ValueError(f"LoRA adapter {self.id!r} has an empty path")
        if not all(
                math.isfinite(strength) for strength in (self.min_strength, self.default_strength, self.max_strength)):
            raise ValueError(f"LoRA adapter {self.id!r} has a non-finite strength")
        if not self.min_strength <= self.default_strength <= self.max_strength:
            raise ValueError(f"LoRA adapter {self.id!r} needs min_strength <= default_strength <= "
                             f"max_strength, got {self.min_strength}, {self.default_strength}, {self.max_strength}")


@dataclass
class ServeLoRAConfig:
    """The LoRAs a request to the OpenAI server may select.

    No adapter loads at startup: the transformer gains LoRA layers when the first request selects one.
    """
    adapters: list[LoRAAdapterConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [adapter.id for adapter in self.adapters]
        duplicates = sorted({adapter_id for adapter_id in ids if ids.count(adapter_id) > 1})
        if duplicates:
            raise ValueError(f"lora.adapters repeats id(s): {', '.join(duplicates)}")


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
    lora: ServeLoRAConfig | None = None

    def __post_init__(self) -> None:
        if not self.lora or not self.lora.adapters:
            return
        if self.generator.pipeline.components.lora_path:
            raise ValueError("`lora.adapters` cannot be combined with a startup adapter "
                             "(generator.pipeline.components.lora_path); use one or the other.")
        if self.streaming is not None:
            raise ValueError("`lora.adapters` applies to the OpenAI server and cannot be combined with `streaming:`.")


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
    "LoRAAdapterConfig",
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
    "ServeLoRAConfig",
    "ServerConfig",
    "StreamingConfig",
    "WarmupConfig",
]
