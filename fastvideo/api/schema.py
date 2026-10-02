# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

# Field metadata key that holds a field's name in the flat keyword API: the keyword arguments of
# ``VideoGenerator.from_pretrained`` and the fields of ``FastVideoArgs``. ``fastvideo.api.compat`` converts between
# the nested config and the flat keywords through this name, so a field that declares it needs no other mapping.
FLAT_NAME = "flat_name"


def flat_field(flat_name: str, default: Any = MISSING, *, default_factory: Any = MISSING) -> Any:
    """Declare a dataclass field whose name in the flat keyword API is ``flat_name``."""
    return field(default=default, default_factory=default_factory, metadata={FLAT_NAME: flat_name})


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "outputs/"
    served_model_name: str | None = None


@dataclass
class ParallelismConfig:
    tp_size: int = flat_field("tp_size", -1)
    sp_size: int = flat_field("sp_size", -1)
    hsdp_replicate_dim: int = flat_field("hsdp_replicate_dim", 1)
    hsdp_shard_dim: int = flat_field("hsdp_shard_dim", -1)
    dist_timeout: int | None = flat_field("dist_timeout", None)


@dataclass
class OffloadConfig:
    dit: bool = flat_field("dit_cpu_offload", True)
    dit_layerwise: bool = flat_field("dit_layerwise_offload", True)
    text_encoder: bool = flat_field("text_encoder_cpu_offload", True)
    image_encoder: bool = flat_field("image_encoder_cpu_offload", True)
    vae: bool = flat_field("vae_cpu_offload", True)
    pin_cpu_memory: bool = flat_field("pin_cpu_memory", True)
    # Not a CPU offload: loads each heavy component on first use and frees it
    # after the last stage that needs it, so peak memory is the largest
    # overlapping set rather than the sum. Grouped here because it is the same
    # decision the offload knobs answer, which is how much of the model has to
    # be resident at once. ``None`` auto-enables on unified-memory devices.
    lazy_module_load: bool | None = flat_field("lazy_module_load", None)


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

    enabled: bool = flat_field("enable_torch_compile", False)
    backend: str | None = None
    fullgraph: bool | None = None
    mode: str | None = None
    dynamic: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    text_encoder_enabled: bool | None = flat_field("enable_torch_compile_text_encoder", None)
    vae_enabled: bool | None = flat_field("enable_torch_compile_vae", None)
    audio_vae_enabled: bool | None = flat_field("enable_torch_compile_audio_vae", None)
    regional: bool | None = flat_field("inference_torch_compile", None)
    """Regional fullgraph compile of each DiT transformer block, independent of ``enabled``. The loader applies it
    with fixed options and ignores the kwargs below. ``None`` falls back to ``FASTVIDEO_INFERENCE_TORCH_COMPILE``."""

    dit_kwargs: dict[str, Any] = flat_field("torch_compile_kwargs_dit", default_factory=dict)
    text_encoder_kwargs: dict[str, Any] = flat_field("torch_compile_kwargs_text_encoder", default_factory=dict)
    vae_kwargs: dict[str, Any] = flat_field("torch_compile_kwargs_vae", default_factory=dict)
    audio_vae_kwargs: dict[str, Any] = flat_field("torch_compile_kwargs_audio_vae", default_factory=dict)


@dataclass
class AttentionConfig:
    backend: str | None = flat_field("attention_backend", None)
    """Default attention backend request, such as ``FLASH_ATTN`` or ``TORCH_SDPA``, applied per component at load
    time. ``None`` falls back to ``FASTVIDEO_ATTENTION_BACKEND``, then per-layer defaults, then automatic selection."""
    vsa_sparsity: float | None = flat_field("VSA_sparsity", None)
    """Video sparse attention (VSA) sparsity at inference. ``None`` keeps the default of 0.0."""
    vsa_tile_size: int | None = flat_field("VSA_tile_size", None)
    """VSA tile size in tokens, 256 or 64; 64 runs the native Triton block-sparse path. ``None`` keeps 256."""
    moba_config_path: str | None = flat_field("moba_config_path", None)
    """Path to a JSON config for V-MoBA attention."""


Precision = Literal["fp32", "fp16", "bf16"]


@dataclass
class PrecisionConfig:
    """Numeric precision of each model component. ``None`` keeps the model's default."""

    dit: Precision | None = flat_field("dit_precision", None)
    vae: Precision | None = flat_field("vae_precision", None)
    vae_decode: Precision | None = flat_field("vae_decode_precision", None)
    image_encoder: Precision | None = flat_field("image_encoder_precision", None)
    text_encoders: list[Precision] | None = flat_field("text_encoder_precisions", None)
    """One precision per text encoder, in the model's text encoder order."""


@dataclass
class QuantizationConfig:
    text_encoder_quant: str | None = flat_field("override_text_encoder_quant", None)
    transformer_quant: str | None = None


@dataclass
class EngineConfig:
    num_gpus: int = flat_field("num_gpus", 1)
    execution_backend: Literal["mp", "ray"] = flat_field("distributed_executor_backend", "mp")
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    enable_stage_verification: bool = flat_field("enable_stage_verification", True)
    use_fsdp_inference: bool = flat_field("use_fsdp_inference", False)
    disable_autocast: bool = flat_field("disable_autocast", False)
    quantization: QuantizationConfig | None = None


@dataclass
class ComponentConfig:
    config_root: str | None = flat_field("config_model_path", None)
    pipeline_config_path: str | None = flat_field("pipeline_config", None)
    text_encoder_weights: str | None = flat_field("override_text_encoder_safetensors", None)
    transformer_weights: str | None = flat_field("init_weights_from_safetensors", None)
    transformer_2_weights: str | None = flat_field("init_weights_from_safetensors_2", None)
    vae_weights: str | None = None
    upsampler_weights: str | None = flat_field("ltx2_refine_upsampler_path", None)
    lora_path: str | None = flat_field("lora_path", None)
    lora_nickname: str = flat_field("lora_nickname", "default")
    lora_strength: float = flat_field("lora_strength", 1.0)
    lora_target_modules: list[str] | None = flat_field("lora_target_modules", None)
    """Module name substrings that restrict LoRA injection, such as ``["q_proj", "v_proj"]``. ``None`` adapts the
    default modules."""
    override_pipeline_cls_name: str | None = flat_field("override_pipeline_cls_name", None)
    override_transformer_cls_name: str | None = flat_field("override_transformer_cls_name", None)


@dataclass
class LTX2RefineOptions:
    """Stage-2 refine assets that ``preset_overrides.refine`` and ``components`` do not cover."""

    transformer_path: str | None = flat_field("ltx2_refine_transformer_path", None)
    lora_path: str | None = flat_field("ltx2_refine_lora_path", None)
    """LoRA applied to the refine transformer only. ``None`` uses the checkpoint's default
    (``fastvideo_refine_lora_path`` in ``model_index.json``); an empty string disables the refine LoRA."""
    noise_path: str | None = flat_field("ltx2_refine_noise_path", None)
    audio_noise_path: str | None = flat_field("ltx2_refine_audio_noise_path", None)


@dataclass
class LTX2Options:
    """LTX-2 settings. ``None`` keeps the model's default."""

    vae_spatial_tile_size_in_pixels: int | None = flat_field("ltx2_vae_spatial_tile_size_in_pixels", None)
    vae_spatial_tile_overlap_in_pixels: int | None = flat_field("ltx2_vae_spatial_tile_overlap_in_pixels", None)
    vae_temporal_tile_size_in_frames: int | None = flat_field("ltx2_vae_temporal_tile_size_in_frames", None)
    vae_temporal_tile_overlap_in_frames: int | None = flat_field("ltx2_vae_temporal_tile_overlap_in_frames", None)
    initial_latent_path: str | None = flat_field("ltx2_initial_latent_path", None)
    """Path to load or save a precomputed initial video latent."""
    audio_latent_path: str | None = flat_field("ltx2_audio_latent_path", None)
    """Path to load or save a precomputed initial audio latent."""
    legacy_native_noise_order: bool | None = flat_field("ltx2_legacy_native_noise_order", None)
    """Draw latent noise in the legacy native order, which earlier SSIM references use."""
    use_distilled_sigmas: bool | None = flat_field("ltx2_use_distilled_sigmas", None)
    """Use the distilled sigma schedule when the checkpoint provides one."""
    refine: LTX2RefineOptions = field(default_factory=LTX2RefineOptions)


@dataclass
class MiniMaxH3Options:
    """MiniMax-H3 settings. ``None`` keeps the model's default."""

    sequential_load: bool | None = flat_field("h3_sequential_load", None)
    """Encode with Qwen3-VL, release that encoder, then load the DiT and VAEs. ``None`` enables it on
    unified-memory devices only."""
    video_decode_backend: Literal["h3-vae", "taeh3"] | None = flat_field("video_decode_backend", None)
    """``h3-vae`` is the full VAE; ``taeh3`` is a fast approximate preview decoder."""
    taeh3_checkpoint: str | None = flat_field("taeh3_checkpoint", None)
    """Local ``taeh3.safetensors`` path. ``None`` downloads the pinned upstream weights."""
    taeh3_chunk_size: int | None = flat_field("taeh3_chunk_size", None)
    """TAEH3 latent frames per execution chunk."""
    vae_parallel_decode: bool | None = flat_field("vae_parallel_decode", None)
    """Spread VAE decode chunks across the sequence-parallel ranks. ``None`` falls back to
    ``FASTVIDEO_VAE_PARALLEL_DECODE``."""
    vae_parallel_encode: bool | None = flat_field("vae_parallel_encode", None)
    """Spread reference-video VAE encode clips across the sequence-parallel ranks. ``None`` falls back to
    ``FASTVIDEO_VAE_PARALLEL_ENCODE``."""
    vae_parallel_decode_strategy: Literal["gather", "all_gather"] | None = flat_field(
        "vae_parallel_decode_strategy", None)
    """Collective that moves decoded chunks. ``None`` falls back to ``FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY``,
    then ``gather``."""


@dataclass
class LongCatOptions:
    """LongCat block sparse attention (BSA) settings. ``None`` keeps the model's default."""

    enable_bsa: bool | None = flat_field("enable_bsa", None)
    bsa_sparsity: float | None = flat_field("bsa_sparsity", None)
    bsa_cdf_threshold: float | None = flat_field("bsa_cdf_threshold", None)
    bsa_chunk_q: list[int] | None = flat_field("bsa_chunk_q", None)
    """Query chunk shape as ``[T, H, W]``."""
    bsa_chunk_k: list[int] | None = flat_field("bsa_chunk_k", None)
    """Key chunk shape as ``[T, H, W]``."""


@dataclass
class PipelineSelection:
    workload_type: Literal["t2v", "i2v", "t2i", "i2i", "v2a", "t2a"] | None = flat_field("workload_type", None)
    preset: str | None = None
    preset_version: int | None = None
    components: ComponentConfig = field(default_factory=ComponentConfig)
    vae_tiling: bool | None = flat_field("ltx2_vae_tiling", None)
    """Tile-based VAE decode. ``None`` keeps the model's default."""
    vae_sp: bool | None = flat_field("vae_sp", None)
    """VAE spatial parallelism across ranks; requires ``vae_tiling``. ``None`` keeps the model's default."""
    flow_shift: float | None = flat_field("flow_shift", None)
    """Flow-matching scheduler shift. ``None`` keeps the model's default."""
    embedded_cfg_scale: float | None = flat_field("embedded_cfg_scale", None)
    """Guidance scale that guidance-distilled models take as a DiT input. ``None`` keeps the model's default."""
    dmd_denoising_steps: list[int] | None = flat_field("dmd_denoising_steps", None)
    """Timesteps of a few-step distilled (DMD) sampler. ``None`` keeps the model's default."""
    dit: dict[str, Any] = field(default_factory=dict)
    """Overrides for fields of the model's DiT config, such as ``prefix``."""
    vae: dict[str, Any] = field(default_factory=dict)
    """Overrides for fields of the model's VAE config, such as ``load_encoder`` or ``use_tiling``."""
    ltx2: LTX2Options = field(default_factory=LTX2Options)
    minimax_h3: MiniMaxH3Options = field(default_factory=MiniMaxH3Options)
    longcat: LongCatOptions = field(default_factory=LongCatOptions)
    preset_overrides: dict[str, Any] = field(default_factory=dict)
    experimental: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    model_path: str = flat_field("model_path")
    revision: str | None = flat_field("revision", None)
    trust_remote_code: bool = flat_field("trust_remote_code", False)
    engine: EngineConfig = field(default_factory=EngineConfig)
    pipeline: PipelineSelection = field(default_factory=PipelineSelection)


@dataclass
class InputConfig:
    prompt_path: str | None = None
    image_path: str | list[str] | None = None
    video_path: str | list[str] | None = None
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
    refine_from: str | None = None
    stage1_video: Any | None = None


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
    num_inference_steps: int = 50
    num_inference_steps_sr: int = 50
    guidance_scale: float = 1.0
    batch_cfg: bool = False
    guidance_scale_2: float | None = None
    cfg_normalization: bool = False
    cfg_truncation: float | None = 1.0
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    use_embedded_guidance: bool | None = None
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
    "AttentionConfig",
    "CompileConfig",
    "ComponentConfig",
    "ContinuationState",
    "EngineConfig",
    "GenerationPlan",
    "GenerationRequest",
    "GeneratorConfig",
    "GpuPoolConfig",
    "InputConfig",
    "LTX2Options",
    "LTX2RefineOptions",
    "LongCatOptions",
    "MiniMaxH3Options",
    "OffloadConfig",
    "OutputConfig",
    "ParallelismConfig",
    "PipelineSelection",
    "PlannedStage",
    "PrecisionConfig",
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
