# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal


class ExecutionMode(str, Enum):
    """
    Enumeration for different pipeline modes.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    INFERENCE = "inference"
    PREPROCESS = "preprocess"
    FINETUNING = "finetuning"
    DISTILLATION = "distillation"

    @classmethod
    def from_string(cls, value: str) -> ExecutionMode:
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None


class WorkloadType(str, Enum):
    """
    Enumeration for different workload types.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image
    V2A = "v2a"  # Video to Audio
    T2A = "t2a"  # Text to Audio

    @classmethod
    def from_string(cls, value: str) -> WorkloadType:
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "outputs/"
    served_model_name: str | None = None


@dataclass
class ParallelismConfig:
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None
    master_port: int | None = None
    """Port of the rendezvous that the executor opens for its workers. ``None`` picks an open port."""


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
    regional: bool | None = None
    """Regional fullgraph compile of each DiT transformer block, independent of ``enabled``. The loader applies it
    with fixed options and ignores the kwargs below. ``None`` falls back to ``FASTVIDEO_INFERENCE_TORCH_COMPILE``."""

    dit_kwargs: dict[str, Any] = field(default_factory=dict)
    text_encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    vae_kwargs: dict[str, Any] = field(default_factory=dict)
    audio_vae_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionConfig:
    backend: str | None = None
    """Default attention backend request, such as ``FLASH_ATTN`` or ``TORCH_SDPA``, applied per component at load
    time. ``None`` falls back to ``FASTVIDEO_ATTENTION_BACKEND``, then per-layer defaults, then automatic selection."""
    vsa_sparsity: float | None = None
    """Video sparse attention (VSA) sparsity at inference. ``None`` keeps the default of 0.0."""
    vsa_tile_size: int | None = None
    """VSA tile size in tokens, 256 or 64; 64 runs the native Triton block-sparse path. ``None`` keeps 256."""
    moba_config_path: str | None = None
    """Path to a JSON config for V-MoBA attention."""
    moba_config: dict[str, Any] | None = None
    """V-MoBA attention settings. Resolution loads them from ``moba_config_path`` when that path is set."""
    nvfp4_fa4: bool = False
    """FlashAttention-4 with Q and K quantized to NVFP4. Resolution exports ``FASTVIDEO_NVFP4_FA4=1`` and the
    ``CUTE_DSL_ENABLE_TVM_FFI`` default that the CuTe DSL kernels need when this is true."""


Precision = Literal["fp32", "fp16", "bf16"]


@dataclass
class PrecisionConfig:
    """Numeric precision of each model component. ``None`` keeps the model's default."""

    dit: Precision | None = None
    vae: Precision | None = None
    vae_decode: Precision | None = None
    image_encoder: Precision | None = None
    text_encoders: list[Precision] | None = None
    """One precision per text encoder, in the model's text encoder order."""


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
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
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
    lora_target_modules: list[str] | None = None
    """Module name substrings that restrict LoRA injection, such as ``["q_proj", "v_proj"]``. ``None`` adapts the
    default modules."""
    override_pipeline_cls_name: str | None = None
    override_transformer_cls_name: str | None = None


@dataclass
class LTX2RefineOptions:
    """Stage-2 refine settings. ``pipeline.components.upsampler_weights`` holds the refine upsampler.

    Resolution copies ``pipeline.preset_overrides.refine`` into the fields of the same name; then the step
    ``fill_ltx2_refine_from_checkpoint`` fills each field that is still ``None`` from the ``fastvideo_refine_*``
    defaults of the checkpoint's ``model_index.json`` (``fastvideo_refine_enabled`` can only turn refine on); and
    ``fill_runtime_defaults`` gives ``enabled``, ``add_noise``, ``guidance_scale``, and ``num_inference_steps`` that
    are still ``None`` the values ``False``, ``True``, 1.0, and 3.
    """

    enabled: bool | None = None
    """Run the stage-2 spatial refine. ``None`` is ``False``."""
    num_inference_steps: int | None = None
    """Stage-2 denoising steps, 2 or 3. ``None`` is 3."""
    guidance_scale: float | None = None
    """Stage-2 guidance scale. ``None`` is 1.0."""
    add_noise: bool | None = None
    """Add noise to the upsampled latents before stage 2. ``None`` is ``True``."""
    image_crf: int | None = None
    """Stage-2 image conditioning CRF from ``preset_overrides.refine``. Requests set it per call."""
    video_position_offset_sec: float | None = None
    """Stage-2 video position offset from ``preset_overrides.refine``. Requests set it per call."""
    transformer_path: str | None = None
    lora_path: str | None = None
    """LoRA applied to the refine transformer only. ``None`` uses the checkpoint's default
    (``fastvideo_refine_lora_path`` in ``model_index.json``); an empty string disables the refine LoRA."""
    noise_path: str | None = None
    audio_noise_path: str | None = None


@dataclass
class ModelOptions:
    """Settings of one model family, written as ``pipeline.model`` with exactly one key that names the family.

    ``ModelOptions`` is a tagged union: the parser reads the single key of the mapping, looks the family up in
    ``TAGS``, and parses the key's value as that family's options class, so a ``pipeline.model`` value is always an
    instance of one of the ``TAGS`` classes with ``family`` set to its tag. The ``family`` key itself is not written
    in the input. ``dit`` and ``vae`` hold overrides for fields of the model's DiT and VAE arch configs, such as
    ``prefix`` or ``load_encoder``; the families add their own settings.
    """

    TAGS: ClassVar[dict[str, type[ModelOptions]]]
    family: str = "generic"
    dit: dict[str, Any] = field(default_factory=dict)
    vae: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericModelOptions(ModelOptions):
    """The ``generic`` block: arch overrides for a model that has no family settings, such as Wan."""


@dataclass
class LTX2Options(ModelOptions):
    """The ``ltx2`` block: LTX-2 settings. ``None`` keeps the model's default."""

    family: str = "ltx2"
    vae_spatial_tile_size_in_pixels: int | None = None
    vae_spatial_tile_overlap_in_pixels: int | None = None
    vae_temporal_tile_size_in_frames: int | None = None
    vae_temporal_tile_overlap_in_frames: int | None = None
    initial_latent_path: str | None = None
    """Path to load or save a precomputed initial video latent."""
    audio_latent_path: str | None = None
    """Path to load or save a precomputed initial audio latent."""
    legacy_native_noise_order: bool | None = None
    """Draw latent noise in the legacy native order, which earlier SSIM references use."""
    use_distilled_sigmas: bool | None = None
    """Use the distilled sigma schedule when the checkpoint provides one."""
    refine: LTX2RefineOptions = field(default_factory=LTX2RefineOptions)


@dataclass
class MiniMaxH3Options(ModelOptions):
    """The ``minimax_h3`` block: MiniMax-H3 settings. ``None`` keeps the model's default."""

    family: str = "minimax_h3"
    sequential_load: bool | None = None
    """Encode with Qwen3-VL, release that encoder, then load the DiT and VAEs. ``None`` enables it on
    unified-memory devices only."""
    video_decode_backend: Literal["h3-vae", "taeh3"] | None = None
    """``h3-vae`` is the full VAE; ``taeh3`` is a fast approximate preview decoder."""
    taeh3_checkpoint: str | None = None
    """Local ``taeh3.safetensors`` path. ``None`` downloads the pinned upstream weights."""
    taeh3_chunk_size: int | None = None
    """TAEH3 latent frames per execution chunk."""
    vae_parallel_decode: bool | None = None
    """Spread VAE decode chunks across the sequence-parallel ranks. ``None`` falls back to
    ``FASTVIDEO_VAE_PARALLEL_DECODE``."""
    vae_parallel_encode: bool | None = None
    """Spread reference-video VAE encode clips across the sequence-parallel ranks. ``None`` falls back to
    ``FASTVIDEO_VAE_PARALLEL_ENCODE``."""
    vae_parallel_decode_strategy: Literal["gather", "all_gather"] | None = None
    """Collective that moves decoded chunks. ``None`` falls back to ``FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY``,
    then ``gather``."""


@dataclass
class LongCatOptions(ModelOptions):
    """The ``longcat`` block: LongCat block sparse attention (BSA) settings. ``None`` keeps the model's default."""

    family: str = "longcat"
    enable_bsa: bool | None = None
    bsa_sparsity: float | None = None
    bsa_cdf_threshold: float | None = None
    bsa_chunk_q: list[int] | None = None
    """Query chunk shape as ``[T, H, W]``."""
    bsa_chunk_k: list[int] | None = None
    """Key chunk shape as ``[T, H, W]``."""


ModelOptions.TAGS = {
    "generic": GenericModelOptions,
    "ltx2": LTX2Options,
    "minimax_h3": MiniMaxH3Options,
    "longcat": LongCatOptions,
}


@dataclass
class PipelineSelection:
    workload_type: WorkloadType | None = None
    preset: str | None = None
    preset_version: int | None = None
    components: ComponentConfig = field(default_factory=ComponentConfig)
    vae_tiling: bool | None = None
    """Tile-based VAE decode. ``None`` keeps the model's default."""
    vae_sp: bool | None = None
    """VAE spatial parallelism across ranks; requires ``vae_tiling``. ``None`` keeps the model's default."""
    flow_shift: float | None = None
    """Flow-matching scheduler shift. ``None`` keeps the model's default."""
    embedded_cfg_scale: float | None = None
    """Guidance scale that guidance-distilled models take as a DiT input. ``None`` keeps the model's default."""
    dmd_denoising_steps: list[int] | None = None
    """Timesteps of a few-step distilled (DMD) sampler. ``None`` keeps the model's default."""
    boundary_ratio: float | None = None
    """Mixture-of-experts switch point of a two-transformer model. ``None`` keeps the model's default."""
    output_type: str = "pil"
    """Output of the decoding stage: ``pil`` for decoded frames, ``latent`` to skip the VAE decode."""
    model: ModelOptions | None = None
    """Model-family settings and arch overrides, keyed by the family: ``ltx2``, ``minimax_h3``, ``longcat``, or
    ``generic``. The family must be the one that the registry picks for ``model_path``, or ``generic``. Resolution
    fills the family's empty block when the field is unset, so a resolved config always has ``pipeline.model``."""
    preset_overrides: dict[str, Any] = field(default_factory=dict)
    experimental: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    model_path: str
    mode: ExecutionMode = ExecutionMode.INFERENCE
    """What the run does: inference, preprocessing, finetuning, or distillation."""
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
    embedded_cfg_scale: float | None = None
    """Guidance scale that guidance-distilled models take as a DiT input, for this request. ``None`` uses
    ``generator.pipeline.embedded_cfg_scale``."""
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

    See :func:`fastvideo.api.compat.explicit_request_raw` for the
    projection and ``entrypoints/openai/request_adapter.py::build_generation_request``
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
    "ExecutionMode",
    "GenerationPlan",
    "GenerationRequest",
    "GeneratorConfig",
    "GenericModelOptions",
    "GpuPoolConfig",
    "InputConfig",
    "LTX2Options",
    "LTX2RefineOptions",
    "LongCatOptions",
    "MiniMaxH3Options",
    "ModelOptions",
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
    "WorkloadType",
]
