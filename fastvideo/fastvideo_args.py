# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of FastVideo Inference."""
import argparse
import dataclasses
import json
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any, TYPE_CHECKING

from fastvideo.configs.configs import PreprocessConfig
from fastvideo.configs.pipelines.base import PipelineConfig, STA_Mode
from fastvideo.configs.utils import clean_cli_args
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser, StoreBoolean

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup
else:
    RuntimeEnv = Any
    PlacementGroup = Any

logger = init_logger(__name__)


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
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [mode.value for mode in cls]


class WorkloadType(str, Enum):
    """
    Enumeration for different workload types.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image

    @classmethod
    def from_string(cls, value: str) -> "WorkloadType":
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [workload.value for workload in cls]


# args for fastvideo framework
@dataclasses.dataclass
class FastVideoArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Running mode
    mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"

    # a few attributes for ray related
    ray_placement_group: PlacementGroup | None = None

    ray_runtime_env: RuntimeEnv | None = None

    inference_mode: bool = True  # if False == training mode

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    preprocess_config: PreprocessConfig | None = None

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = True
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False
    torch_compile_kwargs: dict[str, Any] = field(default_factory=dict)

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed training/inference
    master_port: int | None = None

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_txt: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(default_factory=lambda: {
        "transformer": True,
        "vae": True,
    })
    override_transformer_cls_name: str | None = None
    init_weights_from_safetensors: str = ""  # path to safetensors file for initial weight loading
    init_weights_from_safetensors_2: str = ""  # path to safetensors file for initial weight loading for transformer_2

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = 0.875

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    def __post_init__(self):
        if self.moba_config_path:
            try:
                with open(self.moba_config_path) as f:
                    self.moba_config = json.load(f)
                logger.info("Loaded V-MoBA config from %s",
                            self.moba_config_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error("Failed to load V-MoBA config from %s: %s",
                             self.moba_config_path, e)
                raise
        self.check_fastvideo_args()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help=
            "The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )

        # Running mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=ExecutionMode.choices(),
            default=FastVideoArgs.mode.value,
            help="The mode to run FastVideo",
        )

        # Workload type
        parser.add_argument(
            "--workload-type",
            type=str,
            choices=WorkloadType.choices(),
            default=FastVideoArgs.workload_type.value,
            help="The workload type",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=FastVideoArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        parser.add_argument(
            "--inference-mode",
            action=StoreBoolean,
            default=FastVideoArgs.inference_mode,
            help="Whether to use inference mode",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=FastVideoArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=FastVideoArgs.revision,
            help=
            "The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=FastVideoArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=FastVideoArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sp-size",
            type=int,
            default=FastVideoArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=FastVideoArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=FastVideoArgs.hsdp_shard_dim,
            help="The data parallelism shards.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=FastVideoArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Output type
        parser.add_argument(
            "--output-type",
            type=str,
            default=FastVideoArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-txt",
            type=str,
            default=FastVideoArgs.prompt_txt,
            help=
            "Path to a text file containing prompts (one per line) for batch processing",
        )

        # STA (Sliding Tile Attention) parameters
        parser.add_argument(
            "--STA-mode",
            type=str,
            default=FastVideoArgs.STA_mode.value,
            choices=[mode.value for mode in STA_Mode],
            help=
            "STA mode contains STA_inference, STA_searching, STA_tuning, STA_tuning_cfg, None",
        )
        parser.add_argument(
            "--skip-time-steps",
            type=int,
            default=FastVideoArgs.skip_time_steps,
            help="Number of time steps to warmup (full attention) for STA",
        )
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=FastVideoArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference." +
            "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )
        parser.add_argument(
            "--torch-compile-kwargs",
            type=str,
            default=None,
            help=
            "JSON string of kwargs to pass to torch.compile. Example: '{\"backend\":\"inductor\",\"mode\":\"reduce-overhead\"}'",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help=
            "Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help=
            "Pin memory for CPU offload. Only added as a temp workaround if it throws \"CUDA error: invalid argument\". "
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help=
            "Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # VSA parameters
        parser.add_argument(
            "--VSA-sparsity",
            type=float,
            default=FastVideoArgs.VSA_sparsity,
            help="Validation sparsity for VSA",
        )

        # Master port for distributed training/inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=FastVideoArgs.master_port,
            help="Master port for distributed training/inference",
        )

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=FastVideoArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )
        parser.add_argument(
            "--override-transformer-cls-name",
            type=str,
            default=FastVideoArgs.override_transformer_cls_name,
            help="Override transformer cls name",
        )
        parser.add_argument(
            "--init-weights-from-safetensors",
            type=str,
            help="Path to safetensors file for initial weight loading")
        parser.add_argument(
            "--init-weights-from-safetensors-2",
            type=str,
            help="Path to safetensors file for initial weight loading")

        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        # Add preprocessing configuration arguments
        PreprocessConfig.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FastVideoArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == 'pipeline_config':
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs['pipeline_config'] = pipeline_config
            elif attr == 'preprocess_config':
                preprocess_config = PreprocessConfig.from_kwargs(provided_args)
                kwargs['preprocess_config'] = preprocess_config
            elif attr == 'mode':
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, FastVideoArgs.mode.value)
                kwargs['mode'] = ExecutionMode.from_string(
                    mode_value) if isinstance(mode_value, str) else mode_value
            elif attr == 'torch_compile_kwargs':
                # Parse JSON string for torch.compile kwargs
                torch_compile_kwargs_str = getattr(args, 'torch_compile_kwargs',
                                                   None)
                if torch_compile_kwargs_str:
                    try:
                        import json
                        kwargs['torch_compile_kwargs'] = json.loads(
                            torch_compile_kwargs_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON for torch_compile_kwargs: {e}"
                        ) from e
                else:
                    kwargs['torch_compile_kwargs'] = {}
            elif attr == 'workload_type':
                # Convert string to WorkloadType enum
                workload_type_value = getattr(args, 'workload_type',
                                              FastVideoArgs.workload_type.value)
                kwargs['workload_type'] = WorkloadType.from_string(
                    workload_type_value) if isinstance(
                        workload_type_value, str) else workload_type_value
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check if it has a default_factory
                field = dataclasses.fields(cls)[next(
                    i for i, f in enumerate(dataclasses.fields(cls))
                    if f.name == attr)]
                if field.default_factory is not dataclasses.MISSING:
                    # Use the default_factory to create the default value
                    default_value = field.default_factory()
                else:
                    default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

        return cls(**kwargs)  # type: ignore

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "FastVideoArgs":
        # Convert mode string to enum if necessary
        if 'mode' in kwargs and isinstance(kwargs['mode'], str):
            kwargs['mode'] = ExecutionMode.from_string(kwargs['mode'])

        # Convert workload_type string to enum if necessary
        if 'workload_type' in kwargs and isinstance(kwargs['workload_type'],
                                                    str):
            kwargs['workload_type'] = WorkloadType.from_string(
                kwargs['workload_type'])

        kwargs['pipeline_config'] = PipelineConfig.from_kwargs(kwargs)
        kwargs['preprocess_config'] = PreprocessConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    def check_fastvideo_args(self) -> None:
        """Validate inference arguments for consistency"""
        from fastvideo.platforms import current_platform

        if current_platform.is_mps():
            self.use_fsdp_inference = False

        # Validate mode and inference_mode consistency
        assert isinstance(
            self.mode, ExecutionMode
        ), f"Mode must be an ExecutionMode enum, got {type(self.mode)}"
        assert self.mode in ExecutionMode.choices(
        ), f"Invalid execution mode: {self.mode}"

        # Validate workload type
        assert isinstance(
            self.workload_type, WorkloadType
        ), f"Workload type must be a WorkloadType enum, got {type(self.workload_type)}"
        assert self.workload_type in WorkloadType.choices(
        ), f"Invalid workload type: {self.workload_type}"

        if self.mode in [ExecutionMode.DISTILLATION, ExecutionMode.FINETUNING
                         ] and self.inference_mode:
            logger.warning(
                "Mode is 'training' but inference_mode is True. Setting inference_mode to False."
            )
            self.inference_mode = False
        elif self.mode in [ExecutionMode.INFERENCE, ExecutionMode.PREPROCESS
                           ] and not self.inference_mode:
            logger.warning(
                "Mode is '%s' but inference_mode is False. Setting inference_mode to True.",
                self.mode)
            self.inference_mode = True

        if not self.inference_mode:
            assert self.hsdp_replicate_dim != -1, "hsdp_replicate_dim must be set for training"
            assert self.hsdp_shard_dim != -1, "hsdp_shard_dim must be set for training"
            assert self.sp_size != -1, "sp_size must be set for training"

        if self.tp_size == -1:
            self.tp_size = 1
        if self.sp_size == -1:
            self.sp_size = self.num_gpus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0, "num_gpus must >= and be divisible by sp_size"
        assert self.hsdp_replicate_dim <= self.num_gpus and self.num_gpus % self.hsdp_replicate_dim == 0, "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert self.hsdp_shard_dim <= self.num_gpus and self.num_gpus % self.hsdp_shard_dim == 0, "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)

        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in FastVideoArgs")

        self.pipeline_config.check_pipeline_config()

        # Add preprocessing config validation if needed
        if self.mode == ExecutionMode.PREPROCESS:
            if self.preprocess_config is None:
                raise ValueError(
                    "preprocess_config is not set in FastVideoArgs when mode is PREPROCESS"
                )
            if self.preprocess_config.model_path == "":
                self.preprocess_config.model_path = self.model_path
            if not self.pipeline_config.vae_config.load_encoder:
                self.pipeline_config.vae_config.load_encoder = True
            self.preprocess_config.check_preprocess_config()


_current_fastvideo_args = None


def prepare_fastvideo_args(argv: list[str]) -> FastVideoArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    FastVideoArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    fastvideo_args = FastVideoArgs.from_cli_args(raw_args)
    global _current_fastvideo_args
    _current_fastvideo_args = fastvideo_args
    return fastvideo_args


@contextmanager
def set_current_fastvideo_args(fastvideo_args: FastVideoArgs):
    """
    Temporarily set the current fastvideo config.
    Used during model initialization.
    We save the current fastvideo config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the fastvideo config to determine how to dispatch.
    """
    global _current_fastvideo_args
    old_fastvideo_args = _current_fastvideo_args
    try:
        _current_fastvideo_args = fastvideo_args
        yield
    finally:
        _current_fastvideo_args = old_fastvideo_args


def get_current_fastvideo_args() -> FastVideoArgs:
    if _current_fastvideo_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the fastvideo config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current fastvideo args is not set.")
    return _current_fastvideo_args


@dataclasses.dataclass
class TrainingArgs(FastVideoArgs):
    """
    Training arguments. Inherits from FastVideoArgs and adds training-specific
    arguments. If there are any conflicts, the training arguments will take
    precedence.
    """
    data_path: str = ""
    dataloader_num_workers: int = 0
    num_height: int = 0
    num_width: int = 0
    num_frames: int = 0

    train_batch_size: int = 0
    num_latent_t: int = 0
    group_frame: bool = False
    group_resolution: bool = False

    # text encoder & vae & diffusion model
    pretrained_model_name_or_path: str = ""

    # DMD model paths - separate paths for each network
    real_score_model_path: str = ""  # path for real score (teacher) model
    fake_score_model_path: str = ""  # path for fake score (critic) model

    # diffusion setting
    ema_decay: float = 0.0
    ema_start_step: int = 0
    training_cfg_rate: float = 0.0
    precondition_outputs: bool = False

    # validation & logs
    validation_dataset_file: str = ""
    validation_preprocessed_path: str = ""
    validation_sampling_steps: str = ""
    validation_guidance_scale: str = ""
    validation_steps: float = 0.0
    log_validation: bool = False
    trackers: list[str] = dataclasses.field(default_factory=list)
    tracker_project_name: str = ""
    wandb_run_name: str = ""
    seed: int | None = None

    # output
    output_dir: str = ""
    checkpoints_total_limit: int = 0
    resume_from_checkpoint: str = ""  # specify the checkpoint folder to resume from

    # optimizer & scheduler
    num_train_epochs: int = 0
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 0
    learning_rate: float = 0.0
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 0.0
    enable_gradient_checkpointing_type: str | None = None
    selective_checkpointing: float = 0.0
    mixed_precision: str = ""
    train_sp_batch_size: int = 0
    fsdp_sharding_startegy: str = ""

    weighting_scheme: str = ""
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 0.0

    num_euler_timesteps: int = 0
    lr_num_cycles: int = 0
    lr_power: float = 0.0
    min_lr_ratio: float = 0.5  # minimum learning rate ratio for cosine_with_min_lr scheduler
    not_apply_cfg_solver: bool = False
    distill_cfg: float = 0.0
    scheduler_type: str = ""
    linear_quadratic_threshold: float = 0.0
    linear_range: float = 0.0
    weight_decay: float = 0.0
    betas: str = "0.9,0.999"  # betas for optimizer, format: "beta1,beta2"
    use_ema: bool = False
    multi_phased_distill_schedule: str = ""
    pred_decay_weight: float = 0.0
    pred_decay_type: str = ""
    hunyuan_teacher_disable_cfg: bool = False

    # master_weight_type
    master_weight_type: str = ""

    # VSA training decay parameters
    VSA_decay_rate: float = 0.01  # decay rate -> 0.02
    VSA_decay_interval_steps: int = 1  # decay interval steps -> 50

    # LoRA training parameters
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_training: bool = False

    # distillation args
    generator_update_interval: int = 5
    dfake_gen_update_ratio: int = 5  # self-forcing: how often to train generator vs critic
    min_timestep_ratio: float = 0.2
    max_timestep_ratio: float = 0.98
    real_score_guidance_scale: float = 3.5
    fake_score_learning_rate: float = 0.0  # separate learning rate for fake_score_transformer, if 0.0, use learning_rate
    fake_score_lr_scheduler: str = "constant"  # separate lr scheduler for fake_score_transformer, if not set, use lr_scheduler
    fake_score_betas: str = "0.9,0.999"  # betas for fake score optimizer, format: "beta1,beta2"
    training_state_checkpointing_steps: int = 0  # for resuming training
    weight_only_checkpointing_steps: int = 0  # for inference
    log_visualization: bool = False
    # simulate generator forward to match inference
    simulate_generator_forward: bool = False
    warp_denoising_step: bool = False

    # Self-forcing specific arguments
    num_frame_per_block: int = 3
    independent_first_frame: bool = False
    enable_gradient_masking: bool = True
    gradient_mask_last_n_frames: int = 21
    same_step_across_blocks: bool = False  # Use same exit timestep for all blocks
    last_step_only: bool = False  # Only use the last timestep for training
    context_noise: int = 0  # Context noise level for cache updates

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TrainingArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        logger.info(provided_args)
        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == 'pipeline_config':
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs[attr] = pipeline_config
            elif attr == 'mode':
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, ExecutionMode.FINETUNING.value)
                kwargs[attr] = ExecutionMode.from_string(
                    mode_value) if isinstance(mode_value, str) else mode_value
            elif attr == 'workload_type':
                # Convert string to WorkloadType enum
                workload_type_value = getattr(args, 'workload_type',
                                              WorkloadType.T2V.value)
                kwargs[attr] = WorkloadType.from_string(
                    workload_type_value) if isinstance(
                        workload_type_value, str) else workload_type_value
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check its default value
                field = dataclasses.fields(cls)[next(
                    i for i, f in enumerate(dataclasses.fields(cls))
                    if f.name == attr)]

                # Check if the attribute is provided in args
                if hasattr(args, attr):
                    value = getattr(args, attr)
                else:
                    # Use the field's default value
                    if field.default_factory is not dataclasses.MISSING:
                        value = field.default_factory()
                    elif field.default is not dataclasses.MISSING:
                        value = field.default
                    else:
                        # No default value, use None
                        value = None

                kwargs[attr] = value

        return cls(**kwargs)  # type: ignore

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser.add_argument("--data-path",
                            type=str,
                            required=True,
                            help="Path to parquet files")
        parser.add_argument("--dataloader-num-workers",
                            type=int,
                            required=True,
                            help="Number of workers for dataloader")
        parser.add_argument("--num-height",
                            type=int,
                            required=True,
                            help="Number of heights")
        parser.add_argument("--num-width",
                            type=int,
                            required=True,
                            help="Number of widths")
        parser.add_argument("--num-frames",
                            type=int,
                            required=True,
                            help="Number of frames")

        # Training batch and model configuration
        parser.add_argument("--train-batch-size",
                            type=int,
                            required=True,
                            help="Training batch size")
        parser.add_argument("--num-latent-t",
                            type=int,
                            required=True,
                            help="Number of latent time steps")
        parser.add_argument("--group-frame",
                            action=StoreBoolean,
                            help="Whether to group frames during training")
        parser.add_argument("--group-resolution",
                            action=StoreBoolean,
                            help="Whether to group resolutions during training")

        # Model paths
        parser.add_argument("--pretrained-model-name-or-path",
                            type=str,
                            required=True,
                            help="Path to pretrained model or model name")
        parser.add_argument("--dit-model-name-or-path",
                            type=str,
                            required=False,
                            help="Path to DiT model or model name")
        parser.add_argument("--cache-dir",
                            type=str,
                            help="Directory to cache models")

        # DMD model paths - separate paths for each network
        parser.add_argument(
            "--generator-model-path",
            type=str,
            help="Path to generator (student) model for DMD distillation")
        parser.add_argument(
            "--real-score-model-path",
            type=str,
            help="Path to real score (teacher) model for DMD distillation")
        parser.add_argument(
            "--fake-score-model-path",
            type=str,
            help="Path to fake score (critic) model for DMD distillation")

        # Diffusion settings
        parser.add_argument("--ema-decay",
                            type=float,
                            default=0.999,
                            help="EMA decay rate")
        parser.add_argument("--ema-start-step",
                            type=int,
                            default=0,
                            help="Step to start EMA")
        parser.add_argument("--training-cfg-rate",
                            type=float,
                            help="Classifier-free guidance scale")
        parser.add_argument(
            "--precondition-outputs",
            action=StoreBoolean,
            help="Whether to precondition the outputs of the model")

        # Validation and logging
        parser.add_argument("--validation-dataset-file",
                            type=str,
                            help="Path to unprocessed validation dataset")
        parser.add_argument("--validation-preprocessed-path",
                            type=str,
                            help="Path to processed validation dataset")
        parser.add_argument("--validation-sampling-steps",
                            type=str,
                            help="Validation sampling steps")
        parser.add_argument("--validation-guidance-scale",
                            type=str,
                            help="Validation guidance scale")
        parser.add_argument("--validation-steps",
                            type=float,
                            help="Number of validation steps")
        parser.add_argument("--log-validation",
                            action=StoreBoolean,
                            help="Whether to log validation results")
        parser.add_argument("--tracker-project-name",
                            type=str,
                            help="Project name for tracking")
        parser.add_argument("--wandb-run-name",
                            type=str,
                            help="Run name for wandb")
        parser.add_argument("--seed",
                            type=int,
                            default=42,
                            help="Seed for deterministic training")

        # Output configuration
        parser.add_argument("--output-dir",
                            type=str,
                            required=True,
                            help="Output directory for checkpoints and logs")
        parser.add_argument("--checkpoints-total-limit",
                            type=int,
                            help="Maximum number of checkpoints to keep")
        parser.add_argument(
            "--training-state-checkpointing-steps",
            type=int,
            help=
            "Steps between training state checkpoints (for resuming training)")
        parser.add_argument(
            "--weight-only-checkpointing-steps",
            type=int,
            help="Steps between weight-only checkpoints (for inference)")
        parser.add_argument("--resume-from-checkpoint",
                            type=str,
                            help="Path to checkpoint to resume from")
        parser.add_argument("--logging-dir",
                            type=str,
                            help="Directory for logging")

        # Training configuration
        parser.add_argument("--num-train-epochs",
                            type=int,
                            help="Number of training epochs")
        parser.add_argument("--max-train-steps",
                            type=int,
                            help="Maximum number of training steps")
        parser.add_argument("--gradient-accumulation-steps",
                            type=int,
                            help="Number of steps to accumulate gradients")
        parser.add_argument("--learning-rate",
                            type=float,
                            required=True,
                            help="Learning rate")
        parser.add_argument("--scale-lr",
                            action=StoreBoolean,
                            help="Whether to scale learning rate")
        parser.add_argument("--lr-scheduler",
                            type=str,
                            default="constant",
                            help="Learning rate scheduler type")
        parser.add_argument("--lr-warmup-steps",
                            type=int,
                            default=10,
                            help="Number of warmup steps for learning rate")
        parser.add_argument("--max-grad-norm",
                            type=float,
                            help="Maximum gradient norm")
        parser.add_argument("--enable-gradient-checkpointing-type",
                            type=str,
                            choices=["full", "ops", "block_skip"],
                            default=None,
                            help="Gradient checkpointing type")
        parser.add_argument("--selective-checkpointing",
                            type=float,
                            help="Selective checkpointing threshold")
        parser.add_argument("--mixed-precision",
                            type=str,
                            help="Mixed precision training type")
        parser.add_argument("--train-sp-batch-size",
                            type=int,
                            help="Training spatial parallelism batch size")

        parser.add_argument("--fsdp-sharding-strategy",
                            type=str,
                            help="FSDP sharding strategy")

        parser.add_argument(
            "--weighting_scheme",
            type=str,
            default="uniform",
            choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
        )
        parser.add_argument(
            "--logit_mean",
            type=float,
            default=0.0,
            help="mean to use when using the `'logit_normal'` weighting scheme.",
        )
        parser.add_argument(
            "--logit_std",
            type=float,
            default=1.0,
            help="std to use when using the `'logit_normal'` weighting scheme.",
        )
        parser.add_argument(
            "--mode_scale",
            type=float,
            default=1.29,
            help=
            "Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
        )

        # Additional training parameters
        parser.add_argument("--num-euler-timesteps",
                            type=int,
                            help="Number of Euler timesteps")
        parser.add_argument("--lr-num-cycles",
                            type=int,
                            help="Number of learning rate cycles")
        parser.add_argument("--lr-power",
                            type=float,
                            help="Learning rate power")
        parser.add_argument(
            "--min-lr-ratio",
            type=float,
            default=TrainingArgs.min_lr_ratio,
            help="Minimum learning rate ratio for cosine_with_min_lr scheduler")
        parser.add_argument("--not-apply-cfg-solver",
                            action=StoreBoolean,
                            help="Whether to not apply CFG solver")
        parser.add_argument("--distill-cfg",
                            type=float,
                            help="Distillation CFG scale")
        parser.add_argument("--scheduler-type", type=str, help="Scheduler type")
        parser.add_argument("--linear-quadratic-threshold",
                            type=float,
                            help="Linear quadratic threshold")
        parser.add_argument("--linear-range", type=float, help="Linear range")
        parser.add_argument("--weight-decay", type=float, help="Weight decay")
        parser.add_argument("--betas",
                            type=str,
                            default=TrainingArgs.betas,
                            help="Betas for optimizer (format: 'beta1,beta2')")
        parser.add_argument("--use-ema",
                            action=StoreBoolean,
                            help="Whether to use EMA")
        parser.add_argument("--multi-phased-distill-schedule",
                            type=str,
                            help="Multi-phased distillation schedule")
        parser.add_argument("--pred-decay-weight",
                            type=float,
                            help="Prediction decay weight")
        parser.add_argument("--pred-decay-type",
                            type=str,
                            help="Prediction decay type")
        parser.add_argument("--hunyuan-teacher-disable-cfg",
                            action=StoreBoolean,
                            help="Whether to disable CFG for Hunyuan teacher")
        parser.add_argument("--master-weight-type",
                            type=str,
                            help="Master weight type")

        # VSA parameters for training with dense to sparse adaption
        parser.add_argument(
            "--VSA-decay-rate",  # decay rate, how much sparsity you want to decay each step
            type=float,
            default=TrainingArgs.VSA_decay_rate,
            help="VSA decay rate")
        parser.add_argument(
            "--VSA-decay-interval-steps",  # how many steps for training with current sparsity
            type=int,
            default=TrainingArgs.VSA_decay_interval_steps,
            help="VSA decay interval steps")
        parser.add_argument("--lora-training",
                            action=StoreBoolean,
                            help="Whether to use LoRA training")
        parser.add_argument("--lora-rank", type=int, help="LoRA rank")
        parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")

        # V-MoBA parameters
        parser.add_argument(
            "--moba-config-path",
            type=str,
            default=None,
            help=
            "Path to a JSON file containing V-MoBA specific configurations.",
        )

        # Distillation arguments
        parser.add_argument("--generator-update-interval",
                            type=int,
                            default=TrainingArgs.generator_update_interval,
                            help="Ratio of student updates to critic updates.")
        parser.add_argument(
            "--dfake-gen-update-ratio",
            type=int,
            default=TrainingArgs.dfake_gen_update_ratio,
            help=
            "Self-forcing: How often to train generator vs critic (train generator every N steps)."
        )
        parser.add_argument("--min-timestep-ratio",
                            type=float,
                            default=TrainingArgs.min_timestep_ratio,
                            help="Minimum step ratio")
        parser.add_argument("--max-timestep-ratio",
                            type=float,
                            default=TrainingArgs.max_timestep_ratio,
                            help="Maximum step ratio")
        parser.add_argument("--real-score-guidance-scale",
                            type=float,
                            default=TrainingArgs.real_score_guidance_scale,
                            help="Teacher guidance scale")
        parser.add_argument("--fake-score-learning-rate",
                            type=float,
                            default=TrainingArgs.fake_score_learning_rate,
                            help="Learning rate for fake score transformer")
        parser.add_argument(
            "--fake-score-betas",
            type=str,
            default=TrainingArgs.fake_score_betas,
            help="Betas for fake score optimizer (format: 'beta1,beta2')")
        parser.add_argument(
            "--fake-score-lr-scheduler",
            type=str,
            default=TrainingArgs.fake_score_lr_scheduler,
            help="Learning rate scheduler for fake score transformer")
        parser.add_argument("--log-visualization",
                            action=StoreBoolean,
                            help="Whether to log visualization")
        parser.add_argument(
            "--simulate-generator-forward",
            action=StoreBoolean,
            help="Whether to simulate generator forward to match inference")
        parser.add_argument(
            "--warp-denoising-step",
            action=StoreBoolean,
            help=
            "Whether to warp denoising step according to the scheduler time shift"
        )

        # Self-forcing specific arguments
        parser.add_argument(
            "--num-frame-per-block",
            type=int,
            default=TrainingArgs.num_frame_per_block,
            help="Number of frames per block for causal generation")
        parser.add_argument(
            "--independent-first-frame",
            action=StoreBoolean,
            help="Whether the first frame is independent in causal generation")
        parser.add_argument(
            "--enable-gradient-masking",
            action=StoreBoolean,
            help="Whether to enable frame-level gradient masking")
        parser.add_argument(
            "--gradient-mask-last-n-frames",
            type=int,
            default=TrainingArgs.gradient_mask_last_n_frames,
            help="Number of last frames to enable gradients for")
        parser.add_argument(
            "--validate-cache-structure",
            action=StoreBoolean,
            help="Whether to validate KV cache structure (debug flag)")
        parser.add_argument(
            "--same-step-across-blocks",
            action=StoreBoolean,
            help="Whether to use the same exit timestep for all blocks")
        parser.add_argument(
            "--last-step-only",
            action=StoreBoolean,
            help="Whether to only use the last timestep for training")
        parser.add_argument("--context-noise",
                            type=int,
                            default=TrainingArgs.context_noise,
                            help="Context noise level for cache updates")

        return parser


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]
