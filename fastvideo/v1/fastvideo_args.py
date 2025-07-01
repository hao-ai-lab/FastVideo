# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of FastVideo Inference."""

import argparse
import dataclasses
from contextlib import contextmanager
from dataclasses import field
from typing import Any, Dict, List, Optional

from fastvideo.v1.configs.pipelines.base import PipelineConfig, STA_Mode
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import FlexibleArgumentParser, StoreBoolean

logger = init_logger(__name__)


def clean_cli_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Clean the arguments by removing the ones that not explicitly provided by the user.
    """
    provided_args = {}
    for k, v in vars(args).items():
        if (v is not None and hasattr(args, '_provided')
                and k in args._provided):
            provided_args[k] = v

    return provided_args


# args for fastvideo framework
@dataclasses.dataclass
class FastVideoArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"

    inference_mode: bool = True  # if False == training mode

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: Optional[str] = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: Optional[int] = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)

    output_type: str = "pil"

    use_cpu_offload: bool = True
    use_fsdp_inference: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # Stage verification
    enable_stage_verification: bool = True

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    def __post_init__(self):
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
            help=
            "Use torch.compile for speeding up STA inference without teacache",
        )

        parser.add_argument(
            "--use-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for model inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help=
            "Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
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

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=FastVideoArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )

        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FastVideoArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs = {}
        for attr in attrs:
            if attr == 'pipeline_config':
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs[attr] = pipeline_config
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

        return cls(**kwargs)  # type: ignore

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> "FastVideoArgs":
        kwargs['pipeline_config'] = PipelineConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    def check_fastvideo_args(self) -> None:
        """Validate inference arguments for consistency"""
        if not self.inference_mode:
            assert self.hsdp_replicate_dim != -1, "hsdp_replicate_dim must be set for training"
            assert self.hsdp_shard_dim != -1, "hsdp_shard_dim must be set for training"
            assert self.sp_size != -1, "sp_size must be set for training"

        if self.tp_size == -1:
            self.tp_size = self.num_gpus
        if self.sp_size == -1:
            self.sp_size = self.num_gpus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0, "num_gpus must >= and be divisible by sp_size"
        assert self.hsdp_replicate_dim <= self.num_gpus and self.num_gpus % self.hsdp_replicate_dim == 0, "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert self.hsdp_shard_dim <= self.num_gpus and self.num_gpus % self.hsdp_shard_dim == 0, "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)

        if self.tp_size != self.sp_size:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be equal to sp_size ({self.sp_size})"
            )

        if self.enable_torch_compile and self.num_gpus > 1:
            logger.warning(
                "Currently torch compile does not work with multi-gpu. Setting enable_torch_compile to False"
            )
            self.enable_torch_compile = False

        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in FastVideoArgs")

        self.pipeline_config.check_pipeline_config()


_current_fastvideo_args = None


def prepare_fastvideo_args(argv: List[str]) -> FastVideoArgs:
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
    dit_model_name_or_path: str = ""

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
    tracker_project_name: str = ""
    wandb_run_name: str = ""
    seed: Optional[int] = None

    # output
    output_dir: str = ""
    checkpoints_total_limit: int = 0
    checkpointing_steps: int = 0
    resume_from_checkpoint: bool = False

    # optimizer & scheduler
    num_train_epochs: int = 0
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 0
    learning_rate: float = 0.0
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 0.0
    enable_gradient_checkpointing_type: Optional[str] = None
    selective_checkpointing: float = 0.0
    allow_tf32: bool = False
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
    not_apply_cfg_solver: bool = False
    distill_cfg: float = 0.0
    scheduler_type: str = ""
    linear_quadratic_threshold: float = 0.0
    linear_range: float = 0.0
    weight_decay: float = 0.0
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

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TrainingArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        logger.info(provided_args)
        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs = {}
        for attr in attrs:
            if attr == 'pipeline_config':
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs[attr] = pipeline_config
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

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
        parser.add_argument("--checkpointing-steps",
                            type=int,
                            help="Steps between checkpoints")
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
        parser.add_argument("--allow-tf32",
                            action=StoreBoolean,
                            help="Whether to allow TF32")
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

        return parser
