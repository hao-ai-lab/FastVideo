# SPDX-License-Identifier: Apache-2.0
"""Typed roots of the legacy training stack (``fastvideo/training/``) and of the preprocessing entry points.

``TrainingRunConfig`` and ``PreprocessRunConfig`` extend ``GeneratorConfig`` with a ``training`` or a ``preprocess``
section. Both resolve to a ``ResolvedGeneratorConfig`` through the generator resolution steps plus their own, so the
pipelines, loaders, and stages read one runtime config type in every mode.

The field defaults are the effective values of the command line that these runs used: the argparse defaults of the
``TrainingArgs`` and ``FastVideoArgs`` flags, including the engine offload settings, which default to off there
(``CommandLineEngineConfig``). Comma strings of the command line (``betas``, ``validation_sampling_steps``) are typed
tuples and lists here.

``load_resolved_run_config`` is the one loader of the entry points: ``--config <yaml>`` plus dotted overrides such as
``--training.optimizer.learning_rate 1e-5``.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from fastvideo.api.inference_resolution import generator_resolution_steps, resolve_config
from fastvideo.api.resolution import ResolutionStep, ResolutionView, ResolvedGeneratorConfig
from fastvideo.api.schema import EngineConfig, ExecutionMode, GeneratorConfig, OffloadConfig
from fastvideo.configs.configs import PreprocessConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TrainingDataOptions:
    """Training data and its loader."""

    data_path: str = ""
    dataloader_num_workers: int = 0
    num_height: int = 0
    num_width: int = 0
    num_frames: int = 0
    train_batch_size: int = 0
    num_latent_t: int = 0
    training_cfg_rate: float | None = None
    """Probability of dropping the text condition of a sample. ``None`` keeps the dataset default."""
    seed: int = 42
    train_sp_batch_size: int | None = None


@dataclass
class TrainingOptimizerOptions:
    """Optimizer and learning-rate schedule of the trained transformer."""

    learning_rate: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float | None = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 10
    lr_num_cycles: int | None = None
    lr_power: float | None = None
    min_lr_ratio: float = 0.5
    """Minimum learning rate ratio of the ``cosine_with_min_lr`` scheduler."""
    max_grad_norm: float | None = None


@dataclass
class TrainingLoopOptions:
    max_train_steps: int | None = None
    gradient_accumulation_steps: int | None = None


@dataclass
class TrainingCheckpointOptions:
    output_dir: str = ""
    resume_from_checkpoint: str | None = None
    training_state_checkpointing_steps: int | None = None
    """Steps between training-state checkpoints, which resume a run."""
    weight_only_checkpointing_steps: int | None = None
    """Steps between weight-only checkpoints, which inference loads."""
    checkpoints_total_limit: int | None = None


@dataclass
class TrainingTrackerOptions:
    trackers: list[str] = field(default_factory=list)
    project_name: str | None = None
    run_name: str | None = None


@dataclass
class TrainingValidationOptions:
    """Validation runs during training, and the intermediate-latent visualizations logged with them."""

    enabled: bool = False
    dataset_file: str | None = None
    sampling_steps: list[int] | None = None
    """Denoising step counts of the validation runs; each count runs once."""
    guidance_scale: float | None = None
    """Guidance scale of the validation runs. ``None`` keeps the model's sampling default."""
    every_steps: int | None = None
    log_visualization: bool = False
    visualization_steps: int | None = None


@dataclass
class TrainingDistillationOptions:
    """DMD distillation: the teacher (real score) and critic (fake score) models and their updates."""

    real_score_model_path: str | None = None
    fake_score_model_path: str | None = None
    generator_update_interval: int = 5
    """Critic updates per student update."""
    min_timestep_ratio: float = 0.2
    max_timestep_ratio: float = 0.98
    real_score_guidance_scale: float = 3.5
    """Teacher CFG scale in the parameterization ``x_cond + w * (x_cond - x_uncond)``; 3.5 is a standard scale of
    4.5."""
    fake_score_learning_rate: float = 0.0
    """Learning rate of the critic; 0.0 uses ``optimizer.learning_rate``."""
    fake_score_lr_scheduler: str = "constant"
    fake_score_betas: tuple[float, float] = (0.9, 0.999)
    simulate_generator_forward: bool = False
    warp_denoising_step: bool = False


@dataclass
class TrainingEMAOptions:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0


@dataclass
class TrainingLoRAOptions:
    enabled: bool = False
    rank: int | None = None
    alpha: int | None = None
    """LoRA alpha. ``None`` uses ``rank``."""


@dataclass
class TrainingVSAOptions:
    """Video sparse attention schedule during training; ``engine.attention.vsa_sparsity`` is the target."""

    decay_rate: float = 0.01
    decay_interval_steps: int = 1
    cache_tile_buf: bool = False
    """Reuse the per-step padded VSA tile buffer across attention layers. Off by default because, under full
    activation checkpointing, the cached buffer survives into the backward recompute and raises peak memory."""


@dataclass
class TrainingSelfForcingOptions:
    dfake_gen_update_ratio: int = 5
    """Train the generator every this many steps."""
    num_frame_per_block: int = 3
    independent_first_frame: bool = False
    same_step_across_blocks: bool = False
    last_step_only: bool = False
    context_noise: int = 0


@dataclass
class TrainingModelOptions:
    """Loss weighting and memory settings of the trained transformer."""

    weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"] = "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    precondition_outputs: bool = False
    enable_gradient_checkpointing_type: Literal["full", "ops", "block_skip"] | None = None
    ltx2_first_frame_conditioning_p: float = 0.1
    """Probability of conditioning an LTX-2 sample on its first frame."""


@dataclass
class TrainingOptions:
    data: TrainingDataOptions = field(default_factory=TrainingDataOptions)
    optimizer: TrainingOptimizerOptions = field(default_factory=TrainingOptimizerOptions)
    loop: TrainingLoopOptions = field(default_factory=TrainingLoopOptions)
    checkpoint: TrainingCheckpointOptions = field(default_factory=TrainingCheckpointOptions)
    tracker: TrainingTrackerOptions = field(default_factory=TrainingTrackerOptions)
    validation: TrainingValidationOptions = field(default_factory=TrainingValidationOptions)
    distillation: TrainingDistillationOptions = field(default_factory=TrainingDistillationOptions)
    ema: TrainingEMAOptions = field(default_factory=TrainingEMAOptions)
    lora: TrainingLoRAOptions = field(default_factory=TrainingLoRAOptions)
    vsa: TrainingVSAOptions = field(default_factory=TrainingVSAOptions)
    self_forcing: TrainingSelfForcingOptions = field(default_factory=TrainingSelfForcingOptions)
    model: TrainingModelOptions = field(default_factory=TrainingModelOptions)


@dataclass
class CommandLineOffloadConfig(OffloadConfig):
    """Offload settings with the defaults of the ``FastVideoArgs`` command line: every offload and pinned host memory
    off."""

    dit: bool = False
    dit_layerwise: bool = False
    text_encoder: bool = False
    image_encoder: bool = False
    vae: bool = False
    pin_cpu_memory: bool = False


@dataclass
class CommandLineEngineConfig(EngineConfig):
    """Engine settings with the offload defaults of the ``FastVideoArgs`` command line."""

    offload: CommandLineOffloadConfig = field(default_factory=CommandLineOffloadConfig)


@dataclass
class TrainingRunConfig(GeneratorConfig):
    """Root config of a ``fastvideo/training/`` run. ``model_path`` is the pretrained model that training starts
    from."""

    mode: ExecutionMode = ExecutionMode.FINETUNING
    engine: CommandLineEngineConfig = field(default_factory=CommandLineEngineConfig)
    training: TrainingOptions = field(default_factory=TrainingOptions)


@dataclass
class PreprocessRunConfig(GeneratorConfig):
    """Root config of a preprocessing run, which encodes a dataset into training inputs."""

    mode: ExecutionMode = ExecutionMode.PREPROCESS
    engine: CommandLineEngineConfig = field(default_factory=CommandLineEngineConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def resolve_training_offload_conflicts(view: ResolutionView) -> dict[str, Any]:
    """Settle offload modes that cannot run together in training, before any device is bound.

    On MPS, FSDP inference and layerwise offload turn off. With layerwise offload on, FSDP inference and DiT CPU
    offload turn off.
    """
    from fastvideo.platforms import current_platform

    values: dict[str, Any] = {}
    if current_platform.is_mps():
        values.update({"engine.use_fsdp_inference": False, "engine.offload.dit_layerwise": False})
    if values.get("engine.offload.dit_layerwise", view.get("engine.offload.dit_layerwise")):
        if view.get("engine.use_fsdp_inference"):
            logger.warning("dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference.")
            values["engine.use_fsdp_inference"] = False
        if view.get("engine.offload.dit"):
            logger.warning("dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload.")
            values["engine.offload.dit"] = False
    return values


def validate_training_parallel_sizes(view: ResolutionView) -> dict[str, Any]:
    """Training needs ``hsdp_replicate_dim``, ``hsdp_shard_dim``, and ``sp_size`` set; -1 is not allowed."""
    for name in ("hsdp_replicate_dim", "hsdp_shard_dim", "sp_size"):
        if view.get(f"engine.parallelism.{name}") == -1:
            raise ValueError(f"{name} must be set for training")
    return {}


def derive_lora_alpha_from_rank(view: ResolutionView) -> dict[str, Any]:
    """``training.lora.alpha`` takes ``training.lora.rank`` while it is unset."""
    rank = view.get("training.lora.rank")
    if view.get("training.lora.alpha") is not None or rank is None:
        return {}
    return {"training.lora.alpha": rank}


def fill_preprocess_model_path(view: ResolutionView) -> dict[str, Any]:
    """``preprocess.model_path`` takes the top-level ``model_path`` while it is empty."""
    if view.get("preprocess.model_path"):
        return {}
    return {"preprocess.model_path": view.get("model_path")}


def validate_preprocess_options(view: ResolutionView) -> dict[str, Any]:
    """A dataset preprocessing run needs ``preprocess.dataset_path`` and positive output sizes.

    The per-task pipelines, which set ``preprocess.data_merge_path`` instead, are not checked.
    """
    if view.get("preprocess.data_merge_path"):
        return {}
    if view.get("preprocess.dataset_path") == "":
        raise ValueError("dataset_path must be set for preprocess mode")
    if view.get("preprocess.samples_per_file") <= 0:
        raise ValueError("samples_per_file must be greater than 0")
    if view.get("preprocess.flush_frequency") <= 0:
        raise ValueError("flush_frequency must be greater than 0")
    return {}


def training_resolution_steps(config: GeneratorConfig, defaults: Any = None) -> tuple[ResolutionStep, ...]:
    """The resolution steps of a ``TrainingRunConfig``, in the order that they run.

    The offload conflicts and the parallel-size check run on the placeholders, before ``derive_parallel_sizes``
    replaces them.
    """
    return generator_resolution_steps(config,
                                      defaults,
                                      before_placeholders=(resolve_training_offload_conflicts,
                                                           validate_training_parallel_sizes),
                                      after=(derive_lora_alpha_from_rank, ))


def derive_video_preprocess_vae_precision(view: ResolutionView) -> dict[str, Any]:
    """A ``preprocess.data_merge_path`` run encodes video in every task except ``text_only``, so its VAE runs in fp32."""
    if (not view.get("preprocess.data_merge_path") or view.get("preprocess.preprocess_task") == "text_only"
            or view.get("engine.precision.vae") == "fp32"):
        return {}
    return {"engine.precision.vae": "fp32"}


def preprocess_resolution_steps(config: GeneratorConfig, defaults: Any = None) -> tuple[ResolutionStep, ...]:
    """The resolution steps of a ``PreprocessRunConfig``, in the order that they run."""
    return generator_resolution_steps(config,
                                      defaults,
                                      after=(derive_video_preprocess_vae_precision, fill_preprocess_model_path,
                                             validate_preprocess_options))


def resolve_training_config(config: TrainingRunConfig | Mapping[str, Any]) -> ResolvedGeneratorConfig:
    """Resolve a ``TrainingRunConfig`` and freeze the result, with its ``PipelineConfig`` as ``pipeline_config``."""
    return resolve_config(config, TrainingRunConfig, training_resolution_steps)


def resolve_preprocess_config(config: PreprocessRunConfig | Mapping[str, Any]) -> ResolvedGeneratorConfig:
    """Resolve a ``PreprocessRunConfig`` and freeze the result, with its ``PipelineConfig`` as ``pipeline_config``."""
    return resolve_config(config, PreprocessRunConfig, preprocess_resolution_steps)


_RESOLVERS: dict[type[GeneratorConfig], Callable[[Any], ResolvedGeneratorConfig]] = {
    TrainingRunConfig: resolve_training_config,
    PreprocessRunConfig: resolve_preprocess_config,
}


def load_resolved_run_config(
    config_class: type[TrainingRunConfig] | type[PreprocessRunConfig],
    argv: Sequence[str] | None = None,
    *,
    mode: ExecutionMode | None = None,
) -> ResolvedGeneratorConfig:
    """Load ``--config <yaml>`` and its dotted overrides from ``argv`` and resolve them as ``config_class``.

    ``argv`` defaults to ``sys.argv[1:]``. Every token after the known ``--config`` option must be a dotted override
    (``--engine.num_gpus 2`` or ``--engine.num_gpus=2``). ``mode`` is the entry point's mode; the file can set
    ``mode`` itself.
    """
    from fastvideo.api.overrides import apply_overrides, parse_cli_overrides
    from fastvideo.api.parser import load_raw_config

    parser = argparse.ArgumentParser(description=f"Run with a {config_class.__name__} YAML file")
    parser.add_argument("--config", required=True, help="Path to the YAML or JSON config file.")
    args, override_tokens = parser.parse_known_args(argv)
    raw = load_raw_config(args.config)
    if mode is not None:
        raw.setdefault("mode", mode.value)
    overrides = parse_cli_overrides(list(override_tokens))
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _RESOLVERS[config_class](raw)


__all__ = [
    "PreprocessRunConfig",
    "TrainingCheckpointOptions",
    "TrainingDataOptions",
    "TrainingDistillationOptions",
    "TrainingEMAOptions",
    "TrainingLoRAOptions",
    "TrainingLoopOptions",
    "TrainingModelOptions",
    "TrainingOptimizerOptions",
    "TrainingOptions",
    "TrainingRunConfig",
    "TrainingSelfForcingOptions",
    "TrainingTrackerOptions",
    "TrainingVSAOptions",
    "TrainingValidationOptions",
    "CommandLineEngineConfig",
    "CommandLineOffloadConfig",
    "derive_lora_alpha_from_rank",
    "fill_preprocess_model_path",
    "load_resolved_run_config",
    "preprocess_resolution_steps",
    "resolve_preprocess_config",
    "resolve_training_config",
    "resolve_training_offload_conflicts",
    "training_resolution_steps",
    "validate_preprocess_options",
    "validate_training_parallel_sizes",
]
