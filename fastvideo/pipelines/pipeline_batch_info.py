# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

import pprint
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import PIL.Image
import torch

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder

import time
from collections import OrderedDict

from fastvideo.attention import AttentionMetadata
from fastvideo.configs.sample.teacache import TeaCacheParams, WanTeaCacheParams


class PipelineLoggingInfo:
    """Simple approach using OrderedDict to track stage metrics."""

    def __init__(self):
        # OrderedDict preserves insertion order and allows easy access
        self.stages: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add_stage_execution_time(self, stage_name: str, execution_time: float):
        """Add execution time for a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name]['execution_time'] = execution_time
        self.stages[stage_name]['timestamp'] = time.time()

    def add_stage_metric(self, stage_name: str, metric_name: str, value: Any):
        """Add any metric for a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name][metric_name] = value

    def get_stage_info(self, stage_name: str) -> dict[str, Any]:
        """Get all info for a specific stage."""
        return self.stages.get(stage_name, {})

    def get_execution_order(self) -> list[str]:
        """Get stages in execution order."""
        return list(self.stages.keys())

    def get_total_execution_time(self) -> float:
        """Get total pipeline execution time."""
        return sum(
            stage.get('execution_time', 0) for stage in self.stages.values())


@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.
    
    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """
    # TODO(will): double check that args are separate from fastvideo_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    data_type: str

    generator: torch.Generator | list[torch.Generator] | None = None

    # Image inputs
    image_path: str | None = None
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    pil_image: torch.Tensor | PIL.Image.Image | None = None
    preprocessed_image: torch.Tensor | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_video_name: str | None = None

    # Video inputs
    video_path: str | None = None
    video_latent: torch.Tensor | None = None

    # Primary encoder embeddings
    prompt_embeds: list[torch.Tensor] = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None
    clip_embedding_pos: list[torch.Tensor] | None = None
    clip_embedding_neg: list[torch.Tensor] | None = None

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

    # Batch info
    batch_size: int | None = None
    num_videos_per_prompt: int = 1
    seed: int | None = None
    seeds: list[int] | None = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: list[int] | int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    fps: list[int] | int | None = None

    # Timesteps
    timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None
    boundary_ratio: float | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    # Final output (after pipeline completion)
    output: torch.Tensor | None = None
    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_video: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    teacache_params: TeaCacheParams | WanTeaCacheParams | None = None

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    # Logging info
    logging_info: PipelineLoggingInfo = field(
        default_factory=PipelineLoggingInfo)

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)


@dataclass
class TrainingBatch:
    current_timestep: int = 0
    current_vsa_sparsity: float = 0.0

    # Dataloader batch outputs
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_latents: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    encoder_attention_mask: torch.Tensor | None = None
    # i2v
    preprocessed_image: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None
    image_latents: torch.Tensor | None = None
    infos: list[dict[str, Any]] | None = None
    mask_lat_size: torch.Tensor | None = None

    # ODE trajectory supervision
    trajectory_latents: torch.Tensor | None = None
    trajectory_timesteps: torch.Tensor | None = None

    # Transformer inputs
    noisy_model_input: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    noise: torch.Tensor | None = None

    attn_metadata_vsa: AttentionMetadata | None = None
    attn_metadata: AttentionMetadata | None = None

    # input kwargs
    input_kwargs: dict[str, Any] | None = None

    # Training loss
    loss: torch.Tensor | None = None

    # Training outputs
    total_loss: float | None = None
    grad_norm: float | None = None

    # Distillation-specific attributes
    encoder_hidden_states_neg: torch.Tensor | None = None
    encoder_attention_mask_neg: torch.Tensor | None = None
    conditional_dict: dict[str, Any] | None = None
    unconditional_dict: dict[str, Any] | None = None

    # Distillation losses
    generator_loss: float = 0.0
    fake_score_loss: float = 0.0

    dmd_latent_vis_dict: dict[str, Any] = field(default_factory=dict)
    latent_vis_dict: dict[str, Any] = field(default_factory=dict)
    fake_score_latent_vis_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessBatch(ForwardBatch):
    video_loader: list["VideoDecoder"] | list[str] = field(default_factory=list)
    video_file_name: list[str] = field(default_factory=list)
