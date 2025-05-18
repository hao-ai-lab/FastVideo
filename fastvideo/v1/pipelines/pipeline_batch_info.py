# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.v1.configs.sample.teacache import (TeaCacheParams,
                                                  WanTeaCacheParams)


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

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"

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
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None

    # Latent dimensions
    height_latents: int | None = None
    width_latents: int | None = None
    num_frames: int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: int | None = None
    width: int | None = None
    fps: int | None = None

    # Timesteps
    timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    # Final output (after pipeline completion)
    output: Any = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_video: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    teacache_params: TeaCacheParams | WanTeaCacheParams | None = None

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
            self.negative_prompt_embeds = []
