# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

import pprint
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import PIL.Image
import torch

from fastvideo.v1.attention import AttentionMetadata
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

    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None

    # Image inputs
    image_path: Optional[str] = None
    image_embeds: List[torch.Tensor] = field(default_factory=list)
    pil_image: Optional[PIL.Image.Image] = None
    preprocessed_image: Optional[torch.Tensor] = None

    # Text inputs
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"

    # Primary encoder embeddings
    prompt_embeds: List[torch.Tensor] = field(default_factory=list)
    negative_prompt_embeds: Optional[List[torch.Tensor]] = None
    prompt_attention_mask: Optional[List[torch.Tensor]] = None
    negative_attention_mask: Optional[List[torch.Tensor]] = None
    clip_embedding_pos: Optional[List[torch.Tensor]] = None
    clip_embedding_neg: Optional[List[torch.Tensor]] = None

    # Additional text-related parameters
    max_sequence_length: Optional[int] = None
    prompt_template: Optional[Dict[str, Any]] = None
    do_classifier_free_guidance: bool = False

    # Batch info
    batch_size: Optional[int] = None
    num_videos_per_prompt: int = 1
    seed: Optional[int] = None
    seeds: Optional[List[int]] = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: Optional[torch.Tensor] = None
    raw_latent_shape: Optional[torch.Tensor] = None
    noise_pred: Optional[torch.Tensor] = None
    image_latent: Optional[torch.Tensor] = None

    # Latent dimensions
    height_latents: Optional[int] = None
    width_latents: Optional[int] = None
    num_frames: int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: Optional[int] = None
    width: Optional[int] = None
    fps: Optional[int] = None

    # Timesteps
    timesteps: Optional[torch.Tensor] = None
    timestep: Optional[Union[torch.Tensor, float, int]] = None
    step_index: Optional[int] = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: Optional[List[float]] = None

    n_tokens: Optional[int] = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: Dict[str, Any] = field(default_factory=dict)

    # Final output (after pipeline completion)
    output: Any = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: Dict[str, Any] = field(default_factory=dict)

    # Misc
    save_video: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    teacache_params: Optional[TeaCacheParams | WanTeaCacheParams] = None

    # STA parameters
    STA_param: Optional[List] = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: Optional[List[List]] = None
    mask_search_final_result_neg: Optional[List[List]] = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)


@dataclass
class TrainingBatch:
    current_timestep: int = 0
    current_vsa_sparsity: float = 0.0

    # Dataloader batch outputs
    latents: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    # i2v
    preprocessed_image: Optional[torch.Tensor] = None
    image_embeds: Optional[torch.Tensor] = None
    image_latents: Optional[torch.Tensor] = None
    infos: Optional[List[Dict[str, Any]]] = None

    # Transformer inputs
    noisy_model_input: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None
    sigmas: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None

    attn_metadata: Optional[AttentionMetadata] = None

    # input kwargs
    input_kwargs: Optional[Dict[str, Any]] = None

    # Training loss
    loss: torch.Tensor | None = None

    # Training outputs
    total_loss: float | None = None
    grad_norm: float | None = None
