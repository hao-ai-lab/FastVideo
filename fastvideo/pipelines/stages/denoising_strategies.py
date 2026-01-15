# SPDX-License-Identifier: Apache-2.0
"""
Strategy interfaces and shared types for unified denoising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


@dataclass
class StrategyState:
    latents: torch.Tensor
    timesteps: torch.Tensor
    num_inference_steps: int
    prompt_embeds: list[torch.Tensor]
    negative_prompt_embeds: list[torch.Tensor] | None
    prompt_attention_mask: list[torch.Tensor] | None
    negative_attention_mask: list[torch.Tensor] | None
    image_embeds: list[torch.Tensor]
    guidance_scale: float
    guidance_scale_2: float | None
    guidance_rescale: float
    do_cfg: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInputs:
    latent_model_input: torch.Tensor
    timestep: torch.Tensor
    prompt_embeds: torch.Tensor | list[torch.Tensor]
    prompt_attention_mask: torch.Tensor | list[torch.Tensor] | None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockPlanItem:
    start_index: int
    num_frames: int
    use_kv_cache: bool
    model_selector: str


@dataclass
class BlockPlan:
    items: list[BlockPlanItem] = field(default_factory=list)


@dataclass
class BlockContext:
    kv_cache: list[dict] | None
    kv_cache_2: list[dict] | None
    crossattn_cache: list[dict] | None
    action_cache: dict[str, list[dict]] | None
    extra: dict[str, Any] = field(default_factory=dict)


class DenoisingStrategy(Protocol):

    def prepare(self, batch: ForwardBatch, args) -> StrategyState:
        ...

    def make_model_inputs(self, state: StrategyState, t: torch.Tensor,
                          step_idx: int) -> ModelInputs:
        ...

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        ...

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        ...

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        ...

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        ...


@runtime_checkable
class BlockDenoisingStrategy(DenoisingStrategy, Protocol):

    def block_plan(self, state: StrategyState) -> BlockPlan:
        ...

    def init_block_context(self, state: StrategyState,
                           block_item: BlockPlanItem,
                           block_idx: int) -> BlockContext:
        ...

    def process_block(self, state: StrategyState, block_ctx: BlockContext,
                      block_item: BlockPlanItem) -> None:
        ...

    def update_context(self, state: StrategyState, block_ctx: BlockContext,
                       block_item: BlockPlanItem) -> None:
        ...
