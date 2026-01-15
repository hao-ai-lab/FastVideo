# SPDX-License-Identifier: Apache-2.0
"""
Scaffolding for a unified denoising engine with hook support.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, TYPE_CHECKING
from collections.abc import Sequence

import torch

from fastvideo.pipelines.stages.denoising_strategies import (
    BlockDenoisingStrategy,
    DenoisingStrategy,
    StrategyState,
)

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class EngineHook(Protocol):

    def on_init(self, engine: DenoisingEngine, batch: ForwardBatch,
                args: FastVideoArgs) -> None:
        ...

    def pre_run(self, state: StrategyState) -> None:
        ...

    def pre_step(self, state: StrategyState, step_idx: int,
                 t: torch.Tensor) -> None:
        ...

    def post_step(self, state: StrategyState, step_idx: int,
                  t: torch.Tensor) -> None:
        ...

    def post_run(self, state: StrategyState, batch: ForwardBatch) -> None:
        ...


class BaseEngineHook:

    def on_init(self, engine: DenoisingEngine, batch: ForwardBatch,
                args: FastVideoArgs) -> None:
        return None

    def pre_run(self, state: StrategyState) -> None:
        return None

    def pre_step(self, state: StrategyState, step_idx: int,
                 t: torch.Tensor) -> None:
        return None

    def post_step(self, state: StrategyState, step_idx: int,
                  t: torch.Tensor) -> None:
        return None

    def post_run(self, state: StrategyState, batch: ForwardBatch) -> None:
        return None


class GuidanceCache:

    def __init__(self, maxsize: int = 8) -> None:
        self._build = lru_cache(maxsize=maxsize)(self._build_impl)

    def _build_impl(self, batch_size: int, dtype: torch.dtype,
                    device: torch.device, guidance_val: float) -> torch.Tensor:
        return (torch.full(
            (batch_size, ),
            guidance_val,
            dtype=torch.float32,
            device=device,
        ).to(dtype) * 1000.0)

    def get(self, batch_size: int, dtype: torch.dtype, device: torch.device,
            guidance_val: float | None) -> torch.Tensor | None:
        if guidance_val is None:
            return None
        return self._build(batch_size, dtype, device, guidance_val)


class DenoisingEngine:

    def __init__(
        self,
        strategy: DenoisingStrategy,
        *,
        hooks: Sequence[EngineHook] | None = None,
    ) -> None:
        self.strategy = strategy
        self.hooks = list(hooks) if hooks is not None else []

    def run(self, batch: ForwardBatch, args: FastVideoArgs) -> ForwardBatch:
        for hook in self.hooks:
            hook.on_init(self, batch, args)

        state = self.strategy.prepare(batch, args)
        for hook in self.hooks:
            hook.pre_run(state)

        if isinstance(self.strategy, BlockDenoisingStrategy):
            self.run_blocks(state)
        else:
            timesteps = state.timesteps
            for i, t in enumerate(timesteps):
                for hook in self.hooks:
                    hook.pre_step(state, i, t)

                model_inputs = self.strategy.make_model_inputs(state, t, i)
                noise_pred = self.strategy.forward(state, model_inputs)
                noise_pred = self.strategy.cfg_combine(state, noise_pred)
                state.latents = self.strategy.scheduler_step(
                    state,
                    noise_pred,
                    t,
                )

                for hook in self.hooks:
                    hook.post_step(state, i, t)

        for hook in self.hooks:
            hook.post_run(state, batch)

        return self.strategy.postprocess(state)

    def run_blocks(
        self,
        state: StrategyState,
        *,
        block_plan=None,
        start_block: int = 0,
        num_blocks: int | None = None,
    ) -> None:
        if not isinstance(self.strategy, BlockDenoisingStrategy):
            raise TypeError("run_blocks requires a BlockDenoisingStrategy")

        strategy = self.strategy
        if block_plan is None:
            block_plan = strategy.block_plan(state)

        items = block_plan.items
        end_block = len(items)
        if num_blocks is not None:
            end_block = min(end_block, start_block + num_blocks)

        for block_idx in range(start_block, end_block):
            block_item = items[block_idx]
            block_ctx = strategy.init_block_context(
                state,
                block_item,
                block_idx,
            )
            strategy.process_block(state, block_ctx, block_item)
            strategy.update_context(state, block_ctx, block_item)
