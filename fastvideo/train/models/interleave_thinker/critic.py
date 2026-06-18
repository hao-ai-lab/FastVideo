# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker critic actor adapter shell."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import ModelBase

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


class _InterleaveThinkerActorModule(torch.nn.Module):
    """Checkpoint-visible placeholder for future actor backends."""


class InterleaveThinkerCriticModel(ModelBase):
    """ModelBase adapter slot for InterleaveThinker critic RL.

    Upstream InterleaveThinker trains a Qwen3-VL critic with EasyR1. FastVideo
    does not yet ship a native Qwen3-VL actor wrapper, so this class documents
    the hooks consumed by :class:`InterleaveThinkerRLMethod` and gives YAML
    configs an importable model target. A concrete backend should override
    ``generate_interleave_responses`` and ``train_interleave_rollouts``.
    """

    def __init__(
        self,
        *,
        init_from: str = "",
        training_config: TrainingConfig | None = None,
        trainable: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(trainable=trainable)
        self.init_from = str(init_from or "")
        self.training_config = training_config
        self.transformer = _InterleaveThinkerActorModule()
        self.noise_scheduler = SimpleNamespace(num_train_timesteps=0)

    def generate_interleave_responses(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del batch, kwargs
        raise NotImplementedError("InterleaveThinkerCriticModel requires a concrete Qwen/VLM actor backend "
                                  "that implements generate_interleave_responses().")

    def train_interleave_rollouts(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        del kwargs
        raise NotImplementedError("InterleaveThinkerCriticModel requires a concrete Qwen/VLM actor backend "
                                  "that implements train_interleave_rollouts().")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del clean_latents, noise, timestep
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del noisy_latents, timestep, batch, conditional, cfg_uncond, attn_kind
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del loss, ctx, grad_accum_rounds
        raise NotImplementedError("InterleaveThinkerCriticModel uses train_interleave_rollouts().")
