# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distillation.roles import RoleHandle

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch


class ModelAdapter(ABC):
    """Operation-centric runtime primitives implemented by a model plugin.

    This interface is intentionally *method-agnostic*:
    - A method selects roles (student/teacher/critic/...) and decides how to use
      them.
    - The adapter implements how to run those roles against FastVideo pipelines,
      forward-context requirements, and batch normalization quirks.

    Implementations typically live next to the model plugin (e.g. `models/wan.py`)
    rather than in a global adapter registry.
    """

    training_args: Any

    @property
    @abstractmethod
    def num_train_timesteps(self) -> int:
        """Return the scheduler's training timestep horizon (usually 1000)."""

    @abstractmethod
    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Apply model/pipeline timestep shifting and clamp into valid range."""

    @abstractmethod
    def on_train_start(self) -> None:
        """Initialize RNG seeds and any cached conditioning needed for training."""

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        """Return RNG generators that should be checkpointed for exact resume."""

        return {}

    @abstractmethod
    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Convert a dataloader batch into forward primitives for methods."""

    @abstractmethod
    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply forward-process noise at `timestep` for a given scheduler."""

    @abstractmethod
    def predict_noise(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Run a role to predict noise/flow for the given noisy latents."""

    @abstractmethod
    def predict_x0(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Run a role to predict x0 for the given noisy latents."""

    @abstractmethod
    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        """Backward hook that may restore forward-context for checkpointed modules."""

