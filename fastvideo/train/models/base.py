# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.logger import init_logger

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.pipelines import TrainingBatch

logger = init_logger(__name__)


class ModelBase(ABC):
    """Per-role model instance.

    Every role (student, teacher, critic, …) gets its own ``ModelBase``
    instance.  Each instance owns its own ``transformer`` and
    ``noise_scheduler``.  Heavyweight resources (VAE, dataloader, RNG
    seeds) are loaded lazily via :meth:`init_preprocessors`, which the
    method calls **only on the student**.
    """

    transformer: torch.nn.Module
    noise_scheduler: Any
    _trainable: bool
    _use_ema: list[str]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Load VAE, build dataloader, seed RNGs.

        Called only on the student by the method's ``__init__``.
        Default is a no-op so teacher/critic instances skip this.
        """

    def on_train_start(self) -> None:
        """Called once before the training loop begins."""

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        """Return RNG generators for checkpoint resume."""
        return {}

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _setup_ema(self) -> None:
        """Create EMA copies of the transformer.

        Call after ``self.transformer`` is set and before FSDP wrapping.
        Each name in ``self._use_ema`` becomes an attribute holding a
        deep copy of ``self.transformer`` in eval mode with no gradients.
        """
        if not getattr(self, "_use_ema", None):
            return
        for name in self._use_ema:
            if hasattr(self, name):
                logger.warning(
                    "EMA network %r already exists, "
                    "skipping initialization.",
                    name,
                )
                continue
            logger.info(
                "Initializing EMA network %r from "
                "transformer",
                name,
            )
            ema = copy.deepcopy(self.transformer)
            ema.eval().requires_grad_(False)
            setattr(self, name, ema)

    @property
    def ema_dict(self) -> dict[str, torch.nn.Module]:
        """Return dict of EMA networks (empty if EMA disabled)."""
        if not getattr(self, "_use_ema", None):
            return {}
        return {
            name: getattr(self, name)
            for name in self._use_ema
            if hasattr(self, name)
        }

    @property
    def transformer_inference(self) -> torch.nn.Module:
        """Return the transformer for inference.

        Returns the first EMA network when available,
        otherwise the training transformer.
        """
        ema = self.ema_dict
        if ema:
            return next(iter(ema.values()))
        return self.transformer

    # ------------------------------------------------------------------
    # Timestep helpers
    # ------------------------------------------------------------------

    @property
    def num_train_timesteps(self) -> int:
        """Return the scheduler's training timestep horizon."""
        return int(self.noise_scheduler.num_train_timesteps)

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Apply model/pipeline timestep shifting and clamp."""
        return timestep

    # ------------------------------------------------------------------
    # Runtime primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Convert a dataloader batch into forward primitives."""

    @abstractmethod
    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply forward-process noise at *timestep*."""

    @abstractmethod
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
        """Predict noise/flow for the given noisy latents."""

    @abstractmethod
    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        """Predict x0 for the given noisy latents."""

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        """Backward that may restore forward-context."""


class CausalModelBase(ModelBase):
    """Extension for causal / streaming model plugins.

    Cache state is internal to the model instance and keyed by
    *cache_tag* (no role handle needed).
    """

    @abstractmethod
    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        """Clear internal caches before starting a new rollout."""

    @abstractmethod
    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        """Streaming predict-noise that may update internal caches."""

    @abstractmethod
    def predict_x0_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        """Streaming predict-x0 that may update internal caches."""
