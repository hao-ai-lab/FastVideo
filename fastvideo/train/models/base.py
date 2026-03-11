# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.models.utils import pred_noise_to_pred_video

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.pipelines import TrainingBatch


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

    @property
    def device(self) -> torch.device:
        """The local CUDA device for this rank."""
        return get_local_torch_device()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(  # noqa: B027
            self,
            training_config: TrainingConfig,
    ) -> None:
        """Load VAE, build dataloader, seed RNGs.

        Called only on the student by the method's ``__init__``.
        Default is a no-op so teacher/critic instances skip this.
        """

    def on_train_start(self) -> None:  # noqa: B027
        """Called once before the training loop begins."""

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
        generator: torch.Generator,
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
        """Predict x0 via ``predict_noise`` + conversion."""
        pred_noise = self.predict_noise(
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        return pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler,
        ).unflatten(0, pred_noise.shape[:2])

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
        """Predict x0 streaming via
        ``predict_noise_streaming`` + conversion."""
        pred_noise = self.predict_noise_streaming(
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cache_tag=cache_tag,
            store_kv=store_kv,
            cur_start_frame=cur_start_frame,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        if pred_noise is None:
            return None
        return pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler,
        ).unflatten(0, pred_noise.shape[:2])
