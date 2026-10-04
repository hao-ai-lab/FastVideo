# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import torch

from fastvideo.attention.selector import coerce_attn_backend
from fastvideo.distributed import get_local_torch_device
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.platforms import AttentionBackendEnum

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.pipelines import TrainingBatch

# Video models return one flow tensor. Joint video/audio models return an
# ordered pair so training methods can apply each modality's scheduler target.
NoisePrediction: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


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

    def __init__(
        self,
        *,
        trainable: bool = True,
        lora: LoraConfig | dict[str, Any] | None = None,
        attention_backend: AttentionBackendEnum | str | None = None,
    ) -> None:
        from fastvideo.train.utils.lora import LoraConfig

        self._trainable = bool(trainable)
        self._lora_config: LoraConfig | None = LoraConfig.coerce(lora)
        self._num_lora_layers = 0
        self.attention_backend = coerce_attn_backend(attention_backend)

    @property
    def attention_backend_name(self) -> str | None:
        """Explicit per-role backend name, or ``None`` for global/default."""
        if self.attention_backend is None:
            return None
        return self.attention_backend.name

    @property
    def device(self) -> torch.device:
        """The local CUDA device for this rank."""
        return get_local_torch_device()

    def _enable_lora_if_configured(
        self,
        transformer: torch.nn.Module,
    ) -> bool:
        """Enable LoRA training for model plugins that request it.

        Concrete models still own transformer loading because class names and
        checkpoint setup are model-specific. The LoRA activation path is shared.
        """
        cfg = self._lora_config
        if cfg is None or not cfg.enable:
            return False
        if not self._trainable:
            raise ValueError("LoRA training requires trainable=true for the role model")

        from fastvideo.train.utils.lora import enable_lora_training

        assert cfg.rank is not None  # guaranteed by LoraConfig validation
        self._num_lora_layers = enable_lora_training(
            transformer,
            lora_rank=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_target_modules=cfg.target_modules,
        )
        return True

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

    def decode_latents(
        self,
        latents_b_t_c_h_w: torch.Tensor,
    ) -> torch.Tensor:
        """Decode ``[B, T, C, H, W]`` latents to ``[B, C, T, H, W]`` media.

        RL reward methods call this hook instead of reaching into
        model-specific VAE normalization details.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement decode_latents()")

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
    ) -> NoisePrediction:
        """Predict video flow or an ordered ``(video, audio)`` flow pair."""

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
        """Convert a video-only flow prediction to clean video latents.

        This helper owns one noisy video tensor and one video scheduler. Joint
        video/audio callers apply their modality-specific conversions where
        both noisy tensors and both schedulers are available.
        """
        pred_noise = self.predict_noise(
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        if isinstance(pred_noise, tuple):
            raise TypeError("predict_x0 requires one video prediction tensor")
        conversion_timestep = timestep
        if (timestep.ndim == 1 and timestep.numel() == noisy_latents.shape[0] and noisy_latents.shape[1] > 1):
            conversion_timestep = timestep.reshape(-1, 1).expand(-1, noisy_latents.shape[1])
        return pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=conversion_timestep,
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

    # ------------------------------------------------------------------
    # TDM contract (distribution matching): per-modality latents, sigma
    # grids, and x0 prediction. A video-only model (Wan) and a joint
    # video+audio model (MiniMax H3) share one code path; the defaults
    # below are exactly the single-video behavior and joint plugins
    # override them.
    # ------------------------------------------------------------------

    def tdm_modalities(self) -> tuple[str, ...]:
        """Modalities TDM trains. Single-video models return one entry."""
        return ("video", )

    def tdm_clean_latents(self, batch: TrainingBatch) -> dict[str, torch.Tensor]:
        """Clean latents per modality for TDM trajectories."""
        if batch.latents is None:
            raise RuntimeError(f"{type(self).__name__} requires batch.latents for TDM")
        return {"video": batch.latents}

    def tdm_sigma_grid(
        self,
        timesteps: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Map integer timestep labels to sigmas for one modality.

        The default mirrors the static shifted-flow grid shared by Wan and
        H3: ``sigma = shift * u / (1 + (shift - 1) * u)`` with
        ``u = t / T``, falling back to the scheduler's own sigma table for
        non-static schedules.
        """
        del modality
        scheduler = self.noise_scheduler
        t = timesteps.to(dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        config = getattr(scheduler, "config", None)
        shift = getattr(scheduler, "shift", None)
        if shift is None:
            shift = getattr(config, "shift", None)
        num_train_timesteps = getattr(
            config,
            "num_train_timesteps",
            getattr(scheduler, "num_train_timesteps", None),
        )
        has_static_flow_schedule = (shift is not None and num_train_timesteps is not None
                                    and not bool(getattr(config, "use_dynamic_shifting", False))
                                    and not getattr(config, "shift_terminal", None)
                                    and not bool(getattr(config, "use_karras_sigmas", False))
                                    and not bool(getattr(config, "use_exponential_sigmas", False))
                                    and not bool(getattr(config, "use_beta_sigmas", False)))
        if has_static_flow_schedule:
            assert shift is not None
            assert num_train_timesteps is not None
            u = t / float(num_train_timesteps)
            flow_shift = float(shift)
            return flow_shift * u / (1.0 + (flow_shift - 1.0) * u)
        scheduler_sigmas = scheduler.sigmas.to(device=t.device, dtype=torch.float32)
        scheduler_timesteps = scheduler.timesteps.to(device=t.device, dtype=torch.float32)
        indices = torch.argmin((scheduler_timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(), dim=1)
        return scheduler_sigmas[indices]

    def tdm_terminal_sigma(self, modality: str) -> torch.Tensor:
        """Largest sigma of one modality's grid (the rollout's start point)."""
        del modality
        scheduler_sigmas = self.noise_scheduler.sigmas
        if scheduler_sigmas is None:
            raise ValueError(f"{type(self).__name__} has no scheduler sigmas for TDM")
        return scheduler_sigmas.to(dtype=torch.float32).max()

    def tdm_max_trajectory_label(self, modality: str) -> int:
        """Largest integer label on one modality's sigma grid."""
        del modality
        return int(self.num_train_timesteps)

    def tdm_sigma_to_model_timestep(
        self,
        sigma: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Map a trajectory sigma to the model's own timestep convention."""
        del modality
        scheduler = self.noise_scheduler
        model_timesteps = scheduler.timesteps.to(device=sigma.device, dtype=torch.float32)
        model_sigmas = scheduler.sigmas[:model_timesteps.numel()].to(device=sigma.device, dtype=torch.float32)
        flat_sigma = sigma.reshape(-1).to(dtype=torch.float32)
        indices = torch.argmin((model_sigmas.unsqueeze(0) - flat_sigma.unsqueeze(1)).abs(), dim=1)
        resolved = model_sigmas[indices]
        if not bool(torch.allclose(resolved, flat_sigma, rtol=1e-5, atol=1e-6)):
            raise ValueError("TDM role scheduler does not contain the requested trajectory sigma")
        return model_timesteps[indices].reshape(sigma.shape)

    def tdm_predict_x0(
        self,
        noisy: dict[str, torch.Tensor],
        sigmas: dict[str, torch.Tensor],
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> dict[str, torch.Tensor]:
        """Predict clean latents per modality from one (joint) forward pass.

        The method works in trajectory-sigma space; each model converts to its
        own timestep convention through ``tdm_sigma_to_model_timestep``.
        """
        model_timestep = self.tdm_sigma_to_model_timestep(sigmas["video"], "video")
        batch.timesteps = model_timestep
        return {
            "video":
            self.predict_x0(
                noisy["video"],
                model_timestep,
                batch,
                conditional=conditional,
                cfg_uncond=cfg_uncond,
                attn_kind=attn_kind,
            )
        }

    def tdm_initial_noise(
        self,
        clean: dict[str, torch.Tensor],
        *,
        generator: torch.Generator,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Draw the TDM rollout's initial noise per modality."""
        return {
            name: torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=dtype)
            for name, tensor in clean.items()
        }


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
