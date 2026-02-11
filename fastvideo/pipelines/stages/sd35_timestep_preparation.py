# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


class SD35TimestepPreparationStage(PipelineStage):
    """Timestep preparation for SD3/SD3.5 (FlowMatchEulerDiscreteScheduler).

    Stable Diffusion 3 pipelines can optionally enable resolution-dependent
    shifting (scheduler.config.use_dynamic_shifting). When enabled, the
    scheduler requires `mu` to be passed to `set_timesteps()`.
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @staticmethod
    def _calculate_mu(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return float(image_seq_len) * m + b

    def _maybe_add_dynamic_mu(
        self,
        *,
        extra_kwargs: dict,
        sig: inspect.Signature,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        if "mu" not in sig.parameters:
            return

        cfg = getattr(self.scheduler, "config", None)
        if cfg is None:
            return

        try:
            use_dynamic = bool(cfg.get("use_dynamic_shifting", False))
        except Exception:
            use_dynamic = bool(getattr(cfg, "use_dynamic_shifting", False))

        if not use_dynamic:
            return

        patch_size = getattr(
            fastvideo_args.pipeline_config.dit_config.arch_config,
            "patch_size",
            None,
        )
        spatial_ratio = getattr(
            fastvideo_args.pipeline_config.vae_config.arch_config,
            "spatial_compression_ratio",
            None,
        )
        if not isinstance(patch_size, int) or not isinstance(spatial_ratio, int):
            raise TypeError(
                "SD3.5 dynamic shifting requires integer patch_size and "
                "spatial_compression_ratio."
            )
        if batch.height is None or batch.width is None:
            raise ValueError("height/width must be set before timesteps.")
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise TypeError("SD3.5 expects integer height/width.")

        required_divisor = spatial_ratio * patch_size
        if batch.height % required_divisor != 0 or batch.width % required_divisor != 0:
            raise ValueError(
                f"height/width must be divisible by {required_divisor} for "
                f"SD3.5 (got height={batch.height}, width={batch.width})."
            )

        h_lat = batch.height // spatial_ratio
        w_lat = batch.width // spatial_ratio
        image_seq_len = (h_lat // patch_size) * (w_lat // patch_size)

        try:
            base_seq_len = int(cfg.get("base_image_seq_len", 256))
            max_seq_len = int(cfg.get("max_image_seq_len", 4096))
            base_shift = float(cfg.get("base_shift", 0.5))
            max_shift = float(cfg.get("max_shift", 1.15))
        except Exception:
            base_seq_len = 256
            max_seq_len = 4096
            base_shift = 0.5
            max_shift = 1.15

        extra_kwargs["mu"] = self._calculate_mu(
            image_seq_len=image_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift,
        )

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas

        sig = inspect.signature(scheduler.set_timesteps)

        extra_set_timesteps_kwargs: dict = {}
        if batch.n_tokens is not None and "n_tokens" in sig.parameters:
            extra_set_timesteps_kwargs["n_tokens"] = batch.n_tokens

        self._maybe_add_dynamic_mu(
            extra_kwargs=extra_set_timesteps_kwargs,
            sig=sig,
            batch=batch,
            fastvideo_args=fastvideo_args,
        )

        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please "
                "choose one to set custom values"
            )

        if timesteps is not None:
            if "timesteps" not in sig.parameters:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s "
                    "`set_timesteps` does not support custom timestep schedules."
                )
            timesteps_for_scheduler = timesteps.cpu(
            ) if isinstance(timesteps, torch.Tensor) else timesteps
            scheduler.set_timesteps(
                timesteps=timesteps_for_scheduler,
                device=device,
                **extra_set_timesteps_kwargs,
            )
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            if "sigmas" not in sig.parameters:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s "
                    "`set_timesteps` does not support custom sigmas schedules."
                )
            scheduler.set_timesteps(sigmas=sigmas,
                                    device=device,
                                    **extra_set_timesteps_kwargs)
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(
                num_inference_steps,
                device=device,
                **extra_set_timesteps_kwargs,
            )
            timesteps = scheduler.timesteps

        batch.timesteps = timesteps
        return batch

