# SPDX-License-Identifier: Apache-2.0
"""Wan validator (model-family validation backend).

Config keys used:
- `training` (DistillTrainingConfig):
  - `data.seed`, `model_path`
  - `data.num_height`, `data.num_width`, `data.num_latent_t`
  - `distributed.tp_size`, `distributed.sp_size`,
    `distributed.num_gpus`, `distributed.pin_cpu_memory`
  - `pipeline_config.flow_shift`
  - `pipeline_config.vae_config.arch_config
     .temporal_compression_ratio`
  - `vsa.sparsity`
- `training.validation.*` (typically parsed by a method into
  `ValidationRequest`):
  - `dataset_file`, `sampling_steps`, `guidance_scale`
  - `sampler_kind` (`ode`/`sde`), `ode_solver` (`euler`/`unipc`),
    `rollout_mode`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.validation_dataset import (
    ValidationDataset, )
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.distillation.utils.moduleloader import (
    make_inference_args, )
from fastvideo.distillation.validators.base import (
    ValidationRequest, )
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.training.trackers import DummyTracker
from fastvideo.utils import shallow_asdict

if TYPE_CHECKING:
    from fastvideo.distillation.utils.distill_config import (
        DistillTrainingConfig, )

logger = init_logger(__name__)


@dataclass(slots=True)
class _ValidationStepResult:
    videos: list[list[np.ndarray]]
    captions: list[str]


class WanValidator:
    """Phase 2 standalone validator for Wan distillation."""

    def __init__(
        self,
        *,
        training_config: DistillTrainingConfig,
        tracker: Any | None = None,
    ) -> None:
        self.training_config = training_config
        self.tracker = tracker or DummyTracker()

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size

        seed = training_config.data.seed
        if seed is None:
            raise ValueError("training.data.seed must be set for validation")
        self.seed = int(seed)
        self.validation_random_generator = (torch.Generator(device="cpu").manual_seed(self.seed))

        self._pipeline: WanPipeline | None = None
        self._pipeline_key: (tuple[int, str, str] | None) = None
        self._sampling_param: SamplingParam | None = None

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = (SamplingParam.from_pretrained(self.training_config.model_path))
        return self._sampling_param

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
        sampler_kind: str,
        ode_solver: str | None,
    ) -> WanPipeline:
        key = (
            id(transformer),
            str(sampler_kind),
            str(ode_solver),
        )
        if (self._pipeline is not None and self._pipeline_key == key):
            return self._pipeline

        tc = self.training_config
        flow_shift = getattr(tc.pipeline_config, "flow_shift", None)

        kwargs: dict[str, Any] = {
            "inference_mode": True,
            "sampler_kind": str(sampler_kind),
            "loaded_modules": {
                "transformer": transformer
            },
            "tp_size": tc.distributed.tp_size,
            "sp_size": tc.distributed.sp_size,
            "num_gpus": tc.distributed.num_gpus,
            "pin_cpu_memory": tc.distributed.pin_cpu_memory,
            "dit_cpu_offload": True,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if ode_solver is not None:
            kwargs["ode_solver"] = str(ode_solver)

        self._pipeline = WanPipeline.from_pretrained(tc.model_path, **kwargs)
        self._pipeline_key = key
        return self._pipeline

    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
        *,
        sampling_timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
    ) -> ForwardBatch:
        tc = self.training_config

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = tc.data.num_height
        sampling_param.width = tc.data.num_width
        sampling_param.num_inference_steps = (num_inference_steps)
        sampling_param.data_type = "video"
        if guidance_scale is not None:
            sampling_param.guidance_scale = float(guidance_scale)
        sampling_param.seed = self.seed

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = (latents_size[0] * latents_size[1] * latents_size[2])

        temporal_compression_factor = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
        num_frames = ((tc.data.num_latent_t - 1) * temporal_compression_factor + 1)
        sampling_param.num_frames = int(num_frames)

        sampling_timesteps_tensor = (torch.tensor(
            [int(s) for s in sampling_timesteps],
            dtype=torch.long,
        ) if sampling_timesteps is not None else None)

        # Build TrainingArgs for inference to pass
        # to pipeline.forward().
        inference_args = make_inference_args(tc, model_path=tc.model_path)

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=tc.vsa.sparsity,
            timesteps=sampling_timesteps_tensor,
            sampling_timesteps=sampling_timesteps_tensor,
        )
        # Store inference_args on batch for pipeline access.
        batch._inference_args = inference_args  # type: ignore[attr-defined]
        return batch

    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        dataset_file: str,
        transformer: torch.nn.Module,
        sampler_kind: str,
        ode_solver: str | None,
        sampling_timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
    ) -> _ValidationStepResult:
        tc = self.training_config
        pipeline = self._get_pipeline(
            transformer=transformer,
            sampler_kind=sampler_kind,
            ode_solver=ode_solver,
        )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(dataset_file)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

        # Build inference args once for this validation run.
        inference_args = make_inference_args(tc, model_path=tc.model_path)

        videos: list[list[np.ndarray]] = []
        captions: list[str] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
                sampling_timesteps=sampling_timesteps,
                guidance_scale=guidance_scale,
            )

            assert (batch.prompt is not None and isinstance(batch.prompt, str))
            captions.append(batch.prompt)

            with torch.no_grad():
                output_batch = pipeline.forward(batch, inference_args)

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue

            video = rearrange(samples, "b c t h w -> t b c h w")
            frames: list[np.ndarray] = []
            for x in video:
                x = torchvision.utils.make_grid(x, nrow=6)
                x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                frames.append((x * 255).numpy().astype(np.uint8))
            videos.append(frames)

        return _ValidationStepResult(videos=videos, captions=captions)

    def log_validation(
        self,
        step: int,
        *,
        request: ValidationRequest | None = None,
    ) -> None:
        if request is None:
            raise ValueError("WanValidator.log_validation requires a "
                             "ValidationRequest")

        dataset_file = getattr(request, "dataset_file", None)
        if not dataset_file:
            raise ValueError("ValidationRequest.dataset_file must be "
                             "provided by the method")

        guidance_scale = getattr(request, "guidance_scale", None)
        validation_steps = getattr(request, "sampling_steps", None)
        if not validation_steps:
            raise ValueError("ValidationRequest.sampling_steps must be "
                             "provided by the method")
        sampler_kind = (getattr(request, "sampler_kind", None) or "ode")
        rollout_mode = getattr(request, "rollout_mode", None)
        if rollout_mode not in {None, "parallel"}:
            raise ValueError("WanValidator only supports "
                             "rollout_mode='parallel'. "
                             f"Got rollout_mode={rollout_mode!r}.")
        ode_solver = getattr(request, "ode_solver", None)
        sampling_timesteps = getattr(request, "sampling_timesteps", None)
        if sampling_timesteps is not None:
            expected = int(len(sampling_timesteps))
            for steps in validation_steps:
                if int(steps) != expected:
                    raise ValueError("validation_sampling_steps must "
                                     "match "
                                     "len(request.sampling_timesteps)="
                                     f"{expected} when "
                                     "sampling_timesteps is provided, "
                                     f"got {validation_steps!r}.")

        sample_handle = getattr(request, "sample_handle", None)
        if sample_handle is None:
            raise ValueError("ValidationRequest.sample_handle must be "
                             "provided by the method")
        transformer = sample_handle.require_module("transformer")
        was_training = bool(getattr(transformer, "training", False))

        tc = self.training_config
        output_dir = (getattr(request, "output_dir", None) or tc.checkpoint.output_dir)

        try:
            transformer.eval()

            num_sp_groups = (self.world_group.world_size // self.sp_group.world_size)

            for num_inference_steps in validation_steps:
                result = self._run_validation_for_steps(
                    num_inference_steps,
                    dataset_file=str(dataset_file),
                    transformer=transformer,
                    sampler_kind=str(sampler_kind),
                    ode_solver=(str(ode_solver) if ode_solver is not None else None),
                    sampling_timesteps=sampling_timesteps,
                    guidance_scale=guidance_scale,
                )

                if self.rank_in_sp_group != 0:
                    continue

                if self.global_rank == 0:
                    all_videos = list(result.videos)
                    all_captions = list(result.captions)
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = (sp_group_idx * self.sp_world_size)
                        recv_videos = (self.world_group.recv_object(src=src_rank))
                        recv_captions = (self.world_group.recv_object(src=src_rank))
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    os.makedirs(output_dir, exist_ok=True)
                    video_filenames: list[str] = []
                    sampling_param = (self._get_sampling_param())
                    for i, video in enumerate(all_videos):
                        filename = os.path.join(
                            output_dir,
                            f"validation_step_{step}"
                            f"_inference_steps_"
                            f"{num_inference_steps}"
                            f"_video_{i}.mp4",
                        )
                        imageio.mimsave(
                            filename,
                            video,
                            fps=sampling_param.fps,
                        )
                        video_filenames.append(filename)

                    video_logs = []
                    for filename, caption in zip(
                            video_filenames,
                            all_captions,
                            strict=True,
                    ):
                        video_artifact = self.tracker.video(filename, caption=caption)
                        if video_artifact is not None:
                            video_logs.append(video_artifact)
                    if video_logs:
                        logs = {
                            f"validation_videos_"
                            f"{num_inference_steps}"
                            f"_steps": video_logs
                        }
                        self.tracker.log_artifacts(logs, step)
                else:
                    self.world_group.send_object(result.videos, dst=0)
                    self.world_group.send_object(result.captions, dst=0)
        finally:
            if was_training:
                transformer.train()
