# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.distillation.validators.base import ValidationRequest
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.utils import shallow_asdict

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
        training_args: Any,
        tracker: Any,
    ) -> None:
        self.training_args = training_args
        self.tracker = tracker

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size

        seed = getattr(training_args, "seed", None)
        if seed is None:
            raise ValueError("training_args.seed must be set for validation")
        self.seed = int(seed)
        self.validation_random_generator = torch.Generator(device="cpu").manual_seed(self.seed)

        self._pipeline: WanPipeline | None = None
        self._pipeline_key: tuple[int, str] | None = None
        self._sampling_param: SamplingParam | None = None

    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = SamplingParam.from_pretrained(self.training_args.model_path)
        return self._sampling_param

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
        sampler_kind: str,
    ) -> WanPipeline:
        key = (id(transformer), str(sampler_kind))
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline

        # NOTE: `ComposedPipelineBase.from_pretrained()` ignores `args` when
        # `inference_mode=True`, so we must pass pipeline knobs via kwargs.
        flow_shift = getattr(self.training_args.pipeline_config, "flow_shift", None)

        self._pipeline = WanPipeline.from_pretrained(
            self.training_args.model_path,
            inference_mode=True,
            sampler_kind=str(sampler_kind),
            flow_shift=float(flow_shift) if flow_shift is not None else None,
            loaded_modules={"transformer": transformer},
            tp_size=self.training_args.tp_size,
            sp_size=self.training_args.sp_size,
            num_gpus=self.training_args.num_gpus,
            pin_cpu_memory=self.training_args.pin_cpu_memory,
            dit_cpu_offload=True,
        )
        self._pipeline_key = key
        return self._pipeline

    def _parse_validation_steps(self) -> list[int]:
        raw = str(getattr(self.training_args, "validation_sampling_steps", "") or "")
        steps = [int(s) for s in raw.split(",") if s.strip()]
        return [s for s in steps if s > 0]

    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
        *,
        sampling_timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
    ) -> ForwardBatch:
        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = self.training_args.num_height
        sampling_param.width = self.training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        if guidance_scale is not None:
            sampling_param.guidance_scale = float(guidance_scale)
        elif getattr(self.training_args, "validation_guidance_scale", ""):
            sampling_param.guidance_scale = float(self.training_args.validation_guidance_scale)
        sampling_param.seed = self.seed

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        temporal_compression_factor = (
            self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        num_frames = (self.training_args.num_latent_t - 1) * temporal_compression_factor + 1
        sampling_param.num_frames = int(num_frames)

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=self.training_args.VSA_sparsity,
            sampling_timesteps=(
                torch.tensor([int(s) for s in sampling_timesteps], dtype=torch.long)
                if sampling_timesteps is not None
                else None
            ),
        )
        return batch

    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        transformer: torch.nn.Module,
        sampler_kind: str,
        sampling_timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
    ) -> _ValidationStepResult:
        training_args = self.training_args
        pipeline = self._get_pipeline(transformer=transformer, sampler_kind=sampler_kind)
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(training_args.validation_dataset_file)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

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

            assert batch.prompt is not None and isinstance(batch.prompt, str)
            captions.append(batch.prompt)

            with torch.no_grad():
                output_batch = pipeline.forward(batch, training_args)

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

    def log_validation(self, step: int, *, request: ValidationRequest | None = None) -> None:
        training_args = self.training_args
        if not getattr(training_args, "log_validation", False):
            return
        if not getattr(training_args, "validation_dataset_file", ""):
            raise ValueError("validation_dataset_file must be set when log_validation is enabled")

        guidance_scale = getattr(request, "guidance_scale", None)
        validation_steps = getattr(request, "sampling_steps", None) or self._parse_validation_steps()
        if not validation_steps:
            return
        sampler_kind = getattr(request, "sampler_kind", None) or "ode"
        sampling_timesteps = getattr(request, "sampling_timesteps", None)
        if sampling_timesteps is not None:
            expected = int(len(sampling_timesteps))
            for steps in validation_steps:
                if int(steps) != expected:
                    raise ValueError(
                        "validation_sampling_steps must match "
                        f"len(request.sampling_timesteps)={expected} when "
                        "sampling_timesteps is provided, got "
                        f"{validation_steps!r}."
                    )

        sample_handle = getattr(request, "sample_handle", None)
        if sample_handle is None:
            raise ValueError("ValidationRequest.sample_handle must be provided by the method")
        transformer = sample_handle.require_module("transformer")
        was_training = bool(getattr(transformer, "training", False))

        output_dir = getattr(request, "output_dir", None) or training_args.output_dir

        old_inference_mode = training_args.inference_mode
        old_dit_cpu_offload = training_args.dit_cpu_offload
        try:
            training_args.inference_mode = True
            training_args.dit_cpu_offload = True
            transformer.eval()

            num_sp_groups = self.world_group.world_size // self.sp_group.world_size

            for num_inference_steps in validation_steps:
                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                    sampler_kind=str(sampler_kind),
                    sampling_timesteps=sampling_timesteps,
                    guidance_scale=guidance_scale,
                )

                if self.rank_in_sp_group != 0:
                    continue

                if self.global_rank == 0:
                    all_videos = list(result.videos)
                    all_captions = list(result.captions)
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size
                        recv_videos = self.world_group.recv_object(src=src_rank)
                        recv_captions = self.world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    os.makedirs(output_dir, exist_ok=True)
                    video_filenames: list[str] = []
                    sampling_param = self._get_sampling_param()
                    for i, video in enumerate(all_videos):
                        filename = os.path.join(
                            output_dir,
                            f"validation_step_{step}_inference_steps_{num_inference_steps}_video_{i}.mp4",
                        )
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    artifacts = []
                    for filename, caption in zip(video_filenames, all_captions, strict=True):
                        video_artifact = self.tracker.video(filename, caption=caption)
                        if video_artifact is not None:
                            artifacts.append(video_artifact)
                    if artifacts:
                        logs = {f"validation_videos_{num_inference_steps}_steps": artifacts}
                        self.tracker.log_artifacts(logs, step)
                else:
                    self.world_group.send_object(result.videos, dst=0)
                    self.world_group.send_object(result.captions, dst=0)
        finally:
            training_args.inference_mode = old_inference_mode
            training_args.dit_cpu_offload = old_dit_cpu_offload
            if was_training:
                transformer.train()
