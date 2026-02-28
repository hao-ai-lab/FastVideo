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
from fastvideo.fastvideo_args import ExecutionMode
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.distillation.validators.base import ValidationRequest
from fastvideo.training.trackers import DummyTracker
from fastvideo.utils import shallow_asdict

logger = init_logger(__name__)


@dataclass(slots=True)
class _ValidationStepResult:
    videos: list[list[np.ndarray]]
    captions: list[str]


class WanGameValidator:
    """Standalone validator for WanGame distillation/finetuning."""

    def __init__(
        self,
        *,
        training_args: Any,
        tracker: Any | None = None,
    ) -> None:
        self.training_args = training_args
        self.tracker = tracker or DummyTracker()

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

        self._pipeline: Any | None = None
        self._pipeline_key: tuple[int, str, str, str] | None = None
        self._sampling_param: SamplingParam | None = None

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def _post_process_validation_frames(
        self,
        frames: list[np.ndarray],
        batch: ForwardBatch,
    ) -> list[np.ndarray]:
        """Optionally overlay action indicators on validation frames.

        Mirrors legacy `WanGameTrainingPipeline._post_process_validation_frames()`.
        """

        keyboard_cond = getattr(batch, "keyboard_cond", None)
        mouse_cond = getattr(batch, "mouse_cond", None)
        if keyboard_cond is None and mouse_cond is None:
            return frames

        try:
            from fastvideo.models.dits.matrixgame.utils import (
                draw_keys_on_frame,
                draw_mouse_on_frame,
            )
        except Exception as e:
            logger.warning("WanGame action overlay is unavailable: %s", e)
            return frames

        if keyboard_cond is not None and torch.is_tensor(keyboard_cond):
            keyboard_np = keyboard_cond.squeeze(0).detach().cpu().float().numpy()
        else:
            keyboard_np = None

        if mouse_cond is not None and torch.is_tensor(mouse_cond):
            mouse_np = mouse_cond.squeeze(0).detach().cpu().float().numpy()
        else:
            mouse_np = None

        # MatrixGame convention: keyboard [W, S, A, D, left, right],
        # mouse [Pitch, Yaw].
        key_names = ["W", "S", "A", "D", "left", "right"]

        processed_frames: list[np.ndarray] = []
        for frame_idx, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())

            if keyboard_np is not None and frame_idx < len(keyboard_np):
                keys = {
                    key_names[i]: bool(keyboard_np[frame_idx, i])
                    for i in range(min(len(key_names), int(keyboard_np.shape[1])))
                }
                draw_keys_on_frame(frame, keys, mode="universal")

            if mouse_np is not None and frame_idx < len(mouse_np):
                pitch = float(mouse_np[frame_idx, 0])
                yaw = float(mouse_np[frame_idx, 1])
                draw_mouse_on_frame(frame, pitch, yaw)

            processed_frames.append(frame)

        return processed_frames

    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = SamplingParam.from_pretrained(self.training_args.model_path)
        return self._sampling_param

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
        rollout_mode: str,
        sampler_kind: str,
        ode_solver: str | None,
    ) -> Any:
        rollout_mode = str(rollout_mode).strip().lower()
        sampler_kind = str(sampler_kind).strip().lower()
        key = (id(transformer), rollout_mode, sampler_kind, str(ode_solver))
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline

        if rollout_mode == "parallel":
            if sampler_kind not in {"ode", "sde"}:
                raise ValueError(
                    f"Unknown sampler_kind for WanGame validation: {sampler_kind!r}"
                )

            flow_shift = getattr(self.training_args.pipeline_config, "flow_shift", None)

            from fastvideo.pipelines.basic.wan.wangame_i2v_pipeline import (
                WanGameActionImageToVideoPipeline,
            )

            self._pipeline = WanGameActionImageToVideoPipeline.from_pretrained(
                self.training_args.model_path,
                inference_mode=True,
                flow_shift=float(flow_shift) if flow_shift is not None else None,
                sampler_kind=sampler_kind,
                ode_solver=str(ode_solver) if ode_solver is not None else None,
                loaded_modules={"transformer": transformer},
                tp_size=self.training_args.tp_size,
                sp_size=self.training_args.sp_size,
                num_gpus=self.training_args.num_gpus,
                pin_cpu_memory=self.training_args.pin_cpu_memory,
                dit_cpu_offload=True,
            )
        elif rollout_mode == "streaming":
            from fastvideo.pipelines.basic.wan.wangame_causal_dmd_pipeline import (
                WanGameCausalDMDPipeline,
            )

            self._pipeline = WanGameCausalDMDPipeline.from_pretrained(
                self.training_args.model_path,
                inference_mode=True,
                loaded_modules={"transformer": transformer},
                tp_size=self.training_args.tp_size,
                sp_size=self.training_args.sp_size,
                num_gpus=self.training_args.num_gpus,
                pin_cpu_memory=self.training_args.pin_cpu_memory,
                dit_cpu_offload=True,
            )
        else:
            raise ValueError(
                "Unknown rollout_mode for WanGame validation: "
                f"{rollout_mode!r}. Expected 'parallel' or 'streaming'."
            )

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
        training_args = self.training_args

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch.get("image_path") or validation_batch.get("video_path")
        sampling_param.num_inference_steps = int(num_inference_steps)
        sampling_param.data_type = "video"
        if guidance_scale is not None:
            sampling_param.guidance_scale = float(guidance_scale)
        sampling_param.seed = self.seed

        temporal_compression_factor = (
            training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        num_frames = (training_args.num_latent_t - 1) * temporal_compression_factor + 1
        sampling_param.num_frames = int(num_frames)

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        sampling_timesteps_tensor = (
            torch.tensor([int(s) for s in sampling_timesteps], dtype=torch.long)
            if sampling_timesteps is not None
            else None
        )

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
            timesteps=sampling_timesteps_tensor,
            sampling_timesteps=sampling_timesteps_tensor,
        )
        if "image" in validation_batch and validation_batch["image"] is not None:
            batch.pil_image = validation_batch["image"]

        if "keyboard_cond" in validation_batch and validation_batch["keyboard_cond"] is not None:
            keyboard_cond = validation_batch["keyboard_cond"]
            keyboard_cond = torch.tensor(keyboard_cond, dtype=torch.bfloat16)
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond

        if "mouse_cond" in validation_batch and validation_batch["mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            mouse_cond = torch.tensor(mouse_cond, dtype=torch.bfloat16)
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond

        return batch

    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        dataset_file: str,
        transformer: torch.nn.Module,
        rollout_mode: str,
        sampler_kind: str,
        ode_solver: str | None,
        sampling_timesteps: list[int] | None = None,
        guidance_scale: float | None = None,
    ) -> _ValidationStepResult:
        training_args = self.training_args
        pipeline = self._get_pipeline(
            transformer=transformer,
            rollout_mode=rollout_mode,
            sampler_kind=sampler_kind,
            ode_solver=ode_solver,
        )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(dataset_file)
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
            frames = self._post_process_validation_frames(frames, batch)
            videos.append(frames)

        return _ValidationStepResult(videos=videos, captions=captions)

    def log_validation(self, step: int, *, request: ValidationRequest | None = None) -> None:
        if request is None:
            raise ValueError("WanGameValidator.log_validation requires a ValidationRequest")

        dataset_file = getattr(request, "dataset_file", None)
        if not dataset_file:
            raise ValueError("ValidationRequest.dataset_file must be provided by the method")

        guidance_scale = getattr(request, "guidance_scale", None)
        validation_steps = getattr(request, "sampling_steps", None)
        if not validation_steps:
            raise ValueError("ValidationRequest.sampling_steps must be provided by the method")
        sampler_kind = getattr(request, "sampler_kind", None) or "ode"
        rollout_mode_raw = getattr(request, "rollout_mode", None) or "parallel"
        if not isinstance(rollout_mode_raw, str):
            raise ValueError(
                "ValidationRequest.rollout_mode must be a string when set, got "
                f"{type(rollout_mode_raw).__name__}"
            )
        rollout_mode = rollout_mode_raw.strip().lower()
        if rollout_mode not in {"parallel", "streaming"}:
            raise ValueError(
                "ValidationRequest.rollout_mode must be one of {parallel, streaming}, got "
                f"{rollout_mode_raw!r}"
            )
        ode_solver = getattr(request, "ode_solver", None)
        if rollout_mode == "streaming":
            if str(sampler_kind).strip().lower() != "sde":
                raise ValueError(
                    "WanGame validation rollout_mode='streaming' requires "
                    "sampler_kind='sde' (it uses the causal DMD-style rollout)."
                )
            if ode_solver is not None:
                raise ValueError(
                    "WanGame validation rollout_mode='streaming' does not support "
                    f"ode_solver={ode_solver!r}."
                )
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

        training_args = self.training_args
        output_dir = getattr(request, "output_dir", None) or training_args.output_dir

        old_inference_mode = training_args.inference_mode
        old_dit_cpu_offload = training_args.dit_cpu_offload
        old_mode = training_args.mode
        old_dmd_denoising_steps = getattr(training_args.pipeline_config, "dmd_denoising_steps", None)
        try:
            training_args.inference_mode = True
            training_args.dit_cpu_offload = True
            training_args.mode = ExecutionMode.INFERENCE
            transformer.eval()

            num_sp_groups = self.world_group.world_size // self.sp_group.world_size

            for num_inference_steps in validation_steps:
                if rollout_mode == "streaming":
                    if sampling_timesteps is not None:
                        training_args.pipeline_config.dmd_denoising_steps = list(sampling_timesteps)
                    else:
                        timesteps = np.linspace(1000, 0, int(num_inference_steps))
                        training_args.pipeline_config.dmd_denoising_steps = [
                            int(max(0, min(1000, round(t)))) for t in timesteps
                        ]

                result = self._run_validation_for_steps(
                    num_inference_steps,
                    dataset_file=str(dataset_file),
                    transformer=transformer,
                    rollout_mode=rollout_mode,
                    sampler_kind=str(sampler_kind),
                    ode_solver=str(ode_solver) if ode_solver is not None else None,
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

                    video_logs = []
                    for filename, caption in zip(video_filenames, all_captions, strict=True):
                        video_artifact = self.tracker.video(filename, caption=caption)
                        if video_artifact is not None:
                            video_logs.append(video_artifact)
                    if video_logs:
                        logs = {f"validation_videos_{num_inference_steps}_steps": video_logs}
                        self.tracker.log_artifacts(logs, step)
                else:
                    self.world_group.send_object(result.videos, dst=0)
                    self.world_group.send_object(result.captions, dst=0)
        finally:
            training_args.inference_mode = old_inference_mode
            training_args.dit_cpu_offload = old_dit_cpu_offload
            training_args.mode = old_mode
            training_args.pipeline_config.dmd_denoising_steps = old_dmd_denoising_steps
            if was_training:
                transformer.train()
