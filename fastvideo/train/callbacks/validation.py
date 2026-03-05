# SPDX-License-Identifier: Apache-2.0
"""Validation callback (unified replacement for WanValidator
and WanGameValidator).

All configuration is read from the YAML ``callbacks.validation``
section.  The pipeline class is resolved from
``pipeline_target``.
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
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.instantiate import resolve_target
from fastvideo.train.utils.moduleloader import (
    make_inference_args, )
from fastvideo.training.trackers import DummyTracker
from fastvideo.utils import shallow_asdict

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

logger = init_logger(__name__)


@dataclass(slots=True)
class _ValidationStepResult:
    videos: list[list[np.ndarray]]
    captions: list[str]


class ValidationCallback(Callback):
    """Generic validation callback driven entirely by YAML
    config.

    Works with any pipeline that follows the
    ``PipelineCls.from_pretrained(...)`` + ``pipeline.forward()``
    contract (Wan, WanGame parallel, WanGame causal/DMD, etc.).
    """

    def __init__(
        self,
        *,
        pipeline_target: str,
        dataset_file: str,
        every_steps: int = 100,
        sampling_steps: list[int] | None = None,
        sampler_kind: str = "ode",
        ode_solver: str | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        output_dir: str | None = None,
        sampling_timesteps: list[int] | None = None,
        rollout_mode: str = "parallel",
        **pipeline_kwargs: Any,
    ) -> None:
        self.pipeline_target = str(pipeline_target)
        self.dataset_file = str(dataset_file)
        self.every_steps = int(every_steps)
        self.sampling_steps = (
            [int(s) for s in sampling_steps]
            if sampling_steps
            else [40]
        )
        self.sampler_kind = str(sampler_kind)
        self.ode_solver = (
            str(ode_solver) if ode_solver is not None
            else None
        )
        self.guidance_scale = (
            float(guidance_scale)
            if guidance_scale is not None
            else None
        )
        self.num_frames = (
            int(num_frames) if num_frames is not None
            else None
        )
        self.output_dir = (
            str(output_dir) if output_dir is not None
            else None
        )
        self.sampling_timesteps = (
            [int(s) for s in sampling_timesteps]
            if sampling_timesteps is not None
            else None
        )
        self.rollout_mode = str(rollout_mode)
        self.pipeline_kwargs = dict(pipeline_kwargs)

        # Set after on_train_start.
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None
        self._sampling_param: SamplingParam | None = None
        self.tracker: Any = DummyTracker()
        self.validation_random_generator: (
            torch.Generator | None
        ) = None
        self.seed: int = 0

    # ----------------------------------------------------------
    # Callback hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        self.method = method
        tc = self.training_config

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.rank_in_sp_group = (
            self.sp_group.rank_in_group
        )
        self.sp_world_size = self.sp_group.world_size

        seed = tc.data.seed
        if seed is None:
            raise ValueError(
                "training.data.seed must be set "
                "for validation"
            )
        self.seed = int(seed)
        self.validation_random_generator = (
            torch.Generator(device="cpu").manual_seed(
                self.seed
            )
        )

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            self.tracker = tracker

    def on_validation_begin(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        if self.every_steps <= 0:
            return
        if iteration % self.every_steps != 0:
            return

        self._run_validation(method, iteration)

    # ----------------------------------------------------------
    # Core validation logic
    # ----------------------------------------------------------

    def _run_validation(
        self,
        method: TrainingMethod,
        step: int,
    ) -> None:
        tc = self.training_config
        transformer = method.student.transformer
        was_training = bool(
            getattr(transformer, "training", False)
        )

        output_dir = (
            self.output_dir
            or tc.checkpoint.output_dir
        )

        # For streaming SDE pipelines we may need to
        # temporarily set dmd_denoising_steps on
        # pipeline_config.
        old_dmd_denoising_steps = getattr(
            tc.pipeline_config,
            "dmd_denoising_steps",
            None,
        )
        try:
            transformer.eval()
            num_sp_groups = (
                self.world_group.world_size
                // self.sp_group.world_size
            )

            for num_inference_steps in self.sampling_steps:
                self._maybe_set_dmd_denoising_steps(
                    tc,
                    num_inference_steps,
                )

                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                )

                if self.rank_in_sp_group != 0:
                    continue

                if self.global_rank == 0:
                    all_videos = list(result.videos)
                    all_captions = list(result.captions)
                    for sp_idx in range(
                        1, num_sp_groups
                    ):
                        src = (
                            sp_idx * self.sp_world_size
                        )
                        recv_v = (
                            self.world_group.recv_object(
                                src=src
                            )
                        )
                        recv_c = (
                            self.world_group.recv_object(
                                src=src
                            )
                        )
                        all_videos.extend(recv_v)
                        all_captions.extend(recv_c)

                    os.makedirs(
                        output_dir, exist_ok=True,
                    )
                    video_filenames: list[str] = []
                    sp = self._get_sampling_param()
                    for i, video in enumerate(all_videos):
                        fname = os.path.join(
                            output_dir,
                            f"validation_step_{step}"
                            f"_inference_steps_"
                            f"{num_inference_steps}"
                            f"_video_{i}.mp4",
                        )
                        imageio.mimsave(
                            fname,
                            video,
                            fps=sp.fps,
                        )
                        video_filenames.append(fname)

                    video_logs = []
                    for fname, cap in zip(
                        video_filenames,
                        all_captions,
                        strict=True,
                    ):
                        art = self.tracker.video(
                            fname, caption=cap,
                        )
                        if art is not None:
                            video_logs.append(art)
                    if video_logs:
                        logs = {
                            f"validation_videos_"
                            f"{num_inference_steps}"
                            f"_steps": video_logs
                        }
                        self.tracker.log_artifacts(
                            logs, step,
                        )
                else:
                    self.world_group.send_object(
                        result.videos, dst=0,
                    )
                    self.world_group.send_object(
                        result.captions, dst=0,
                    )
        finally:
            if hasattr(tc.pipeline_config, "dmd_denoising_steps"):
                tc.pipeline_config.dmd_denoising_steps = (
                    old_dmd_denoising_steps
                )
            if was_training:
                transformer.train()

    def _maybe_set_dmd_denoising_steps(
        self,
        tc: TrainingConfig,
        num_inference_steps: int,
    ) -> None:
        """Set dmd_denoising_steps on pipeline_config for
        streaming SDE validation."""
        if self.rollout_mode != "streaming":
            return
        if self.sampler_kind != "sde":
            return
        if self.sampling_timesteps is not None:
            tc.pipeline_config.dmd_denoising_steps = (  # type: ignore[union-attr]
                list(self.sampling_timesteps)
            )
        else:
            timesteps = np.linspace(
                1000, 0, int(num_inference_steps),
            )
            tc.pipeline_config.dmd_denoising_steps = [  # type: ignore[union-attr]
                int(max(0, min(1000, round(t))))
                for t in timesteps
            ]

        # Also set any pipeline-specific kwargs from
        # YAML (e.g. dmd_denoising_steps override).
        pk = self.pipeline_kwargs
        if "dmd_denoising_steps" in pk:
            tc.pipeline_config.dmd_denoising_steps = [  # type: ignore[union-attr]
                int(s)
                for s in pk["dmd_denoising_steps"]
            ]

    # ----------------------------------------------------------
    # Pipeline management
    # ----------------------------------------------------------

    def _get_sampling_param(self) -> SamplingParam:
        if self._sampling_param is None:
            self._sampling_param = (
                SamplingParam.from_pretrained(
                    self.training_config.model_path
                )
            )
        return self._sampling_param

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
    ) -> Any:
        key = (
            id(transformer),
            self.rollout_mode,
            self.sampler_kind,
            str(self.ode_solver),
        )
        if (
            self._pipeline is not None
            and self._pipeline_key == key
        ):
            return self._pipeline

        tc = self.training_config
        PipelineCls = resolve_target(self.pipeline_target)
        flow_shift = getattr(
            tc.pipeline_config, "flow_shift", None,
        )

        kwargs: dict[str, Any] = {
            "inference_mode": True,
            "sampler_kind": self.sampler_kind,
            "loaded_modules": {
                "transformer": transformer,
            },
            "tp_size": tc.distributed.tp_size,
            "sp_size": tc.distributed.sp_size,
            "num_gpus": tc.distributed.num_gpus,
            "pin_cpu_memory": (
                tc.distributed.pin_cpu_memory
            ),
            "dit_cpu_offload": True,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if self.ode_solver is not None:
            kwargs["ode_solver"] = str(self.ode_solver)

        # For streaming pipelines, build and inject a
        # scheduler if needed.
        if self.rollout_mode == "streaming":
            scheduler = self._build_streaming_scheduler(
                flow_shift,
            )
            if scheduler is not None:
                kwargs["loaded_modules"]["scheduler"] = (
                    scheduler
                )

        self._pipeline = PipelineCls.from_pretrained(
            tc.model_path, **kwargs,
        )
        self._pipeline_key = key
        return self._pipeline

    def _build_streaming_scheduler(
        self, flow_shift: float | None,
    ) -> Any | None:
        """Build scheduler for streaming validation."""
        if flow_shift is None:
            return None

        if self.sampler_kind == "sde":
            from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler, )

            return FlowMatchEulerDiscreteScheduler(
                shift=float(flow_shift),
            )

        # ODE mode — choose based on ode_solver.
        ode_solver_norm = (
            str(self.ode_solver).strip().lower()
            if self.ode_solver is not None
            else "unipc"
        )
        if ode_solver_norm in {
            "unipc",
            "unipc_multistep",
            "multistep",
        }:
            from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
                FlowUniPCMultistepScheduler, )

            return FlowUniPCMultistepScheduler(
                shift=float(flow_shift),
            )
        if ode_solver_norm in {
            "euler",
            "flowmatch",
            "flowmatch_euler",
        }:
            from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler, )

            return FlowMatchEulerDiscreteScheduler(
                shift=float(flow_shift),
            )

        raise ValueError(
            f"Unknown ode_solver for streaming "
            f"validation: {self.ode_solver!r}"
        )

    # ----------------------------------------------------------
    # Batch preparation
    # ----------------------------------------------------------

    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
    ) -> ForwardBatch:
        tc = self.training_config

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = tc.data.num_height
        sampling_param.width = tc.data.num_width
        sampling_param.num_inference_steps = int(
            num_inference_steps
        )
        sampling_param.data_type = "video"
        if self.guidance_scale is not None:
            sampling_param.guidance_scale = float(
                self.guidance_scale
            )
        sampling_param.seed = self.seed

        # image_path for I2V pipelines.
        img_path = (
            validation_batch.get("image_path")
            or validation_batch.get("video_path")
        )
        if img_path is not None:
            sampling_param.image_path = img_path

        temporal_compression_factor = int(
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        default_num_frames = (
            (tc.data.num_latent_t - 1)
            * temporal_compression_factor
            + 1
        )
        if self.num_frames is not None:
            sampling_param.num_frames = int(
                self.num_frames
            )
        else:
            sampling_param.num_frames = int(
                default_num_frames
            )

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = (
            latents_size[0]
            * latents_size[1]
            * latents_size[2]
        )

        sampling_timesteps_tensor = (
            torch.tensor(
                [int(s) for s in self.sampling_timesteps],
                dtype=torch.long,
            )
            if self.sampling_timesteps is not None
            else None
        )

        inference_args = make_inference_args(
            tc, model_path=tc.model_path,
        )

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
        batch._inference_args = inference_args  # type: ignore[attr-defined]

        # Conditionally set I2V / WanGame fields.
        if (
            "image" in validation_batch
            and validation_batch["image"] is not None
        ):
            batch.pil_image = validation_batch["image"]

        self._maybe_set_action_conds(
            batch, validation_batch, sampling_param,
        )
        return batch

    def _maybe_set_action_conds(
        self,
        batch: ForwardBatch,
        validation_batch: dict[str, Any],
        sampling_param: SamplingParam,
    ) -> None:
        """Set keyboard_cond / mouse_cond on the batch if
        present in the dataset."""
        target_len = int(sampling_param.num_frames)

        if (
            "keyboard_cond" in validation_batch
            and validation_batch["keyboard_cond"]
            is not None
        ):
            kb = torch.as_tensor(
                validation_batch["keyboard_cond"]
            ).to(dtype=torch.bfloat16)
            if kb.ndim == 3 and kb.shape[0] == 1:
                kb = kb.squeeze(0)
            if kb.ndim != 2:
                raise ValueError(
                    "validation keyboard_cond must have"
                    " shape (T, K), got "
                    f"{tuple(kb.shape)}"
                )
            if kb.shape[0] > target_len:
                kb = kb[:target_len]
            elif kb.shape[0] < target_len:
                pad = torch.zeros(
                    (
                        target_len - kb.shape[0],
                        kb.shape[1],
                    ),
                    dtype=kb.dtype,
                    device=kb.device,
                )
                kb = torch.cat([kb, pad], dim=0)
            batch.keyboard_cond = kb.unsqueeze(0)

        if (
            "mouse_cond" in validation_batch
            and validation_batch["mouse_cond"]
            is not None
        ):
            mc = torch.as_tensor(
                validation_batch["mouse_cond"]
            ).to(dtype=torch.bfloat16)
            if mc.ndim == 3 and mc.shape[0] == 1:
                mc = mc.squeeze(0)
            if mc.ndim != 2:
                raise ValueError(
                    "validation mouse_cond must have "
                    "shape (T, 2), got "
                    f"{tuple(mc.shape)}"
                )
            if mc.shape[0] > target_len:
                mc = mc[:target_len]
            elif mc.shape[0] < target_len:
                pad = torch.zeros(
                    (
                        target_len - mc.shape[0],
                        mc.shape[1],
                    ),
                    dtype=mc.dtype,
                    device=mc.device,
                )
                mc = torch.cat([mc, pad], dim=0)
            batch.mouse_cond = mc.unsqueeze(0)

    # ----------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------

    def _post_process_validation_frames(
        self,
        frames: list[np.ndarray],
        batch: ForwardBatch,
    ) -> list[np.ndarray]:
        """Overlay action indicators if conditions present."""
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
            logger.warning(
                "Action overlay unavailable: %s", e,
            )
            return frames

        if (
            keyboard_cond is not None
            and torch.is_tensor(keyboard_cond)
        ):
            keyboard_np = (
                keyboard_cond.squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
        else:
            keyboard_np = None

        if (
            mouse_cond is not None
            and torch.is_tensor(mouse_cond)
        ):
            mouse_np = (
                mouse_cond.squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
        else:
            mouse_np = None

        key_names = ["W", "S", "A", "D", "left", "right"]
        processed: list[np.ndarray] = []
        for fi, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())
            if (
                keyboard_np is not None
                and fi < len(keyboard_np)
            ):
                keys = {
                    key_names[i]: bool(
                        keyboard_np[fi, i]
                    )
                    for i in range(
                        min(
                            len(key_names),
                            int(keyboard_np.shape[1]),
                        )
                    )
                }
                draw_keys_on_frame(
                    frame, keys, mode="universal",
                )
            if (
                mouse_np is not None
                and fi < len(mouse_np)
            ):
                pitch = float(mouse_np[fi, 0])
                yaw = float(mouse_np[fi, 1])
                draw_mouse_on_frame(frame, pitch, yaw)
            processed.append(frame)
        return processed

    # ----------------------------------------------------------
    # Validation loop
    # ----------------------------------------------------------

    def _run_validation_for_steps(
        self,
        num_inference_steps: int,
        *,
        transformer: torch.nn.Module,
    ) -> _ValidationStepResult:
        tc = self.training_config
        pipeline = self._get_pipeline(
            transformer=transformer,
        )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(self.dataset_file)
        dataloader = DataLoader(
            dataset, batch_size=None, num_workers=0,
        )

        inference_args = make_inference_args(
            tc, model_path=tc.model_path,
        )

        videos: list[list[np.ndarray]] = []
        captions: list[str] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
            )

            assert (
                batch.prompt is not None
                and isinstance(batch.prompt, str)
            )
            captions.append(batch.prompt)

            with torch.no_grad():
                output_batch = pipeline.forward(
                    batch, inference_args,
                )

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue

            video = rearrange(
                samples, "b c t h w -> t b c h w",
            )
            frames: list[np.ndarray] = []
            for x in video:
                x = torchvision.utils.make_grid(
                    x, nrow=6,
                )
                x = (
                    x.transpose(0, 1)
                    .transpose(1, 2)
                    .squeeze(-1)
                )
                frames.append(
                    (x * 255).numpy().astype(np.uint8)
                )
            frames = (
                self._post_process_validation_frames(
                    frames, batch,
                )
            )
            videos.append(frames)

        return _ValidationStepResult(
            videos=videos, captions=captions,
        )

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.validation_random_generator is not None:
            state["validation_rng"] = (
                self.validation_random_generator.get_state()
            )
        return state

    def load_state_dict(
        self, state_dict: dict[str, Any],
    ) -> None:
        rng_state = state_dict.get("validation_rng")
        if (
            rng_state is not None
            and self.validation_random_generator is not None
        ):
            self.validation_random_generator.set_state(
                rng_state
            )
