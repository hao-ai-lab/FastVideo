# SPDX-License-Identifier: Apache-2.0
"""Validation callback.

All configuration is read from the YAML ``callbacks.validation``
section.  The pipeline class is resolved from
``pipeline_target``.
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import imageio
import numpy as np
import torch
import torch.distributed as dist
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
    action_paths: list[str | None]


class ValidationCallback(Callback):
    """Generic validation callback driven entirely by YAML
    config.

    Works with any pipeline that follows the
    ``PipelineCls.from_pretrained(...)`` + ``pipeline.forward()``
    contract.
    """

    _FLOW_EVAL_SCALAR_KEYS = (
        "mf_epe_mean",
        "mf_angle_err_mean",
        "mf_cosine_mean",
        "mf_mag_ratio_mean",
        "pixel_epe_mean_mean",
        "px_angle_rmse_mean",
        "fl_all_mean",
        "foe_dist_mean",
        "flow_kl_2d_mean",
    )

    def __init__(
        self,
        *,
        pipeline_target: str,
        dataset_file: str,
        every_steps: int = 100,
        sampling_steps: list[int] | None = None,
        scheduler_target: str | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        output_dir: str | None = None,
        sampling_timesteps: list[int] | None = None,
        evaluate_ptlflow: bool = False,
        ptlflow_dir: str = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/benchmarks/ptlflow",
        ptlflow_ckpt: str | None = None,
        ptlflow_calibration_path: str | None = None,
        ptlflow_use_depth: bool = True,
        **pipeline_kwargs: Any,
    ) -> None:
        self.pipeline_target = str(pipeline_target)
        self.dataset_file = str(dataset_file)
        self.every_steps = int(every_steps)
        self.sampling_steps = ([int(s) for s in sampling_steps] if sampling_steps else [40])

        pipeline_cls = resolve_target(self.pipeline_target)
        self._requires_dmd_denoising_steps = bool(
            getattr(
                pipeline_cls,
                "requires_dmd_denoising_steps",
                False,
            )
        )
        self.scheduler_target = (
            str(scheduler_target)
            if scheduler_target is not None
            else None
        )
        self.guidance_scale = (float(guidance_scale) if guidance_scale is not None else None)
        self.num_frames = (int(num_frames) if num_frames is not None else None)
        self.output_dir = (str(output_dir) if output_dir is not None else None)
        self.sampling_timesteps = ([int(s) for s in sampling_timesteps] if sampling_timesteps is not None else None)

        self.evaluate_ptlflow = bool(evaluate_ptlflow)
        self.ptlflow_dir = str(ptlflow_dir)
        self.ptlflow_ckpt = (
            str(ptlflow_ckpt)
            if ptlflow_ckpt is not None
            else str(Path(self.ptlflow_dir) / "dpflow-things-2012b5d6.ckpt")
        )
        self.ptlflow_calibration_path = (
            str(ptlflow_calibration_path)
            if ptlflow_calibration_path is not None
            else str(Path(self.ptlflow_dir) / "calibration.json")
        )
        self.ptlflow_use_depth = bool(ptlflow_use_depth)
        self.pipeline_kwargs = dict(pipeline_kwargs)

        # Set after on_train_start.
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None
        self._sampling_param: SamplingParam | None = None
        self.tracker: Any = DummyTracker()
        self.validation_random_generator: (torch.Generator | None) = None
        self.seed: int = 0
        self._flow_eval_init_done: bool = False
        self._flow_eval_ready: bool = False
        self._flow_eval_fn: Any | None = None
        self._last_scalar_metrics: dict[str, float] = {}
        self._last_mf_angle_err_mean: float = float("inf")
        self._last_eval_step: int = -1

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
        self.rank_in_sp_group = (self.sp_group.rank_in_group)
        self.sp_world_size = self.sp_group.world_size

        seed = tc.data.seed
        if seed is None:
            raise ValueError("training.data.seed must be set "
                             "for validation")
        self.seed = int(seed)
        self.validation_random_generator = (torch.Generator(device="cpu").manual_seed(self.seed))

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
        self._last_scalar_metrics = {}
        self._last_mf_angle_err_mean = float("inf")
        self._last_eval_step = int(step)
        # Use EMA transformer for validation when available.
        transformer = method.transformer_inference
        was_training = bool(
            getattr(transformer, "training", False)
        )

        output_dir = (
            self.output_dir
            or tc.checkpoint.output_dir
            or os.getcwd()
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
            num_sp_groups = (self.world_group.world_size // self.sp_group.world_size)

            for num_inference_steps in self.sampling_steps:
                self._maybe_set_dmd_denoising_steps(
                    tc,
                    num_inference_steps,
                )

                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                )
                flow_metric_keys = list(
                    self._FLOW_EVAL_SCALAR_KEYS
                )
                local_video_filenames: list[str] = []
                local_video_captions: list[str] = []
                local_metric_sums = {
                    k: 0.0 for k in flow_metric_keys
                }
                local_metric_counts = {
                    k: 0.0 for k in flow_metric_keys
                }
                local_eval_errors: list[str] = []
                flow_eval_enabled = bool(
                    self.evaluate_ptlflow
                )

                if self.rank_in_sp_group == 0:
                    os.makedirs(
                        output_dir, exist_ok=True,
                    )
                    sp = self._get_sampling_param()
                    for i, video in enumerate(result.videos):
                        fname = os.path.join(
                            output_dir,
                            f"validation_step_{step}"
                            f"_inference_steps_"
                            f"{num_inference_steps}"
                            f"_rank_{self.global_rank}"
                            f"_video_{i}.mp4",
                        )
                        imageio.mimsave(
                            fname,
                            video,
                            fps=sp.fps,
                        )
                        local_video_filenames.append(
                            fname
                        )
                    local_video_captions = list(
                        result.captions
                    )

                    if flow_eval_enabled:
                        self._init_flow_eval_module()
                        if not self._flow_eval_ready:
                            logger.warning(
                                "PTLFlow evaluation is "
                                "enabled but evaluator "
                                "initialization failed. "
                                "Skipping flow metrics "
                                "for this validation run."
                            )
                            flow_eval_enabled = False

                    if flow_eval_enabled:
                        for fname, cap, action_path in zip(
                            local_video_filenames,
                            result.captions,
                            result.action_paths,
                            strict=True,
                        ):
                            try:
                                sample_metrics = (
                                    self._evaluate_validation_video(
                                        video_path=fname,
                                        caption=cap,
                                        action_path=action_path,
                                        global_step=step,
                                        num_inference_steps=num_inference_steps,
                                    )
                                )
                                if sample_metrics:
                                    for key in (
                                        flow_metric_keys
                                    ):
                                        val = (
                                            sample_metrics.get(
                                                key
                                            )
                                        )
                                        if not isinstance(
                                            val,
                                            (
                                                float,
                                                int,
                                                np.floating,
                                                np.integer,
                                            ),
                                        ):
                                            continue
                                        val_float = float(
                                            val
                                        )
                                        if not np.isfinite(
                                            val_float
                                        ):
                                            continue
                                        local_metric_sums[
                                            key
                                        ] += val_float
                                        local_metric_counts[
                                            key
                                        ] += 1.0
                            except Exception as e:
                                err = (
                                    "Validation flow "
                                    "evaluation failed "
                                    f"on rank "
                                    f"{self.global_rank} "
                                    f"for {fname}: {e}"
                                )
                                logger.exception(err)
                                local_eval_errors.append(
                                    err
                                )
                            finally:
                                # PTLFlow allocates large CUDA
                                # buffers. Free caches aggressively
                                # so training backward can allocate
                                # cuBLAS handles.
                                self._release_validation_cuda_memory()

                if (
                    self.rank_in_sp_group == 0
                    and self.global_rank == 0
                ):
                    all_video_filenames = list(
                        local_video_filenames
                    )
                    all_captions = list(
                        local_video_captions
                    )
                    for sp_idx in range(
                        1, num_sp_groups
                    ):
                        src = (
                            sp_idx * self.sp_world_size
                        )
                        recv_video_filenames = (
                            self.world_group.recv_object(
                                src=src
                            )
                        )
                        recv_captions = (
                            self.world_group.recv_object(
                                src=src
                            )
                        )
                        all_video_filenames.extend(
                            recv_video_filenames
                        )
                        all_captions.extend(
                            recv_captions
                        )

                    video_logs = []
                    for fname, cap in zip(
                        all_video_filenames,
                        all_captions,
                        strict=True,
                    ):
                        art = self.tracker.video(
                            fname,
                            caption=cap,
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
                            logs,
                            step,
                        )
                elif self.rank_in_sp_group == 0:
                    self.world_group.send_object(
                        local_video_filenames, dst=0,
                    )
                    self.world_group.send_object(
                        local_video_captions, dst=0,
                    )

                use_cuda_tensor = (
                    dist.is_available()
                    and dist.is_initialized()
                    and dist.get_backend() == "nccl"
                    and torch.cuda.is_available()
                )
                tensor_device = (
                    torch.device("cuda")
                    if use_cuda_tensor
                    else torch.device("cpu")
                )

                if (
                    dist.is_available()
                    and dist.is_initialized()
                ):
                    eval_failed = torch.tensor(
                        [
                            1
                            if local_eval_errors
                            else 0
                        ],
                        device=tensor_device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(
                        eval_failed,
                        op=dist.ReduceOp.MAX,
                    )
                    if int(eval_failed.item()) > 0:
                        if local_eval_errors:
                            raise RuntimeError(
                                "Validation flow "
                                "evaluation failed:\n"
                                + "\n".join(
                                    local_eval_errors
                                )
                            )
                        raise RuntimeError(
                            "Validation flow "
                            "evaluation failed on at "
                            "least one rank. Check "
                            "per-rank logs."
                        )
                elif local_eval_errors:
                    raise RuntimeError(
                        "Validation flow evaluation "
                        "failed:\n"
                        + "\n".join(local_eval_errors)
                    )

                if not self.evaluate_ptlflow:
                    continue

                metric_sum_tensor = torch.zeros(
                    len(flow_metric_keys),
                    device=tensor_device,
                    dtype=torch.float64,
                )
                metric_count_tensor = torch.zeros(
                    len(flow_metric_keys),
                    device=tensor_device,
                    dtype=torch.float64,
                )
                if (
                    self.rank_in_sp_group == 0
                    and flow_eval_enabled
                ):
                    metric_sum_tensor = torch.tensor(
                        [
                            local_metric_sums[k]
                            for k in flow_metric_keys
                        ],
                        device=tensor_device,
                        dtype=torch.float64,
                    )
                    metric_count_tensor = torch.tensor(
                        [
                            local_metric_counts[k]
                            for k in flow_metric_keys
                        ],
                        device=tensor_device,
                        dtype=torch.float64,
                    )

                if (
                    dist.is_available()
                    and dist.is_initialized()
                ):
                    dist.all_reduce(
                        metric_sum_tensor,
                        op=dist.ReduceOp.SUM,
                    )
                    dist.all_reduce(
                        metric_count_tensor,
                        op=dist.ReduceOp.SUM,
                    )

                if self.global_rank == 0:
                    metric_logs: dict[str, float] = {}
                    for i, metric_key in enumerate(
                        flow_metric_keys
                    ):
                        count = float(
                            metric_count_tensor[
                                i
                            ].item()
                        )
                        if count <= 0:
                            continue
                        value = float(
                            metric_sum_tensor[
                                i
                            ].item()
                            / count
                        )
                        if not np.isfinite(value):
                            continue
                        metric_logs[
                            f"metrics/{metric_key}"
                        ] = value
                        self._last_scalar_metrics[
                            metric_key
                        ] = value

                    if metric_logs:
                        self.tracker.log(
                            metric_logs, step
                        )

                    mf_val = metric_logs.get(
                        "metrics/mf_angle_err_mean"
                    )
                    if mf_val is not None:
                        self._last_mf_angle_err_mean = (
                            float(mf_val)
                        )
        finally:
            self._release_validation_cuda_memory()
            if hasattr(tc.pipeline_config, "dmd_denoising_steps"):
                tc.pipeline_config.dmd_denoising_steps = (
                    old_dmd_denoising_steps
                )
            if was_training:
                transformer.train()

        if (
            dist.is_available()
            and dist.is_initialized()
        ):
            backend = dist.get_backend()
            use_cuda = (
                backend == "nccl"
                and torch.cuda.is_available()
            )
            tensor_device = (
                torch.device("cuda")
                if use_cuda
                else torch.device("cpu")
            )
            mf_tensor = torch.tensor(
                [self._last_mf_angle_err_mean],
                dtype=torch.float32,
                device=tensor_device,
            )
            dist.broadcast(mf_tensor, src=0)
            self._last_mf_angle_err_mean = float(
                mf_tensor.item()
            )
            if np.isfinite(
                self._last_mf_angle_err_mean
            ):
                self._last_scalar_metrics[
                    "mf_angle_err_mean"
                ] = self._last_mf_angle_err_mean

    def _maybe_set_dmd_denoising_steps(
        self,
        tc: TrainingConfig,
        num_inference_steps: int,
    ) -> None:
        """Set dmd_denoising_steps on pipeline_config for
        pipelines that require explicit DMD timesteps."""
        if not self._requires_dmd_denoising_steps:
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
            self._sampling_param = (SamplingParam.from_pretrained(self.training_config.model_path))
        return self._sampling_param

    def _get_pipeline(
        self,
        *,
        transformer: torch.nn.Module,
    ) -> Any:
        key = (
            id(transformer),
            self.pipeline_target,
            self.scheduler_target,
        )
        if (self._pipeline is not None and self._pipeline_key == key):
            return self._pipeline

        tc = self.training_config
        PipelineCls = resolve_target(self.pipeline_target)
        flow_shift = getattr(
            tc.pipeline_config,
            "flow_shift",
            None,
        )

        kwargs: dict[str, Any] = {
            "inference_mode": True,
            "loaded_modules": {
                "transformer": transformer,
            },
            "tp_size": tc.distributed.tp_size,
            "sp_size": tc.distributed.sp_size,
            "num_gpus": tc.distributed.num_gpus,
            "pin_cpu_memory": (tc.distributed.pin_cpu_memory),
            "dit_cpu_offload": True,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)

        # Build and inject a scheduler if target is set.
        scheduler = self._build_scheduler(flow_shift)
        if scheduler is not None:
            kwargs["loaded_modules"]["scheduler"] = (
                scheduler
            )

        self._pipeline = PipelineCls.from_pretrained(
            tc.model_path,
            **kwargs,
        )
        self._pipeline_key = key
        return self._pipeline

    def _build_scheduler(
        self, flow_shift: float | None,
    ) -> Any | None:
        """Build scheduler from ``scheduler_target``."""
        if self.scheduler_target is None:
            return None
        if flow_shift is None:
            return None

        SchedulerCls = resolve_target(
            self.scheduler_target
        )
        return SchedulerCls(shift=float(flow_shift))

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
        sampling_param.num_inference_steps = int(num_inference_steps)
        sampling_param.data_type = "video"
        if self.guidance_scale is not None:
            sampling_param.guidance_scale = float(self.guidance_scale)
        sampling_param.seed = self.seed

        # image_path for I2V pipelines.
        img_path = (validation_batch.get("image_path") or validation_batch.get("video_path"))
        if img_path is not None and (img_path.startswith("http") or os.path.isfile(img_path)):
            sampling_param.image_path = img_path

        temporal_compression_factor = int(
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        default_num_frames = ((tc.data.num_latent_t - 1) * temporal_compression_factor + 1)
        if self.num_frames is not None:
            sampling_param.num_frames = int(self.num_frames)
        else:
            sampling_param.num_frames = int(default_num_frames)

        latents_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = (latents_size[0] * latents_size[1] * latents_size[2])

        sampling_timesteps_tensor = (torch.tensor(
            [int(s) for s in self.sampling_timesteps],
            dtype=torch.long,
        ) if self.sampling_timesteps is not None else None)

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=tc.vsa_sparsity,
            timesteps=sampling_timesteps_tensor,
            sampling_timesteps=sampling_timesteps_tensor,
        )
        batch._inference_args = inference_args  # type: ignore[attr-defined]

        # Conditionally set I2V fields.
        if ("image" in validation_batch and validation_batch["image"] is not None):
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
        pipeline = self._get_pipeline(transformer=transformer, )
        sampling_param = self._get_sampling_param()

        dataset = ValidationDataset(self.dataset_file)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        inference_args = make_inference_args(
            tc,
            model_path=tc.model_path,
        )

        videos: list[list[np.ndarray]] = []
        captions: list[str] = []
        action_paths: list[str | None] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
            )
            action_path = validation_batch.get(
                "action_path"
            )
            if not isinstance(action_path, str):
                action_path = None

            assert (batch.prompt is not None and isinstance(batch.prompt, str))

            with torch.no_grad():
                output_batch = pipeline.forward(
                    batch,
                    inference_args,
                )

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue
            captions.append(batch.prompt)
            action_paths.append(action_path)

            video = rearrange(
                samples,
                "b c t h w -> t b c h w",
            )
            frames: list[np.ndarray] = []
            for x in video:
                x = torchvision.utils.make_grid(
                    x,
                    nrow=6,
                )
                x = (x.transpose(0, 1).transpose(1, 2).squeeze(-1))
                frames.append((x * 255).numpy().astype(np.uint8))
            frames = (
                self._post_process_validation_frames(
                    frames, batch,
                )
            )
            videos.append(frames)

        return _ValidationStepResult(
            videos=videos,
            captions=captions,
            action_paths=action_paths,
        )

    def _init_flow_eval_module(self) -> None:
        if self._flow_eval_init_done:
            return
        self._flow_eval_init_done = True
        self._flow_eval_ready = False

        ptlflow_dir = Path(self.ptlflow_dir)
        try:
            ptlflow_dir_str = str(ptlflow_dir.resolve())
            if ptlflow_dir_str not in sys.path:
                sys.path.insert(0, ptlflow_dir_str)

            from eval_flow_divergence import (
                evaluate_pair_synthetic, )

            self._flow_eval_fn = (
                evaluate_pair_synthetic
            )
            self._flow_eval_ready = True
            logger.info(
                "Initialized flow divergence "
                "evaluator: %s",
                ptlflow_dir_str,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize flow "
                "divergence evaluator: %s",
                e,
            )

    def _count_video_frames(
        self,
        video_path: str,
    ) -> int:
        reader: Any | None = None
        try:
            reader = imageio.get_reader(video_path)
            try:
                return int(reader.count_frames())
            except Exception:
                n_frames = 0
                for _ in reader:
                    n_frames += 1
                return int(n_frames)
        except Exception as e:
            logger.warning(
                "Failed to count frames for "
                "validation video %s: %s",
                video_path,
                e,
            )
            return 0
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

    def _prepare_aligned_action_path(
        self,
        *,
        action_path: str,
        video_path: str,
        eval_output_dir: str,
    ) -> str:
        try:
            actions_obj = np.load(
                action_path,
                allow_pickle=True,
            ).item()
        except Exception as e:
            logger.warning(
                "Failed to load action file "
                "%s: %s",
                action_path,
                e,
            )
            return action_path

        if not isinstance(actions_obj, dict):
            logger.warning(
                "Action file %s is not a dict; "
                "skipping action alignment.",
                action_path,
            )
            return action_path

        keyboard = actions_obj.get("keyboard")
        mouse = actions_obj.get("mouse")
        if keyboard is None or mouse is None:
            logger.warning(
                "Action file %s is missing "
                "'keyboard' or 'mouse'; "
                "skipping action alignment.",
                action_path,
            )
            return action_path

        keyboard_arr = np.asarray(keyboard)
        mouse_arr = np.asarray(mouse)
        if (
            keyboard_arr.ndim < 1
            or mouse_arr.ndim < 1
        ):
            logger.warning(
                "Action file %s has invalid "
                "shapes keyboard=%s mouse=%s; "
                "skipping action alignment.",
                action_path,
                keyboard_arr.shape,
                mouse_arr.shape,
            )
            return action_path

        n_frames = self._count_video_frames(video_path)
        if n_frames <= 0:
            return action_path

        n_keyboard = int(keyboard_arr.shape[0])
        n_mouse = int(mouse_arr.shape[0])
        n_use = min(n_frames, n_keyboard, n_mouse)
        if n_use <= 0:
            return action_path

        if (
            n_use == n_keyboard
            and n_use == n_mouse
        ):
            return action_path

        aligned_actions = dict(actions_obj)
        aligned_actions["keyboard"] = keyboard_arr[:n_use]
        aligned_actions["mouse"] = mouse_arr[:n_use]

        os.makedirs(eval_output_dir, exist_ok=True)
        aligned_path = os.path.join(
            eval_output_dir,
            "actions_aligned.npy",
        )
        np.save(
            aligned_path,
            aligned_actions,
            allow_pickle=True,
        )
        logger.info(
            "Aligned action file for PTLFlow: "
            "video_frames=%d keyboard=%d "
            "mouse=%d used=%d path=%s",
            n_frames,
            n_keyboard,
            n_mouse,
            n_use,
            aligned_path,
        )
        return aligned_path

    def _release_validation_cuda_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _evaluate_validation_video(
        self,
        *,
        video_path: str,
        caption: str,
        action_path: str | None,
        global_step: int,
        num_inference_steps: int,
    ) -> dict[str, float]:
        del caption
        self._init_flow_eval_module()
        if not self._flow_eval_ready:
            raise RuntimeError(
                "ptlflow evaluator is not "
                "initialized; cannot compute "
                "flow metrics."
            )
        if (
            not isinstance(action_path, str)
            or not os.path.isfile(action_path)
        ):
            raise FileNotFoundError(
                "Validation sample is missing a "
                f"valid action_path: {action_path}"
            )

        output_root = (
            self.output_dir
            or self.training_config.checkpoint.output_dir
            or os.getcwd()
        )
        eval_output_dir = os.path.join(
            output_root,
            "flow_eval",
            f"step_{global_step}",
            f"inference_steps_{num_inference_steps}",
            Path(video_path).stem,
        )

        aligned_action_path = (
            self._prepare_aligned_action_path(
                action_path=action_path,
                video_path=video_path,
                eval_output_dir=eval_output_dir,
            )
        )
        with torch.inference_mode():
            summary = self._flow_eval_fn(
                gen_video=video_path,
                action_path=aligned_action_path,
                calibration_path=self.ptlflow_calibration_path,
                output_dir=eval_output_dir,
                model_name="dpflow",
                ckpt=self.ptlflow_ckpt,
                no_viz=True,
                use_depth=self.ptlflow_use_depth,
            )
        if not isinstance(summary, dict):
            raise RuntimeError(
                "ptlflow returned invalid summary "
                f"type: {type(summary)}"
            )

        metrics: dict[str, float] = {}
        missing_or_invalid: list[str] = []
        for key in self._FLOW_EVAL_SCALAR_KEYS:
            value = summary.get(key)
            if not isinstance(
                value,
                (
                    float,
                    int,
                    np.floating,
                    np.integer,
                ),
            ):
                missing_or_invalid.append(key)
                continue
            value_float = float(value)
            if not np.isfinite(value_float):
                missing_or_invalid.append(key)
                continue
            metrics[key] = value_float

        if missing_or_invalid:
            raise RuntimeError(
                "ptlflow summary missing/invalid "
                "metrics: "
                + ", ".join(missing_or_invalid)
            )
        return metrics

    def get_latest_metric(
        self,
        metric_key: str,
        *,
        step: int | None = None,
    ) -> float | None:
        if (
            step is not None
            and int(step) != self._last_eval_step
        ):
            return None
        value = self._last_scalar_metrics.get(
            str(metric_key)
        )
        if value is None:
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.validation_random_generator is not None:
            state["validation_rng"] = (self.validation_random_generator.get_state())

        state["last_eval_step"] = int(
            self._last_eval_step
        )
        state["last_mf_angle_err_mean"] = float(
            self._last_mf_angle_err_mean
        )
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        rng_state = state_dict.get("validation_rng")
        if (rng_state is not None and self.validation_random_generator is not None):
            self.validation_random_generator.set_state(rng_state)

        last_step = state_dict.get("last_eval_step")
        if isinstance(last_step, int):
            self._last_eval_step = int(last_step)
        last_mf = state_dict.get(
            "last_mf_angle_err_mean"
        )
        if isinstance(
            last_mf, (float, int, np.floating, np.integer)
        ):
            self._last_mf_angle_err_mean = float(
                last_mf
            )
            if np.isfinite(self._last_mf_angle_err_mean):
                self._last_scalar_metrics[
                    "mf_angle_err_mean"
                ] = self._last_mf_angle_err_mean
