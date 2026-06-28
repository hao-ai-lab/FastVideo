# SPDX-License-Identifier: Apache-2.0
"""Validation callback.

All configuration is read from the YAML ``callbacks.validation``
section.  The pipeline class is resolved from
``pipeline_target``.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.api.sampling_param import SamplingParam
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
    contract.
    """

    def __init__(
        self,
        *,
        pipeline_target: str,
        dataset_file: str,
        every_steps: int = 100,
        sampling_steps: list[int] | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        output_dir: str | None = None,
        sampling_timesteps: list[int] | None = None,
        offload_training_state: bool = False,
        unload_pipeline_after_validation: bool = False,
        **pipeline_kwargs: Any,
    ) -> None:
        self.pipeline_target = str(pipeline_target)
        self.dataset_file = str(dataset_file)
        self.every_steps = int(every_steps)
        self.sampling_steps = ([int(s) for s in sampling_steps] if sampling_steps else [40])
        self.guidance_scale = (float(guidance_scale) if guidance_scale is not None else None)
        self.num_frames = (int(num_frames) if num_frames is not None else None)
        self.output_dir = (str(output_dir) if output_dir is not None else None)
        self.sampling_timesteps = ([int(s) for s in sampling_timesteps] if sampling_timesteps is not None else None)
        self.offload_training_state = self._coerce_bool(offload_training_state)
        self.unload_pipeline_after_validation = self._coerce_bool(unload_pipeline_after_validation)
        self.pipeline_kwargs = dict(pipeline_kwargs)

        # Set after on_train_start.
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None
        self._sampling_param: SamplingParam | None = None
        self.tracker: Any = DummyTracker()
        self.validation_random_generator: (torch.Generator | None) = None
        self.seed: int = 0

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

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

        transformer = method.student.transformer
        try:
            with self._validation_memory_context(
                    method,
                    validation_transformer=transformer,
            ):
                # Look for an EMA callback to temporarily swap
                # EMA weights during validation.
                ema_cb = self._find_ema_callback()
                ctx = ema_cb.ema_context(transformer) if ema_cb is not None else contextlib.nullcontext(transformer)
                with ctx as t:
                    self._run_validation_inner(
                        method,
                        step,
                        t,
                    )
        finally:
            if self.unload_pipeline_after_validation:
                self._clear_pipeline_cache()

    @contextlib.contextmanager
    def _validation_memory_context(
        self,
        method: TrainingMethod,
        *,
        validation_transformer: torch.nn.Module,
    ):
        if not self.offload_training_state:
            yield
            return

        optimizer_tensor_records: list[tuple[Any, Any, torch.device]] = []
        module_records: list[tuple[str, torch.nn.Module, torch.device]] = []
        try:
            self._offload_optimizer_states_to_cpu(
                method,
                optimizer_tensor_records,
            )
            self._offload_inactive_role_modules_to_cpu(
                method,
                validation_transformer=validation_transformer,
                module_records=module_records,
            )
            self._empty_cuda_cache()
            yield
        finally:
            self._restore_inactive_role_modules(module_records)
            self._restore_optimizer_states(optimizer_tensor_records)
            self._empty_cuda_cache()

    def _offload_optimizer_states_to_cpu(
        self,
        method: TrainingMethod,
        records: list[tuple[Any, Any, torch.device]],
    ) -> None:
        optimizers = getattr(method, "_optimizer_dict", {})
        if not optimizers:
            return
        moved = 0
        for optimizer in optimizers.values():
            state = getattr(optimizer, "state", None)
            if not isinstance(state, dict):
                continue
            for param_state in state.values():
                moved += self._offload_tensor_container_to_cpu(
                    param_state,
                    records,
                )
        if moved:
            logger.info(
                "Offloaded %d optimizer state tensors to CPU for validation.",
                moved,
            )

    def _offload_tensor_container_to_cpu(
        self,
        obj: Any,
        records: list[tuple[Any, Any, torch.device]],
    ) -> int:
        moved = 0
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if torch.is_tensor(value) and value.device.type == "cuda":
                    records.append((obj, key, value.device))
                    obj[key] = value.detach().cpu()
                    moved += 1
                else:
                    moved += self._offload_tensor_container_to_cpu(value, records)
            return moved
        if isinstance(obj, list):
            for idx, value in enumerate(list(obj)):
                if torch.is_tensor(value) and value.device.type == "cuda":
                    records.append((obj, idx, value.device))
                    obj[idx] = value.detach().cpu()
                    moved += 1
                else:
                    moved += self._offload_tensor_container_to_cpu(value, records)
        return moved

    def _restore_optimizer_states(
        self,
        records: list[tuple[Any, Any, torch.device]],
    ) -> None:
        for container, key, device in reversed(records):
            value = container[key]
            if torch.is_tensor(value):
                container[key] = value.to(device=device)
        if records:
            logger.info(
                "Restored %d optimizer state tensors after validation.",
                len(records),
            )

    def _offload_inactive_role_modules_to_cpu(
        self,
        method: TrainingMethod,
        *,
        validation_transformer: torch.nn.Module,
        module_records: list[tuple[str, torch.nn.Module, torch.device]],
    ) -> None:
        role_models = getattr(method, "_role_models", {})
        if not isinstance(role_models, dict):
            return

        for role, model in role_models.items():
            module = getattr(model, "transformer", None)
            if not isinstance(module, torch.nn.Module):
                continue
            if module is validation_transformer:
                continue
            device = self._first_cuda_tensor_device(module)
            if device is None:
                continue
            try:
                module.to("cpu")
            except Exception as exc:
                logger.warning(
                    "Could not offload role %r transformer to CPU before validation: %s",
                    role,
                    exc,
                )
                continue
            module_records.append((str(role), module, device))
            logger.info(
                "Offloaded role %r transformer from %s to CPU for validation.",
                role,
                device,
            )

    def _restore_inactive_role_modules(
        self,
        module_records: list[tuple[str, torch.nn.Module, torch.device]],
    ) -> None:
        for role, module, device in reversed(module_records):
            module.to(device)
            logger.info(
                "Restored role %r transformer to %s after validation.",
                role,
                device,
            )

    @staticmethod
    def _first_cuda_tensor_device(module: torch.nn.Module) -> torch.device | None:
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            device = getattr(tensor, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                return device
        return None

    def _clear_pipeline_cache(self) -> None:
        self._pipeline = None
        self._pipeline_key = None
        self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_ema_callback(self) -> Any | None:
        """Find the EMA callback in the callback dict."""
        from fastvideo.train.callbacks.ema import (
            EMACallback, )

        cb_dict = getattr(self, "_callback_dict", None)
        if cb_dict is not None:
            for cb in cb_dict._callbacks.values():
                if isinstance(cb, EMACallback):
                    return cb
        return None

    def _run_validation_inner(
        self,
        method: TrainingMethod,
        step: int,
        transformer: torch.nn.Module,
    ) -> None:
        tc = self.training_config
        was_training = bool(getattr(transformer, "training", False))

        output_dir = (self.output_dir or tc.checkpoint.output_dir)

        try:
            transformer.eval()
            num_sp_groups = (self.world_group.world_size // self.sp_group.world_size)

            for num_inference_steps in self.sampling_steps:
                result = self._run_validation_for_steps(
                    num_inference_steps,
                    transformer=transformer,
                )

                if self.rank_in_sp_group != 0:
                    continue

                if self.global_rank == 0:
                    all_videos = list(result.videos)
                    all_captions = list(result.captions)
                    for sp_idx in range(1, num_sp_groups):
                        src = (sp_idx * self.sp_world_size)
                        recv_v = (self.world_group.recv_object(src=src))
                        recv_c = (self.world_group.recv_object(src=src))
                        all_videos.extend(recv_v)
                        all_captions.extend(recv_c)

                    os.makedirs(
                        output_dir,
                        exist_ok=True,
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
                            fname,
                            caption=cap,
                            fps=sp.fps,
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
                else:
                    self.world_group.send_object(
                        result.videos,
                        dst=0,
                    )
                    self.world_group.send_object(
                        result.captions,
                        dst=0,
                    )
        finally:
            if was_training:
                transformer.train()

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
        key = (id(transformer), )
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
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
        }
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        kwargs.update(self.pipeline_kwargs)

        self._pipeline = PipelineCls.from_pretrained(
            tc.model_path,
            **kwargs,
        )

        scheduler = self._pipeline.get_module("scheduler")
        if (scheduler is not None and type(scheduler).__name__ == "SelfForcingFlowMatchScheduler"):
            scheduler.sigma_min = 0.0
            scheduler.extra_one_step = True
            scheduler.set_timesteps(num_inference_steps=1000, training=True)

        self._pipeline_key = key
        return self._pipeline

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
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=tc.vsa_sparsity,
            timesteps=sampling_timesteps_tensor,
        )
        batch._inference_args = inference_args  # type: ignore[attr-defined]

        # Conditionally set I2V fields.
        if ("image" in validation_batch and validation_batch["image"] is not None):
            batch.pil_image = validation_batch["image"]

        return batch

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

        # Propagate sampling_timesteps to pipeline_config so
        # causal/DMD denoising stages can read them.
        if (self.sampling_timesteps is not None and inference_args.pipeline_config.dmd_denoising_steps is None):
            inference_args.pipeline_config.dmd_denoising_steps = ([int(s) for s in self.sampling_timesteps])

        videos: list[list[np.ndarray]] = []
        captions: list[str] = []

        for validation_batch in dataloader:
            batch = self._prepare_validation_batch(
                sampling_param,
                validation_batch,
                num_inference_steps,
            )

            assert (batch.prompt is not None and isinstance(batch.prompt, str))
            captions.append(batch.prompt)

            with torch.no_grad():
                output_batch = pipeline.forward(
                    batch,
                    inference_args,
                )

            samples = output_batch.output.cpu()
            if self.rank_in_sp_group != 0:
                continue

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
            videos.append(frames)

        return _ValidationStepResult(
            videos=videos,
            captions=captions,
        )

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.validation_random_generator is not None:
            state["validation_rng"] = (self.validation_random_generator.get_state())
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        rng_state = state_dict.get("validation_rng")
        if (rng_state is not None and self.validation_random_generator is not None):
            self.validation_random_generator.set_state(rng_state)
