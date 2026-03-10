# SPDX-License-Identifier: Apache-2.0
"""Knowledge Distillation method for ODE-init training.

Trains a student model with MSE loss to reproduce a teacher model's
multi-step ODE denoising trajectories. The resulting checkpoint
(exported via dcp_to_diffusers) serves as the ``ode_init`` weight
initialization for downstream Self-Forcing training.

Teacher path generation is cached to disk so it only runs once.
Interrupted generation resumes from the last completed sample.

Typical YAML::

    models:
      student:
        _target_: fastvideo.train.models.wan.WanModel
        init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        trainable: true
      teacher:           # omit once cache is complete
        _target_: fastvideo.train.models.wan.WanModel
        init_from: Wan-AI/Wan2.1-T2V-14B-Diffusers
        trainable: false
        disable_custom_init_weights: true

    method:
      _target_: fastvideo.train.methods.knowledge_distillation.kd.KDMethod
      teacher_path_cache: /data/kd_cache/wan14b_4step
      t_list: [999, 937, 833, 624, 0]   # integer timesteps
      student_sample_steps: 4
      teacher_guidance_scale: 1.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler
from fastvideo.distributed import (
    get_sp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
    get_sp_group,
)
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler, )
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_COMPLETE_SENTINEL = "COMPLETE"
_METADATA_FILE = "metadata.json"


class _KDPathCache:
    """Utility class for KD teacher-path on-disk cache management.

    Cache layout::

        <cache_dir>/
        ├── metadata.json       # config written before generation starts
        ├── samples/
        │   ├── 00000000.pt     # one file per source dataset sample
        │   ├── 00000001.pt
        │   └── ...
        └── COMPLETE            # sentinel written only when all samples done

    Each ``.pt`` file contains a dict::

        {
            "path":                 Tensor[num_steps, T, C, H, W],
            "real":                 Tensor[T, C, H, W],
            "text_embedding":       Tensor[L, D],
            "text_attention_mask":  Tensor[L],
            "path_timesteps":       Tensor[num_steps],  # int64, == t_list[:-1]
        }

    ``path[i]`` is the teacher's ODE state at the teacher trajectory step
    whose timestep exactly equals ``path_timesteps[i]`` (i.e. ``t_list[i]``).
    ``t_list`` must be a subset of the teacher's ODE schedule — an error is
    raised at generation time if any student timestep is not present.
    ``real`` is the teacher's final clean x0 prediction after all ODE steps.
    All latents are in permuted ``[T, C, H, W]`` format (not ``[C, T, H, W]``).
    """

    @staticmethod
    def is_complete(cache_dir: str) -> bool:
        return (Path(cache_dir) / _COMPLETE_SENTINEL).exists()

    @staticmethod
    def validate_or_create_metadata(
        cache_dir: str,
        t_list: list[int],
        teacher_id: str,
        total_samples: int,
        source_data_path: str,
    ) -> None:
        """Write metadata if absent; raise if present but mismatched."""
        meta_path = Path(cache_dir) / _METADATA_FILE
        if get_world_rank() == 0:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            if meta_path.exists():
                with open(meta_path) as f:
                    stored = json.load(f)
                if stored.get("t_list") != t_list:
                    raise ValueError(
                        f"Cache t_list {stored['t_list']} != config "
                        f"t_list {t_list}. Delete {cache_dir} and re-run.")
                if stored.get("teacher") != teacher_id:
                    raise ValueError(
                        f"Cache teacher {stored['teacher']!r} != config "
                        f"teacher {teacher_id!r}. Delete {cache_dir} "
                        "and re-run.")
            else:
                meta: dict[str, Any] = {
                    "total_samples": total_samples,
                    "t_list": t_list,
                    "teacher": teacher_id,
                    "source_data_path": source_data_path,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def find_missing(cache_dir: str, total_samples: int) -> list[int]:
        """Return sorted list of sample indices not yet on disk."""
        samples_dir = Path(cache_dir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        existing = {int(f.stem) for f in samples_dir.glob("*.pt")}
        return [i for i in range(total_samples) if i not in existing]

    @staticmethod
    def mark_complete(cache_dir: str) -> None:
        """Write COMPLETE sentinel (rank 0 only) and barrier."""
        if get_world_rank() == 0:
            (Path(cache_dir) / _COMPLETE_SENTINEL).touch()
        if dist.is_initialized():
            dist.barrier()


# ---------------------------------------------------------------------------
# Dataset for reading cached paths at training time
# ---------------------------------------------------------------------------


class _KDPathDataset(Dataset):
    """Reads per-sample ``.pt`` files from a completed KD cache.

    Returns dicts with keys:
        - ``path``                 [num_steps, T, C, H, W]
        - ``real``                 [T, C, H, W]
        - ``text_embedding``       [L, D]
        - ``text_attention_mask``  [L]
        - ``path_timesteps``       [num_steps]  int64
    """

    def __init__(self, cache_dir: str) -> None:
        self._samples_dir = Path(cache_dir) / "samples"
        meta_path = Path(cache_dir) / _METADATA_FILE
        with open(meta_path) as f:
            meta = json.load(f)
        self._total: int = int(meta["total_samples"])

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self._samples_dir / f"{idx:08d}.pt"
        return torch.load(path, weights_only=True, map_location="cpu")


# ---------------------------------------------------------------------------
# Lazy dataloader wrapper
# ---------------------------------------------------------------------------


class _KDDataLoaderWrapper:
    """Wraps the parquet dataloader initially; activates to KD cache after generation.

    The builder captures ``student.dataloader`` after ``KDMethod.__init__``
    runs. At that point this wrapper is the stored value.  The trainer
    calls ``on_train_start()`` before iterating, so by the time
    ``__iter__`` is called the wrapper has been activated and yields
    from the KD cache dataloader.
    """

    def __init__(self, source_loader: Any) -> None:
        self._source = source_loader
        self._active: Any = None

    def activate(
        self,
        cache_dir: str,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> None:
        from torchdata.stateful_dataloader import StatefulDataLoader
        from fastvideo.platforms import current_platform

        dataset = _KDPathDataset(cache_dir)
        sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=len(dataset),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=True,
            seed=seed,
        )
        self._active = StatefulDataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=_kd_collate,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=current_platform.device_name,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self):
        if self._active is None:
            raise RuntimeError("_KDDataLoaderWrapper has not been "
                               "activated. Ensure on_train_start() "
                               "completed successfully.")
        return iter(self._active)

    def state_dict(self):
        if self._active is not None and hasattr(self._active, "state_dict"):
            return self._active.state_dict()
        return {}

    def load_state_dict(self, state: dict) -> None:
        if self._active is not None and hasattr(self._active,
                                                "load_state_dict"):
            self._active.load_state_dict(state)


def _kd_collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batched dict."""
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


# ---------------------------------------------------------------------------
# KD Method
# ---------------------------------------------------------------------------


class KDMethod(TrainingMethod):
    """Knowledge Distillation training method.

    Trains the student with MSE loss on teacher ODE trajectories cached
    to ``method_config.teacher_path_cache``.

    Roles:
        - ``student`` (required, trainable): the model being distilled.
        - ``teacher`` (optional, non-trainable): used to generate the cache
          on first run; freed from GPU memory afterwards.

    If the cache is incomplete and no teacher is configured, an error
    is raised at the start of training.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("KDMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError("KDMethod requires student to be trainable")

        mcfg = self.method_config

        # --- Parse method config ---
        raw_t_list = mcfg.get("t_list")
        if not isinstance(raw_t_list, list) or not raw_t_list:
            raise ValueError("method_config.t_list must be a non-empty list "
                             "of integer timestep values, e.g. "
                             "[999, 937, 833, 624, 0]")
        self._t_list: list[int] = [int(t) for t in raw_t_list]

        raw_steps = mcfg.get("student_sample_steps")
        if raw_steps is None:
            raw_steps = len(self._t_list) - 1
        self._num_steps: int = int(raw_steps)
        if len(self._t_list) != self._num_steps + 1:
            raise ValueError(
                f"len(t_list)={len(self._t_list)} must equal "
                f"student_sample_steps+1={self._num_steps + 1}")

        cache_dir = mcfg.get("teacher_path_cache")
        if not cache_dir:
            raise ValueError("method_config.teacher_path_cache must be set")
        self._cache_dir: str = str(cache_dir)

        self._teacher_guidance_scale: float = float(
            mcfg.get("teacher_guidance_scale", 1.0))
        self._teacher_inference_steps: int = int(
            mcfg.get("teacher_inference_steps", 48))

        # --- Optional teacher ---
        self.teacher: ModelBase | None = role_models.get("teacher")
        if self.teacher is not None and getattr(self.teacher, "_trainable",
                                                False):
            raise ValueError(
                "KDMethod requires teacher to be non-trainable "
                "(set trainable: false in models.teacher)")

        # --- Build parquet dataloader via student.init_preprocessors ---
        self.student.init_preprocessors(self.training_config)

        # Wrap the parquet dataloader so we can swap it after generation.
        self._source_loader = self.student.dataloader
        self._kd_wrapper = _KDDataLoaderWrapper(self._source_loader)
        self.student.dataloader = self._kd_wrapper  # builder captures this

        # --- Build SelfForcingFlowMatchScheduler for sigma lookups ---
        # num_inference_steps=1000 gives a dense grid accurate for any t.
        tc = self.training_config
        self._flow_shift = float(
            getattr(tc.pipeline_config, "flow_shift", 0.0) or 0.0)
        self._sf_scheduler = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=int(self.student.num_train_timesteps),
            shift=self._flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
            training=False,
        )

        # --- Student optimizer / scheduler (same as FineTuneMethod) ---
        self._init_optimizers_and_schedulers()

    # ------------------------------------------------------------------
    # TrainingMethod abstract implementations
    # ------------------------------------------------------------------

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        super().on_train_start()  # sets CUDA RNG, calls student.on_train_start()

        if not _KDPathCache.is_complete(self._cache_dir):
            if self.teacher is None:
                raise RuntimeError(
                    f"teacher_path_cache at {self._cache_dir!r} is "
                    "incomplete and no teacher model is configured. "
                    "Add a 'teacher' entry under 'models:' in the YAML "
                    "or provide a complete cache directory.")

            # Prepare teacher for inference (world_group needed for
            # negative conditioning setup).
            self.teacher.world_group = get_world_group()
            self.teacher.sp_group = get_sp_group()
            self.teacher.on_train_start()

            source_dataset = self._source_loader.dataset
            total = sum(source_dataset.lengths)
            teacher_id = getattr(self.teacher, "_init_from", "unknown")
            data_path = str(
                getattr(self.training_config.data, "data_path", ""))

            _KDPathCache.validate_or_create_metadata(
                self._cache_dir,
                self._t_list,
                teacher_id,
                total,
                data_path,
            )

            missing = _KDPathCache.find_missing(self._cache_dir, total)
            if missing:
                logger.info(
                    "Generating KD teacher paths: %d/%d samples missing. "
                    "Cache: %s",
                    len(missing),
                    total,
                    self._cache_dir,
                )
                self._generate_paths(source_dataset, missing, total)

            _KDPathCache.mark_complete(self._cache_dir)
            logger.info("KD cache complete: %s", self._cache_dir)

        # Free teacher from GPU memory — not needed after cache is built.
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
            self._role_models.pop("teacher", None)
            if "teacher" in self.role_modules:
                del self.role_modules["teacher"]
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()

        # Activate the KD cache dataloader.
        tc = self.training_config
        self._kd_wrapper.activate(
            cache_dir=self._cache_dir,
            batch_size=int(tc.data.train_batch_size),
            num_workers=int(tc.data.dataloader_num_workers),
            seed=int(tc.data.seed),
        )

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def _generate_paths(
        self,
        source_dataset: Any,
        missing: list[int],
        total: int,
    ) -> None:
        """Generate and cache teacher ODE paths for all missing indices.

        All ranks participate in each teacher forward pass (required by
        FSDP). Only rank 0 writes files. A dist.barrier() follows each
        save to keep ranks in sync.
        """
        missing_set = set(missing)
        samples_dir = Path(self._cache_dir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        device = self.student.device

        for global_idx in range(total):
            if global_idx not in missing_set:
                continue  # all ranks skip together — no FSDP divergence

            # Load single sample by stable global index.
            raw = source_dataset.__getitems__([global_idx])
            raw = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }

            with torch.no_grad():
                path, real = self._teacher_ode_rollout(raw)
                # path: [num_steps, T, C, H, W]
                # real: [T, C, H, W]

            if get_world_rank() == 0:
                out = {
                    "path": path.cpu(),
                    "real": real.cpu(),
                    "text_embedding": raw["text_embedding"][0].cpu(),
                    "text_attention_mask":
                    raw["text_attention_mask"][0].cpu(),
                    "path_timesteps":
                    torch.tensor(self._t_list[:-1], dtype=torch.long),
                }
                torch.save(out, samples_dir / f"{global_idx:08d}.pt")
                if (global_idx + 1) % 100 == 0 or global_idx == total - 1:
                    logger.info("  cached %d / %d samples", global_idx + 1,
                                total)

            if dist.is_initialized():
                dist.barrier()

    def _teacher_ode_rollout(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run high-quality teacher ODE; return (path, real).

        The teacher runs ``teacher_inference_steps`` ODE steps (e.g. 48).
        Each timestep in ``t_list[:-1]`` must appear exactly in the teacher's
        schedule — if not, an error is raised listing the teacher's timesteps
        so the user can align ``t_list`` accordingly.

        Only the subsampled states at ``t_list[:-1]`` are returned; the full
        trajectory is not stored.

        Returns:
            path: ``[num_steps, T, C, H, W]``
            real: ``[T, C, H, W]`` — teacher's final clean x0
        """
        assert self.teacher is not None
        device = self.student.device

        # prepare_batch runs text encoding; latents: [1, T, C, H, W]  (B=1)
        training_batch = self.teacher.prepare_batch(
            raw_batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        latents = training_batch.latents
        B, T = latents.shape[:2]

        # Build teacher's full ODE schedule.
        teacher_sched = SelfForcingFlowMatchScheduler(
            num_inference_steps=self._teacher_inference_steps,
            num_train_timesteps=int(self.student.num_train_timesteps),
            shift=self._flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
            training=False,
        )
        teacher_t_list = [int(t.item()) for t in teacher_sched.timesteps]
        teacher_t_set = set(teacher_t_list)

        # Validate that every student timestep is in the teacher schedule.
        missing = [t for t in self._t_list[:-1] if t not in teacher_t_set]
        if missing:
            raise ValueError(
                f"t_list timesteps {missing} are not present in the teacher's "
                f"{self._teacher_inference_steps}-step ODE schedule.\n"
                f"Teacher timesteps: {teacher_t_list}\n"
                f"Adjust t_list to use a subset of the teacher's timesteps, "
                f"or change teacher_inference_steps.")

        # Start from pure noise at the highest teacher timestep.
        noise = torch.randn(
            latents.shape,
            device=device,
            dtype=latents.dtype,
            generator=self.cuda_generator,
        )
        t0 = teacher_t_list[0]
        t0_flat = torch.full((B * T,), t0, device=device, dtype=torch.float32)
        x = self._sf_scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            t0_flat,
        ).unflatten(0, (B, T))  # [B, T, C, H, W]

        # Run full teacher ODE; snapshot states at student timesteps only.
        student_t_set = set(self._t_list[:-1])
        path_by_t: dict[int, torch.Tensor] = {}
        pred_x0: torch.Tensor | None = None
        teacher_next = teacher_t_list[1:] + [0]

        for t_cur, t_next in zip(teacher_t_list, teacher_next, strict=True):
            if t_cur in student_t_set:
                path_by_t[t_cur] = x.squeeze(0).clone()  # [T, C, H, W]

            timestep_b = torch.full((B,), t_cur, device=device,
                                    dtype=torch.float32)
            training_batch.timesteps = timestep_b

            pred_noise = self.teacher.predict_noise(
                x,
                timestep_b,
                training_batch,
                conditional=True,
                attn_kind="dense",
            )
            timestep_bt = timestep_b.unsqueeze(1).expand(B, T)
            pred_x0 = pred_noise_to_pred_video(
                pred_noise=pred_noise.flatten(0, 1),
                noise_input_latent=x.flatten(0, 1),
                timestep=timestep_bt,
                scheduler=self._sf_scheduler,
            ).unflatten(0, (B, T))

            if t_next > 0:
                sigma_cur = self._timestep_to_sigma(t_cur, B * T, device)
                sigma_next = self._timestep_to_sigma(t_next, B * T, device)
                x_flat = x.flatten(0, 1)
                p_flat = pred_x0.flatten(0, 1)
                eps = ((x_flat - (1.0 - sigma_cur) * p_flat) /
                       sigma_cur.clamp_min(1e-8))
                x = ((1.0 - sigma_next) * p_flat +
                     sigma_next * eps).unflatten(0, (B, T))

        assert pred_x0 is not None, "teacher_t_list must have at least one step"
        real = pred_x0.squeeze(0)  # [T, C, H, W]

        path = torch.stack(
            [path_by_t[t] for t in self._t_list[:-1]])  # [num_steps, T, C, H, W]
        return path, real

    def _timestep_to_sigma(
        self,
        timestep_val: int,
        numel: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return sigma for a scalar timestep value, broadcast to [numel, 1, 1, 1]."""
        sigmas = self._sf_scheduler.sigmas.to(device=device, dtype=torch.float32)
        timesteps = self._sf_scheduler.timesteps.to(device=device, dtype=torch.float32)
        t = torch.tensor([float(timestep_val)], device=device)
        idx = torch.argmin((timesteps - t).abs())
        return sigmas[idx].expand(numel).reshape(numel, 1, 1, 1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del iteration
        # batch keys: path [B, S, T, C, H, W], real [B, T, C, H, W],
        #             text_embedding [B, L, D], text_attention_mask [B, L],
        #             path_timesteps [B, S]

        device = self.student.device
        dtype = batch["real"].dtype

        # Randomly select which denoising step to train on.
        step_i = int(
            torch.randint(
                0,
                self._num_steps,
                (1,),
                generator=self.cuda_generator,
            ).item())

        noisy_input = batch["path"][:, step_i].to(device, dtype=dtype)  # [B, T, C, H, W]
        target_x0 = batch["real"].to(device, dtype=dtype)  # [B, T, C, H, W]
        t = int(batch["path_timesteps"][0, step_i].item())
        B, T = noisy_input.shape[:2]

        # Build a proxy batch so prepare_batch sets up text conditioning and
        # attention metadata. We pass "real" as vae_latent (permuted back to
        # [B, C, T, H, W] format that prepare_batch expects), then override
        # the noisy input and timestep below.
        proxy_batch = {
            "vae_latent": target_x0.permute(0, 2, 1, 3, 4),  # [B, C, T, H, W]
            "text_embedding": batch["text_embedding"].to(device, dtype=dtype),
            "text_attention_mask":
            batch["text_attention_mask"].to(device, dtype=dtype),
        }
        training_batch = self.student.prepare_batch(
            proxy_batch,
            generator=self.cuda_generator,
            current_vsa_sparsity=current_vsa_sparsity,
            latents_source="data",
        )

        # Override timestep with the actual KD timestep.
        timestep_b = torch.full((B,), float(t), device=device, dtype=torch.float32)
        training_batch.timesteps = timestep_b

        # Student forward: predict noise from the cached noisy latent.
        pred_noise = self.student.predict_noise(
            noisy_input,
            timestep_b,
            training_batch,
            conditional=True,
            attn_kind="dense",
        )  # [B, T, C, H, W]

        # Convert predicted noise to predicted x0 using _sf_scheduler.
        timestep_bt = timestep_b.unsqueeze(1).expand(B, T)
        pred_x0 = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=timestep_bt,
            scheduler=self._sf_scheduler,
        ).unflatten(0, (B, T))  # [B, T, C, H, W]

        loss = 0.5 * F.mse_loss(pred_x0.float(), target_x0.float())

        loss_map: dict[str, torch.Tensor] = {
            "total_loss": loss,
            "kd_loss": loss,
        }
        outputs: dict[str, Any] = {
            "_fv_backward": (
                training_batch.timesteps,
                training_batch.attn_metadata,
            )
        }
        metrics: dict[str, LogScalar] = {"kd_step_idx": float(step_i)}
        return loss_map, outputs, metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        ctx = outputs.get("_fv_backward")
        if ctx is None:
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return
        self.student.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0")

        student_params = [
            p for p in self.student.transformer.parameters()
            if p.requires_grad
        ]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )
