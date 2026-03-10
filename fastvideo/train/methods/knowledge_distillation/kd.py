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

from tqdm import tqdm

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
            "trajectory_latents":   Tensor[S, T, C, H, W],  # all S teacher steps
            "trajectory_timesteps": Tensor[S],               # int64
            "real":                 Tensor[T, C, H, W],
            "text_embedding":       Tensor[L, D],
            "text_attention_mask":  Tensor[L],
        }

    The full teacher trajectory is stored so that ``t_list`` can be changed
    without regenerating the cache.  Subsampling to the student timesteps
    happens at training time.
    ``real`` is the teacher's final clean x0 prediction after all ODE steps.
    All latents are in permuted ``[T, C, H, W]`` format (not ``[C, T, H, W]``).
    """

    @staticmethod
    def is_complete(cache_dir: str) -> bool:
        return (Path(cache_dir) / _COMPLETE_SENTINEL).exists()

    @staticmethod
    def validate_or_create_metadata(
        cache_dir: str,
        teacher_id: str,
        teacher_inference_steps: int,
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
                if stored.get("teacher") != teacher_id:
                    raise ValueError(
                        f"Cache teacher {stored['teacher']!r} != config "
                        f"teacher {teacher_id!r}. Delete {cache_dir} "
                        "and re-run.")
                if stored.get("teacher_inference_steps") != teacher_inference_steps:
                    raise ValueError(
                        f"Cache teacher_inference_steps "
                        f"{stored['teacher_inference_steps']} != config "
                        f"{teacher_inference_steps}. Delete {cache_dir} "
                        "and re-run.")
            else:
                meta: dict[str, Any] = {
                    "total_samples": total_samples,
                    "teacher": teacher_id,
                    "teacher_inference_steps": teacher_inference_steps,
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

            # Share the student's VAE with the teacher for latent normalization.
            # The teacher doesn't call init_preprocessors() (non-trainable),
            # so self.teacher.vae is None. Both models use the same Wan VAE
            # constants (latents_mean / latents_std), so sharing is safe.
            if getattr(self.teacher, "vae", None) is None:
                self.teacher.vae = self.student.vae

            source_dataset = self._source_loader.dataset
            total = sum(source_dataset.lengths)
            teacher_id = getattr(self.teacher, "_init_from", "unknown")
            data_path = str(
                getattr(self.training_config.data, "data_path", ""))

            _KDPathCache.validate_or_create_metadata(
                self._cache_dir,
                teacher_id,
                self._teacher_inference_steps,
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

        # Validate that t_list is a subset of the cached trajectory timesteps
        # and build a lookup table used in single_train_step.
        sample0 = _KDPathDataset(self._cache_dir)[0]
        traj_ts = sample0["trajectory_timesteps"].tolist()
        t_to_idx: dict[int, int] = {int(t): i for i, t in enumerate(traj_ts)}
        missing_ts = [t for t in self._t_list[:-1] if t not in t_to_idx]
        if missing_ts:
            raise ValueError(
                f"t_list timesteps {missing_ts} are not present in the cached "
                f"trajectory timesteps.\nCached: {traj_ts}\n"
                f"Adjust t_list to use a subset of the cached timesteps.")
        self._t_to_traj_idx: dict[int, int] = {
            t: t_to_idx[t] for t in self._t_list[:-1]
        }
        logger.info(
            "t_list → trajectory indices: %s",
            {t: self._t_to_traj_idx[t] for t in self._t_list[:-1]},
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

        pbar = tqdm(
            total=len(missing),
            desc="Generating KD cache",
            disable=get_world_rank() != 0,
            unit="sample",
        )
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
                traj_latents, traj_timesteps, real = self._teacher_ode_rollout(raw)
                # traj_latents:   [S, T, C, H, W]
                # traj_timesteps: [S]  int64
                # real:           [T, C, H, W]

            if get_world_rank() == 0:
                out = {
                    "trajectory_latents": traj_latents.cpu(),
                    "trajectory_timesteps": traj_timesteps.cpu(),
                    "real": real.cpu(),
                    "text_embedding": raw["text_embedding"][0].cpu(),
                    "text_attention_mask": raw["text_attention_mask"][0].cpu(),
                }
                torch.save(out, samples_dir / f"{global_idx:08d}.pt")
                pbar.update(1)

            if dist.is_initialized():
                dist.barrier()

        pbar.close()

    def _teacher_ode_rollout(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run high-quality teacher ODE; return full trajectory.

        The teacher runs ``teacher_inference_steps`` ODE steps (e.g. 48),
        saving every intermediate state.  The full trajectory is returned so
        the cache is independent of ``t_list`` — subsampling happens at
        training time.

        Returns:
            trajectory_latents:   ``[S, T, C, H, W]`` — state before each step
            trajectory_timesteps: ``[S]`` int64 — teacher timestep at each state
            real:                 ``[T, C, H, W]`` — teacher's final clean x0
        """
        assert self.teacher is not None
        device = self.student.device
        # bf16 is required by FlashAttention and is consistent with training
        # precision. Models load as float32 but inference always runs in bf16.
        dtype = torch.bfloat16

        # prepare_batch runs text encoding; latents: [1, T, C, H, W]  (B=1)
        training_batch = self.teacher.prepare_batch(
            raw_batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        latents = training_batch.latents.to(dtype)
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

        # Start from pure noise at the highest teacher timestep.
        noise = torch.randn(
            latents.shape,
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )
        t0 = teacher_t_list[0]
        t0_flat = torch.full((B * T,), t0, device=device, dtype=torch.float32)
        x = self._sf_scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            t0_flat,
        ).unflatten(0, (B, T)).to(dtype)  # [B, T, C, H, W]; scheduler may upcast to fp32

        # Run full teacher ODE, recording every state before each step.
        traj_states: list[torch.Tensor] = []  # each [T, C, H, W]
        pred_x0: torch.Tensor | None = None
        teacher_next = teacher_t_list[1:] + [0]

        for t_cur, t_next in zip(teacher_t_list, teacher_next, strict=True):
            traj_states.append(x.squeeze(0).clone())

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
                     sigma_next * eps).unflatten(0, (B, T)).to(dtype)

        assert pred_x0 is not None, "teacher_t_list must have at least one step"

        trajectory_latents = torch.stack(traj_states)  # [S, T, C, H, W]
        trajectory_timesteps = torch.tensor(teacher_t_list, dtype=torch.long)
        real = pred_x0.squeeze(0)  # [T, C, H, W]
        return trajectory_latents, trajectory_timesteps, real

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
        # batch keys: trajectory_latents [B, S, T, C, H, W],
        #             trajectory_timesteps [B, S],
        #             real [B, T, C, H, W],
        #             text_embedding [B, L, D], text_attention_mask [B, L]

        device = self.student.device
        dtype = batch["real"].dtype

        # Randomly select which student denoising step to train on.
        step_i = int(
            torch.randint(
                0,
                self._num_steps,
                (1,),
                generator=self.cuda_generator,
            ).item())

        t = self._t_list[step_i]
        traj_idx = self._t_to_traj_idx[t]
        noisy_input = batch["trajectory_latents"][:, traj_idx].to(device, dtype=dtype)  # [B, T, C, H, W]
        target_x0 = batch["real"].to(device, dtype=dtype)  # [B, T, C, H, W]
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


# ---------------------------------------------------------------------------
# Causal KD Method
# ---------------------------------------------------------------------------


class KDCausalMethod(KDMethod):
    """KD for causal Wan: per-frame block-quantized timestep sampling.

    Identical to :class:`KDMethod` except ``single_train_step`` samples
    a **per-frame** denoising step index (block-quantized to groups of
    ``num_frames_per_block`` frames) instead of one index per batch.
    This matches the legacy ``ODEInitTrainingPipeline`` training scheme
    required by causal / streaming student models.

    Additional YAML field under ``method``::

        num_frames_per_block: 3   # frames sharing the same noise level
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self._num_frames_per_block: int = int(
            self.method_config.get("num_frames_per_block", 3))
        if self._num_frames_per_block < 1:
            raise ValueError("num_frames_per_block must be >= 1")

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
        device = self.student.device
        dtype = batch["real"].dtype

        # trajectory_latents: [B, S, T, C, H, W]
        B, _S, T, C, H, W = batch["trajectory_latents"].shape
        K = self._num_steps  # number of student steps (excludes t=0)

        # Gather the K relevant trajectory steps for t_list[:-1].
        traj_indices = torch.tensor(
            [self._t_to_traj_idx[t] for t in self._t_list[:-1]],
            dtype=torch.long,
            device=device,
        )
        relevant = batch["trajectory_latents"].to(
            device, dtype=dtype)[:, traj_indices]  # [B, K, T, C, H, W]

        # Sample per-frame step indices, block-quantized.
        indexes = self._sample_per_frame_step_idx(B, T, K,
                                                  device)  # [B, T] in [0, K)

        # Gather noisy_input per frame: [B, T, C, H, W]
        noisy_input = torch.gather(
            relevant,
            dim=1,
            index=indexes[:, None, :, None, None, None].expand(
                B, 1, T, C, H, W),
        ).squeeze(1)

        target_x0 = batch["real"].to(device, dtype=dtype)  # [B, T, C, H, W]

        # Per-frame timestep [B, T]
        t_list_tensor = torch.tensor(self._t_list[:-1],
                                     dtype=torch.float32,
                                     device=device)
        timestep_per_frame = t_list_tensor[indexes]  # [B, T]

        proxy_batch = {
            "vae_latent": target_x0.permute(0, 2, 1, 3, 4),
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
        training_batch.timesteps = timestep_per_frame  # [B, T]

        pred_noise = self.student.predict_noise(
            noisy_input,
            timestep_per_frame,
            training_batch,
            conditional=True,
            attn_kind="dense",
        )  # [B, T, C, H, W]

        pred_x0 = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=timestep_per_frame.flatten(0, 1),
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
        metrics: dict[str, LogScalar] = {
            "kd_mean_step_idx": float(indexes.float().mean().item()),
        }
        return loss_map, outputs, metrics

    def _sample_per_frame_step_idx(
        self,
        B: int,
        T: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample per-frame step indices, block-quantized.

        Each group of ``num_frames_per_block`` consecutive frames shares
        the same randomly sampled step index in ``[0, K)``.

        Returns:
            ``[B, T]`` long tensor with values in ``[0, K)``.
        """
        n_blocks = (T + self._num_frames_per_block - 1) // self._num_frames_per_block
        block_idx = torch.randint(
            0,
            K,
            (B, n_blocks),
            generator=self.cuda_generator,
            device=device,
        )  # [B, n_blocks]
        # Expand each block index to num_frames_per_block frames, slice to T.
        per_frame = block_idx.repeat_interleave(
            self._num_frames_per_block, dim=1)[:, :T]
        return per_frame  # [B, T]
