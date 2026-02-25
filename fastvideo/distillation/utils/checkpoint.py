# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from fastvideo.distillation.roles import RoleHandle, RoleManager
from fastvideo.logger import init_logger
from fastvideo.training.checkpointing_utils import (
    ModelWrapper,
    OptimizerWrapper,
    RandomStateWrapper,
    SchedulerWrapper,
)

logger = init_logger(__name__)


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(
        getattr(obj, "load_state_dict", None)
    )


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _parse_step_from_dir(checkpoint_dir: Path) -> int:
    match = _CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
    if not match:
        raise ValueError(
            f"Invalid checkpoint directory name {checkpoint_dir.name!r}; "
            "expected 'checkpoint-<step>'"
        )
    return int(match.group(1))


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _CHECKPOINT_DIR_RE.match(child.name):
            continue
        if not (child / "dcp").is_dir():
            continue
        try:
            step = _parse_step_from_dir(child)
        except Exception:
            continue
        candidates.append((step, child))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_resume_checkpoint(resume_from_checkpoint: str, *, output_dir: str) -> Path:
    """Resolve a user-provided resume path to a concrete checkpoint dir.

    Accepted values:
    - /path/to/output_dir/checkpoint-<step>
    - /path/to/output_dir/checkpoint-<step>/dcp
    - /path/to/output_dir (auto-pick latest checkpoint-*/dcp)
    """

    raw = os.path.expanduser(str(resume_from_checkpoint))
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint not found: {path}")

    if path.is_dir() and path.name == "dcp":
        path = path.parent

    if path.is_dir() and _CHECKPOINT_DIR_RE.match(path.name):
        if not (path / "dcp").is_dir():
            raise FileNotFoundError(f"Missing dcp dir under checkpoint: {path / 'dcp'}")
        return path

    # Treat as output_dir -> pick latest.
    latest = _find_latest_checkpoint(path)
    if latest is not None:
        return latest

    # Give a clearer error message.
    out = Path(os.path.expanduser(str(output_dir))).resolve()
    raise ValueError(
        "Could not resolve resume checkpoint. Expected a checkpoint directory "
        f"named 'checkpoint-<step>' (with 'dcp/' inside), or an output_dir "
        f"containing such checkpoints. Got: {path} (output_dir={out})."
    )


class _RoleModuleContainer(torch.nn.Module):
    """Ephemeral container to expose multiple role modules as a single module.

    Needed because `OptimizerWrapper` expects a single root module covering all
    parameters owned by the optimizer.
    """

    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)


@dataclass(slots=True)
class DistillCheckpointConfig:
    save_steps: int
    keep_last: int


class DistillCheckpointManager:
    """Role-based checkpoint manager for Phase 2 distillation runtime.

    - Checkpoint policy lives in YAML (via TrainingArgs fields).
    - Resume path is typically provided via CLI (`--resume-from-checkpoint`).
    """

    def __init__(
        self,
        *,
        bundle: RoleManager,
        dataloader: Any,
        output_dir: str,
        config: DistillCheckpointConfig,
        get_rng_generators: Callable[[], dict[str, torch.Generator]] | None = None,
    ) -> None:
        self.bundle = bundle
        self.dataloader = dataloader
        self.output_dir = str(output_dir)
        self.config = config
        self._get_rng_generators = get_rng_generators
        self._last_saved_step: int | None = None

    def _build_role_states(self, role: str, handle: RoleHandle) -> dict[str, Any]:
        if not handle.trainable:
            return {}

        states: dict[str, Any] = {}
        container = _RoleModuleContainer(handle.modules)

        for module_name, module in handle.modules.items():
            states[f"roles.{role}.{module_name}"] = ModelWrapper(module)

        for name, optimizer in handle.optimizers.items():
            states[f"optimizers.{role}.{name}"] = OptimizerWrapper(container, optimizer)

        for name, scheduler in handle.lr_schedulers.items():
            states[f"schedulers.{role}.{name}"] = SchedulerWrapper(scheduler)

        return states

    def _build_states(self) -> dict[str, Any]:
        states: dict[str, Any] = {}

        # Models/opts/schedulers are role-scoped.
        for role, handle in self.bundle.roles.items():
            states.update(self._build_role_states(role, handle))

        # Dataloader (optional but recommended for exact resume).
        if _is_stateful(self.dataloader):
            states["dataloader"] = self.dataloader

        # RNG states: always save global RNG; also save adapter-provided generators.
        states["random_state"] = RandomStateWrapper(None)
        if self._get_rng_generators is not None:
            for name, gen in (self._get_rng_generators() or {}).items():
                if gen is None:
                    continue
                states[f"random_state.{name}"] = RandomStateWrapper(gen)

        return states

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / f"checkpoint-{step}"

    def _dcp_dir(self, step: int) -> Path:
        return self._checkpoint_dir(step) / "dcp"

    def maybe_save(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        if step % save_steps != 0:
            return
        if self._last_saved_step == step:
            return
        self.save(step)

    def save_final(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        self.save(step)

    def save(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        dcp_dir = self._dcp_dir(step)
        os.makedirs(dcp_dir, exist_ok=True)

        states = self._build_states()
        if _rank() == 0:
            logger.info("Saving Phase 2 checkpoint to %s", checkpoint_dir)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()
        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> int | None:
        if not resume_from_checkpoint:
            return None

        resolved = _resolve_resume_checkpoint(resume_from_checkpoint, output_dir=self.output_dir)
        step = _parse_step_from_dir(resolved)

        states = self._build_states()
        logger.info("Loading Phase 2 checkpoint from %s", resolved)
        dcp.load(states, checkpoint_id=str(resolved / "dcp"))
        _barrier()
        logger.info("Checkpoint loaded; resuming from step=%s", step)
        return step

    def _cleanup_old_checkpoints(self) -> None:
        keep_last = int(self.config.keep_last or 0)
        if keep_last <= 0:
            return

        if _rank() != 0:
            _barrier()
            return

        output_dir = Path(self.output_dir)
        candidates: list[tuple[int, Path]] = []
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if not _CHECKPOINT_DIR_RE.match(child.name):
                continue
            try:
                step = _parse_step_from_dir(child)
            except Exception:
                continue
            candidates.append((step, child))

        candidates.sort(key=lambda x: x[0])
        to_delete = candidates[:-keep_last] if len(candidates) > keep_last else []
        for step, path in to_delete:
            logger.info("Removing old checkpoint (keep_last=%s): %s", keep_last, path)
            shutil.rmtree(path, ignore_errors=True)

        _barrier()
