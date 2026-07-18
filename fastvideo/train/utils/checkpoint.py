# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_DATALOADER_STATE_VERSION = 1
_DATALOADER_STATE_OVERRIDE_ENV = "FASTVIDEO_DATALOADER_STATE_DIR"


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Atomically publish a rank-local torch payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _parse_step_from_dir(checkpoint_dir: Path) -> int:
    match = _CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
    if not match:
        raise ValueError(f"Invalid checkpoint directory name {checkpoint_dir.name!r}; "
                         "expected 'checkpoint-<step>'")
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


def _resolve_resume_checkpoint(resume_from_checkpoint: str, *, output_dir: str) -> Path | None:
    """Resolve a user-provided resume path to a concrete checkpoint dir.

    Accepted values:
    - "latest" (auto-pick latest checkpoint-*/dcp under output_dir,
      or ``None`` if no checkpoint exists yet — starts from scratch)
    - /path/to/output_dir/checkpoint-<step>
    - /path/to/output_dir/checkpoint-<step>/dcp
    - /path/to/output_dir (auto-pick latest checkpoint-*/dcp)
    """

    if str(resume_from_checkpoint).strip().lower() == "latest":
        out = Path(os.path.expanduser(str(output_dir))).resolve()
        latest = _find_latest_checkpoint(out)
        if latest is None:
            logger.info(
                "resume_from_checkpoint='latest' but no "
                "checkpoints found under %s; starting from "
                "scratch.",
                out,
            )
        return latest

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
    raise ValueError("Could not resolve resume checkpoint. Expected a checkpoint directory "
                     f"named 'checkpoint-<step>' (with 'dcp/' inside), or an output_dir "
                     f"containing such checkpoints. Got: {path} (output_dir={out}).")


class _RoleModuleContainer(torch.nn.Module):
    """Ephemeral container to expose multiple role modules as a single
    ``nn.Module``.

    Used by ``OptimizerWrapper`` which expects a single root module
    covering all parameters owned by the optimizer.
    """

    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)


class _FullModelState(Stateful):
    """DCP wrapper that saves frozen model parameters too.

    The shared ``ModelWrapper`` intentionally filters to ``requires_grad``
    parameters. Frozen-but-mutated roles (e.g. DiffusionNFT's old policy,
    causal-CD's EMA target) must still be restored on resume, so they need
    full model state.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)  # type: ignore[no-any-return]

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class _CallbackStateWrapper:
    """Wraps a CallbackDict for DCP save/load."""

    def __init__(self, callbacks: Any) -> None:
        self._callbacks = callbacks

    def state_dict(self) -> dict[str, Any]:
        return self._callbacks.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self._callbacks.load_state_dict(state_dict)


@dataclass(slots=True)
class CheckpointConfig:
    save_steps: int
    keep_last: int


class CheckpointManager:
    """Role-based checkpoint manager for training runtime.

    - Checkpoint policy lives in YAML (via TrainingArgs fields).
    - Resume path is typically provided via CLI (``--resume-from-checkpoint``).
    """

    def __init__(
        self,
        *,
        method: Any,
        dataloader: Any,
        output_dir: str,
        config: CheckpointConfig,
        callbacks: Any | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.dataloader = dataloader
        self.output_dir = str(output_dir)
        self.config = config
        self._callbacks = callbacks
        self._raw_config = raw_config
        self._last_saved_step: int | None = None

    def _build_states(self) -> dict[str, Any]:
        states: dict[str, Any] = self.method.checkpoint_state()

        # Callback state (e.g. EMA shadow weights, validation RNG).
        if self._callbacks is not None and _is_stateful(self._callbacks):
            states["callbacks"] = _CallbackStateWrapper(self._callbacks, )

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
        if self._last_saved_step == step:
            return
        self.save(step)

    def save(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        dcp_dir = self._dcp_dir(step)
        os.makedirs(dcp_dir, exist_ok=True)

        states = self._build_states()
        if _rank() == 0:
            logger.info(
                "Saving checkpoint to %s",
                checkpoint_dir,
            )
            self._write_metadata(checkpoint_dir, step)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()

        # Dataloader state is rank-local.  Saving it under one shared DCP key
        # lets DCP deduplicate fields from different ranks into an invalid
        # chimera.  Keep one atomic sidecar per rank instead.
        self._save_dataloader_snapshot(checkpoint_dir, step)
        _barrier()

        if self._callbacks is not None:
            self._callbacks.on_checkpoint_save(
                self.method,
                checkpoint_dir=checkpoint_dir,
                iteration=step,
            )
        _barrier()

        # Save RNG state AFTER dcp.save so it captures the
        # exact state the continuous run continues with.
        # dcp.save triggers FSDP all-gather ops that can
        # advance the RNG between when DCP captures it and
        # when the save completes.
        self._save_rng_snapshot(checkpoint_dir)
        _barrier()

        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def _write_metadata(
        self,
        checkpoint_dir: Path,
        step: int,
    ) -> None:
        metadata: dict[str, Any] = {"step": step}
        if self._raw_config is not None:
            metadata["config"] = self._raw_config
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(checkpoint_dir: str | Path, ) -> dict[str, Any]:
        """Read ``metadata.json`` from a checkpoint dir."""
        meta_path = Path(checkpoint_dir) / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No metadata.json in {checkpoint_dir}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_rng_snapshot(self, checkpoint_dir: Path) -> None:
        """Save per-rank RNG state to a separate file.

        Called AFTER ``dcp.save`` so the snapshot reflects
        the exact state the continuous run continues with.
        Each rank saves its own file because CUDA RNG and
        custom generators differ across ranks.
        """
        rank = _rank()
        rng: dict[str, Any] = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        rng["cuda_rng"] = torch.cuda.get_rng_state()
        rng["gen_cuda"] = self.method.cuda_generator.get_state()
        torch.save(
            rng,
            checkpoint_dir / f"rng_state_rank{rank}.pt",
        )

    def _save_dataloader_snapshot(
        self,
        checkpoint_dir: Path,
        step: int,
    ) -> None:
        if not _is_stateful(self.dataloader):
            return
        rank = _rank()
        payload = {
            "version": _DATALOADER_STATE_VERSION,
            "rank": rank,
            "world_size": _world_size(),
            "step": int(step),
            "state_kind": "dataloader",
            "state": self.dataloader.state_dict(),
        }
        path = checkpoint_dir / f"dataloader_state_rank{rank}.pt"
        _atomic_torch_save(payload, path)
        logger.info("Saved rank-local dataloader state to %s", path)

    def _resolve_dataloader_snapshot(
        self,
        checkpoint_dir: Path,
    ) -> Path | None:
        rank = _rank()
        default_path = checkpoint_dir / f"dataloader_state_rank{rank}.pt"
        if default_path.is_file():
            return default_path

        override_dir = os.environ.get(_DATALOADER_STATE_OVERRIDE_ENV, "").strip()
        if override_dir:
            override_path = (Path(os.path.expanduser(override_dir)).resolve() /
                             f"dataloader_state_rank{rank}.pt")
            if override_path.is_file():
                return override_path
            raise FileNotFoundError(
                f"{_DATALOADER_STATE_OVERRIDE_ENV} is set, but rank {rank} "
                f"state is missing: {override_path}"
            )

        # A single-rank legacy DCP cannot mix state from different ranks, so
        # loading its original dataloader key remains safe and preserves
        # backwards compatibility.
        if _world_size() == 1:
            return None

        raise RuntimeError(
            "Checkpoint has no rank-local dataloader state sidecar. Legacy "
            "multi-rank DCP dataloader state is unsafe because identically "
            "named rank-local fields can be deduplicated across ranks. "
            f"Create audited migration sidecars and set {_DATALOADER_STATE_OVERRIDE_ENV}."
        )

    def _load_dataloader_snapshot(
        self,
        checkpoint_dir: Path,
        step: int,
    ) -> None:
        if not _is_stateful(self.dataloader):
            return
        path = self._resolve_dataloader_snapshot(checkpoint_dir)
        if path is None:
            logger.warning(
                "Loading legacy single-rank dataloader state from DCP at %s",
                checkpoint_dir,
            )
            dcp.load(
                {"dataloader": self.dataloader},
                checkpoint_id=str(checkpoint_dir / "dcp"),
            )
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid dataloader state payload: {path}")

        expected = {
            "version": _DATALOADER_STATE_VERSION,
            "rank": _rank(),
            "world_size": _world_size(),
            "step": int(step),
        }
        for field, expected_value in expected.items():
            actual = payload.get(field)
            if actual != expected_value:
                raise ValueError(
                    f"Dataloader state {field} mismatch in {path}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )

        state_kind = payload.get("state_kind")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"Dataloader state is not a dictionary: {path}")
        if state_kind == "dataloader":
            self.dataloader.load_state_dict(state)
        elif state_kind == "dataset":
            dataset = getattr(self.dataloader, "dataset", None)
            if not _is_stateful(dataset):
                raise ValueError(
                    "Dataset-only migration state requires a stateful "
                    f"dataloader.dataset: {path}"
                )
            dataset.load_state_dict(state)
        else:
            raise ValueError(
                f"Unsupported dataloader state_kind {state_kind!r} in {path}"
            )
        logger.info("Restored rank-local dataloader state from %s", path)

    def load_rng_snapshot(
        self,
        checkpoint_path: str,
    ) -> None:
        """Restore per-rank RNG state from the snapshot file.

        Must be called AFTER ``dcp.load`` **and** after
        ``iter(dataloader)`` so no later operation can
        clobber the restored state.
        """
        resolved = _resolve_resume_checkpoint(
            checkpoint_path,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return
        rank = _rank()
        rng_path = resolved / f"rng_state_rank{rank}.pt"
        if not rng_path.is_file():
            # Fall back to legacy single-file snapshot.
            rng_path = resolved / "rng_state.pt"
        if not rng_path.is_file():
            logger.warning(
                "No rng_state in %s; skipping "
                "RNG snapshot restore.",
                resolved,
            )
            return

        rng = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )
        if "torch_rng" in rng:
            torch.set_rng_state(rng["torch_rng"])
        if "python_rng" in rng:
            random.setstate(rng["python_rng"])
        if "numpy_rng" in rng:
            np.random.set_state(rng["numpy_rng"])

        torch.cuda.set_rng_state(rng["cuda_rng"])
        self.method.cuda_generator.set_state(rng["gen_cuda"])
        logger.info(
            "Restored RNG snapshot from %s",
            rng_path,
        )

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> int | None:
        if not resume_from_checkpoint:
            return None

        resolved = _resolve_resume_checkpoint(
            resume_from_checkpoint,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return None
        step = _parse_step_from_dir(resolved)

        states = self._build_states()
        logger.info("Loading Phase 2 checkpoint from %s", resolved)
        dcp.load(states, checkpoint_id=str(resolved / "dcp"))
        self._load_dataloader_snapshot(resolved, step)
        if self._callbacks is not None:
            self._callbacks.on_checkpoint_load(
                self.method,
                checkpoint_dir=resolved,
                iteration=step,
            )
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
