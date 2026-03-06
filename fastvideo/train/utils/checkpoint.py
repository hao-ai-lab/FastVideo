# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from fastvideo.train.models.base import ModelBase
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


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
    raise ValueError("Could not resolve resume checkpoint. Expected a checkpoint directory "
                     f"named 'checkpoint-<step>' (with 'dcp/' inside), or an output_dir "
                     f"containing such checkpoints. Got: {path} (output_dir={out}).")


def save_role_pretrained(
    *,
    role: str,
    base_model_path: str,
    output_dir: str,
    module_names: list[str] | None = None,
    overwrite: bool = False,
    model: ModelBase,
) -> str:
    """Export a role's modules into a diffusers-style model directory.

    This is intended to produce a ``model_path`` that can be loaded by
    ``PipelineComponentLoader`` (i.e., has ``model_index.json``,
    ``transformer/``, ``vae/``, and other pipeline components copied
    from ``base_model_path``).
    """

    # Resolve HF IDs to local directories (same behavior as module loader).
    from fastvideo.utils import maybe_download_model

    local_base = Path(maybe_download_model(str(base_model_path))).resolve()
    dst = Path(os.path.expanduser(str(output_dir))).resolve()

    if _rank() == 0:
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst, ignore_errors=True)
            else:
                raise FileExistsError(f"Refusing to overwrite existing directory: {dst}. "
                                      "Pass overwrite=True to replace it.")

        def _copy_or_link(src: str, dest: str) -> None:
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)

        logger.info("Creating pretrained export dir at %s (base=%s)", dst, local_base)
        shutil.copytree(local_base, dst, symlinks=True, copy_function=_copy_or_link)

    _barrier()

    modules: dict[str, torch.nn.Module] = {}
    if model.transformer is not None:
        modules["transformer"] = model.transformer

    if module_names is None:
        module_names = sorted(modules.keys())

    for module_name in module_names:
        if module_name not in modules:
            raise KeyError(f"Role {role!r} does not have module {module_name!r}. "
                           f"Available: {sorted(modules.keys())}")

        module_dir = dst / module_name
        if not module_dir.is_dir():
            raise FileNotFoundError(f"Export directory missing component dir {module_name!r}: {module_dir}")

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(modules[module_name], options=options)

        if _rank() == 0:
            # Remove existing *.safetensors to avoid loading duplicate weights.
            for path in module_dir.glob("*.safetensors"):
                path.unlink(missing_ok=True)

            tensor_state: dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expected tensor in state_dict for {module_name}.{key}, "
                                    f"got {type(value).__name__}")
                tensor_state[key] = value.detach().cpu()

            from safetensors.torch import save_file

            out_path = module_dir / "model.safetensors"
            logger.info(
                "Saving %s weights to %s (%s tensors)",
                module_name,
                out_path,
                len(tensor_state),
            )
            save_file(tensor_state, str(out_path))

        _barrier()

    return str(dst)


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


class _CallbackStateWrapper:
    """Wraps a CallbackDict for DCP save/load."""

    def __init__(self, callbacks: Any) -> None:
        self._callbacks = callbacks

    def state_dict(self) -> dict[str, Any]:
        return self._callbacks.state_dict()

    def load_state_dict(
        self, state_dict: dict[str, Any],
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

        # Dataloader (optional but recommended for exact resume).
        if _is_stateful(self.dataloader):
            states["dataloader"] = self.dataloader

        # Callback state (e.g. EMA shadow weights, validation RNG).
        if self._callbacks is not None and _is_stateful(self._callbacks):
            states["callbacks"] = _CallbackStateWrapper(
                self._callbacks,
            )

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
            logger.info(
                "Saving checkpoint to %s", checkpoint_dir,
            )
            self._write_metadata(checkpoint_dir, step)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()
        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def _write_metadata(
        self, checkpoint_dir: Path, step: int,
    ) -> None:
        metadata: dict[str, Any] = {"step": step}
        if self._raw_config is not None:
            metadata["config"] = self._raw_config
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(
        checkpoint_dir: str | Path,
    ) -> dict[str, Any]:
        """Read ``metadata.json`` from a checkpoint dir."""
        meta_path = Path(checkpoint_dir) / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"No metadata.json in {checkpoint_dir}"
            )
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> int | None:
        if not resume_from_checkpoint:
            return None

        resolved = _resolve_resume_checkpoint(
            resume_from_checkpoint,
            output_dir=self.output_dir,
        )
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
