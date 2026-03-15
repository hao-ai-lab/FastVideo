# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_BEST_CHECKPOINT_DIR_RE = re.compile(
    r"^checkpoint-best-step-(\d+)$"
)


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_int(value: int, *, src: int = 0) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    backend = dist.get_backend()
    use_cuda = (
        backend == "nccl"
        and torch.cuda.is_available()
    )
    device = (
        torch.device("cuda")
        if use_cuda
        else torch.device("cpu")
    )
    tensor = torch.tensor(
        [int(value)],
        dtype=torch.int32,
        device=device,
    )
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def _parse_step_from_dir(checkpoint_dir: Path) -> int:
    match = (
        _CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
        or _BEST_CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
    )
    if not match:
        raise ValueError(
            f"Invalid checkpoint directory name {checkpoint_dir.name!r}; "
            "expected 'checkpoint-<step>' or 'checkpoint-best-step-<step>'"
        )
    return int(match.group(1))


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if not (
            _CHECKPOINT_DIR_RE.match(child.name)
            or _BEST_CHECKPOINT_DIR_RE.match(child.name)
        ):
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

    if path.is_dir() and (
        _CHECKPOINT_DIR_RE.match(path.name)
        or _BEST_CHECKPOINT_DIR_RE.match(path.name)
    ):
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
                     "named 'checkpoint-<step>' or 'checkpoint-best-step-<step>' "
                     f"(with 'dcp/' inside), or an output_dir containing such "
                     f"checkpoints. Got: {path} (output_dir={out}).")


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
        tracker: Any | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.dataloader = dataloader
        self.output_dir = str(output_dir)
        self.config = config
        self._callbacks = callbacks
        self._tracker = tracker
        self._raw_config = raw_config
        self._last_saved_step: int | None = None
        self._last_best_saved_step: int | None = None

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

    def _best_checkpoint_dir(self, step: int) -> Path:
        return (
            Path(self.output_dir)
            / f"checkpoint-best-step-{step}"
        )

    def _checkpoint_best_alias_path(self) -> Path:
        return Path(self.output_dir) / "checkpoint-best"

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
        self._save_checkpoint_dir(
            checkpoint_dir,
            step=step,
            log_prefix="Saving checkpoint",
        )
        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def _save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        step: int,
        log_prefix: str,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        dcp_dir = checkpoint_dir / "dcp"
        os.makedirs(dcp_dir, exist_ok=True)

        states = self._build_states()
        if _rank() == 0:
            logger.info("%s to %s", log_prefix, checkpoint_dir)
            self._write_metadata(
                checkpoint_dir,
                step,
                extra=metadata_extra,
            )
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()
        self._save_rng_snapshot(checkpoint_dir)
        _barrier()

    def _write_metadata(
        self,
        checkpoint_dir: Path,
        step: int,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        metadata: dict[str, Any] = {"step": step}
        if self._raw_config is not None:
            metadata["config"] = self._raw_config
        if extra:
            metadata.update(extra)
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

    def _list_best_checkpoint_entries(
        self,
        metric_name: str,
    ) -> list[dict[str, Any]]:
        output_dir = Path(self.output_dir)
        if not output_dir.is_dir():
            return []

        entries: list[dict[str, Any]] = []
        for child in output_dir.iterdir():
            if (
                not child.is_dir()
                or not _BEST_CHECKPOINT_DIR_RE.match(
                    child.name
                )
            ):
                continue
            metric_path = child / "best_metric.json"
            if not metric_path.is_file():
                logger.warning(
                    "Skipping %s: best_metric.json "
                    "is missing",
                    child,
                )
                continue
            try:
                with open(
                    metric_path, encoding="utf-8"
                ) as f:
                    meta = json.load(f)
                metric_raw = meta.get(metric_name)
                if metric_raw is None:
                    metric_raw = meta.get(
                        "mf_angle_err_mean"
                    )
                if metric_raw is None:
                    metric_raw = meta.get(
                        "metric_value"
                    )

                step_raw = meta.get("step")
                if step_raw is None:
                    match = (
                        _BEST_CHECKPOINT_DIR_RE.match(
                            child.name
                        )
                    )
                    if match is None:
                        continue
                    step_raw = int(match.group(1))

                metric_val = float(metric_raw)
                step_val = int(step_raw)
                if not math.isfinite(metric_val):
                    raise ValueError(
                        "metric is non-finite"
                    )
            except Exception as e:
                logger.warning(
                    "Skipping %s: invalid "
                    "best metric metadata (%s)",
                    child,
                    e,
                )
                continue
            entries.append(
                {
                    "path": child,
                    "step": step_val,
                    "metric": metric_val,
                }
            )

        entries.sort(
            key=lambda x: (
                float(x["metric"]),
                int(x["step"]),
            )
        )
        return entries

    def _update_best_checkpoint_alias(
        self,
        best_checkpoint_path: Path,
    ) -> None:
        alias_path = self._checkpoint_best_alias_path()
        try:
            if alias_path.is_symlink() or alias_path.is_file():
                alias_path.unlink()
            elif alias_path.is_dir():
                shutil.rmtree(alias_path)
            os.symlink(
                os.path.basename(str(best_checkpoint_path)),
                str(alias_path),
            )
        except OSError as e:
            logger.warning(
                "Failed to update checkpoint-best "
                "alias: %s",
                e,
            )

    def maybe_save_best(
        self,
        *,
        step: int,
        metric_value: float | None,
        metric_name: str = "mf_angle_err_mean",
        start_step: int = 0,
        top_k: int = 1,
    ) -> None:
        start_step = int(start_step or 0)
        if start_step <= 0:
            return
        if int(step) < start_step:
            return
        if metric_value is None:
            return
        metric = float(metric_value)
        if not math.isfinite(metric):
            return
        if self._last_best_saved_step == int(step):
            return

        top_k = max(1, int(top_k or 1))
        metric_name = str(metric_name)

        should_save = 0
        if _rank() == 0:
            entries = self._list_best_checkpoint_entries(
                metric_name
            )
            if len(entries) < top_k:
                should_save = 1
            else:
                worst = entries[-1]
                if metric < float(
                    worst["metric"]
                ):
                    should_save = 1
        should_save = _broadcast_int(
            should_save, src=0
        )
        if should_save == 0:
            return

        checkpoint_dir = self._best_checkpoint_dir(
            int(step)
        )
        if _rank() == 0 and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        _barrier()

        logger.info(
            "%s=%.6f at step %s entered top-%s "
            "best checkpoints; saving.",
            metric_name,
            metric,
            step,
            top_k,
        )
        self._save_checkpoint_dir(
            checkpoint_dir,
            step=int(step),
            log_prefix="Saving best checkpoint",
            metadata_extra={
                "kind": "best",
                "metric_name": metric_name,
                "metric_value": metric,
            },
        )
        self._last_best_saved_step = int(step)

        if _rank() == 0:
            metric_meta = {
                "step": int(step),
                metric_name: metric,
                "metric_name": metric_name,
                "metric_value": metric,
            }
            if metric_name == "mf_angle_err_mean":
                metric_meta[
                    "mf_angle_err_mean"
                ] = metric
            metric_path = (
                checkpoint_dir / "best_metric.json"
            )
            with open(
                metric_path, "w", encoding="utf-8"
            ) as f:
                json.dump(metric_meta, f, indent=2)

            best_entries = self._list_best_checkpoint_entries(
                metric_name
            )
            kept_entries = best_entries[:top_k]
            for stale_entry in best_entries[top_k:]:
                stale_path = Path(
                    stale_entry["path"]
                )
                logger.info(
                    "Removing non-top-k best "
                    "checkpoint: %s",
                    stale_path,
                )
                shutil.rmtree(
                    stale_path, ignore_errors=True
                )

            if kept_entries:
                top1 = kept_entries[0]
                self._update_best_checkpoint_alias(
                    Path(top1["path"])
                )
                if self._tracker is not None:
                    self._tracker.log(
                        {
                            f"best/{metric_name}": float(
                                top1["metric"]
                            ),
                            "best/step": int(
                                top1["step"]
                            ),
                            "best/topk_count": len(
                                kept_entries
                            ),
                        },
                        int(step),
                    )
        _barrier()

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
        try:
            dcp.load(states, checkpoint_id=str(resolved / "dcp"))
        except BaseException as exc:
            if not isinstance(exc, dcp.CheckpointException):
                raise
            msg = str(exc)
            fallback_prefixes = (
                "optimizers.",
                "schedulers.",
                "dataloader",
                "callbacks.",
                "random_state",
            )
            can_fallback = (
                "Missing key in checkpoint state_dict:" in msg
                and any(
                    f"Missing key in checkpoint state_dict: {prefix}"
                    in msg
                    for prefix in fallback_prefixes
                )
            )
            if not can_fallback:
                raise

            model_only_states = {
                key: value
                for key, value in states.items()
                if key.startswith("roles.")
            }
            if not model_only_states:
                raise

            logger.warning(
                "Resume checkpoint is missing non-model state "
                "(optimizer/scheduler/dataloader/callback/RNG). "
                "Falling back to model-only restore from %s "
                "with %d role states. Optimizer/scheduler/etc. "
                "will be reinitialized.",
                resolved,
                len(model_only_states),
            )
            dcp.load(
                model_only_states,
                checkpoint_id=str(resolved / "dcp"),
            )
        _barrier()
        logger.info("Checkpoint loaded; resuming from step=%s", step)
        return step

    def _save_rng_snapshot(self, checkpoint_dir: Path) -> None:
        """Save per-rank RNG state after DCP save completes."""
        rank = _rank()
        rng: dict[str, Any] = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng["cuda_rng"] = torch.cuda.get_rng_state()
        cuda_generator = getattr(self.method, "cuda_generator", None)
        if cuda_generator is not None:
            rng["gen_cuda"] = cuda_generator.get_state()
        torch.save(
            rng,
            checkpoint_dir / f"rng_state_rank{rank}.pt",
        )

    def load_rng_snapshot(
        self,
        checkpoint_path: str,
    ) -> None:
        resolved = _resolve_resume_checkpoint(
            checkpoint_path,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return
        rank = _rank()
        rng_path = resolved / f"rng_state_rank{rank}.pt"
        if not rng_path.is_file():
            rng_path = resolved / "rng_state.pt"
        if not rng_path.is_file():
            logger.warning(
                "No rng_state in %s; skipping RNG snapshot restore.",
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
        if torch.cuda.is_available() and "cuda_rng" in rng:
            torch.cuda.set_rng_state(rng["cuda_rng"])
        cuda_generator = getattr(self.method, "cuda_generator", None)
        if cuda_generator is not None and "gen_cuda" in rng:
            cuda_generator.set_state(rng["gen_cuda"])

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
