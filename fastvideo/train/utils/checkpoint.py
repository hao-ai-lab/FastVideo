# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import stat
import time
from collections import deque
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
_TRAINING_CHECKPOINT_COMPLETE_MARKER = ".complete"
_RANK_RNG_STATE_RE = re.compile(r"^rng_state_rank(\d+)\.pt$")


def _inference_dependency_identity(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    identity: tuple[int, ...] = (info.st_dev, info.st_ino, info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        identity += (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return identity


def _trace_inference_dependency(path: Path, snapshots: dict[Path, tuple[int, ...]]) -> tuple[set[Path], Path]:
    """Resolve component by component without discarding intermediate owners.

    No directory walk or payload reads: only lstat/readlink of the dependency
    path. Uncertain chains are errors so retention can keep every export.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    pending = deque(absolute.parts[1:])
    dependencies: set[Path] = set()
    seen_links: set[Path] = set()
    components = 0
    while pending:
        components += 1
        if components > 4096:
            raise RuntimeError(f"Inference dependency traversal limit exceeded: {path}")
        part = pending.popleft()
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        identity = _inference_dependency_identity(candidate)
        if candidate in snapshots and snapshots[candidate] != identity:
            raise RuntimeError(f"Inference dependency changed during traversal: {candidate}")
        snapshots[candidate] = identity
        dependencies.add(candidate)
        if stat.S_ISLNK(identity[2]):
            if candidate in seen_links or len(seen_links) >= 64:
                raise RuntimeError(f"Cyclic or excessive inference dependency aliases: {path}")
            seen_links.add(candidate)
            target = Path(os.readlink(candidate))
            if _inference_dependency_identity(candidate) != identity:
                raise RuntimeError(f"Inference dependency changed during traversal: {candidate}")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            if pending and not stat.S_ISDIR(identity[2]):
                raise RuntimeError(f"Non-directory in inference dependency: {candidate}")
            current = candidate
    return dependencies, current


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


def _saved_checkpoint_world_size(metadata: dict[str, Any]) -> int | None:
    try:
        world_size = metadata["config"]["training"]["distributed"]["num_gpus"]
    except (KeyError, TypeError):
        return None
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        return None
    return world_size


def _is_complete_training_checkpoint(
    checkpoint_dir: Path,
    *,
    require_complete_marker: bool,
) -> bool:
    """Return whether ``checkpoint_dir`` is safe to select for resume.

    ``dcp/.metadata`` is the historical completion contract. Strict callers
    additionally require the marker published after every rank has written its
    RNG snapshot. Keeping strictness opt-in preserves compatibility with
    checkpoints created before the stronger marker existed.
    """
    dcp_metadata = checkpoint_dir / "dcp" / ".metadata"
    if not dcp_metadata.is_file():
        return False
    if not require_complete_marker:
        return True

    try:
        step = _parse_step_from_dir(checkpoint_dir)
        marker = (checkpoint_dir / _TRAINING_CHECKPOINT_COMPLETE_MARKER).read_text(encoding="utf-8")
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    if not isinstance(metadata, dict) or marker != "complete\n" or metadata.get("step") != step:
        return False

    world_size = _saved_checkpoint_world_size(metadata)
    if world_size is None:
        return False
    expected_rng_names = {f"rng_state_rank{rank}.pt" for rank in range(world_size)}
    try:
        actual_rng_paths = list(checkpoint_dir.glob("rng_state_rank*.pt"))
        actual_rng_names = {path.name for path in actual_rng_paths if _RANK_RNG_STATE_RE.match(path.name)}
        rng_files_complete = all(path.is_file() and path.stat().st_size > 0 for path in actual_rng_paths)
    except OSError:
        return False
    return (actual_rng_names == expected_rng_names and len(actual_rng_paths) == len(expected_rng_names)
            and rng_files_complete)


def _find_latest_checkpoint(
    output_dir: Path,
    *,
    require_complete_marker: bool = False,
) -> Path | None:
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _CHECKPOINT_DIR_RE.match(child.name):
            continue
        if not _is_complete_training_checkpoint(
                child,
                require_complete_marker=require_complete_marker,
        ):
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


def _publish_training_checkpoint_complete(checkpoint_dir: Path) -> None:
    """Atomically publish the marker that makes a training checkpoint visible."""
    marker = checkpoint_dir / _TRAINING_CHECKPOINT_COMPLETE_MARKER
    temporary = checkpoint_dir / f"{_TRAINING_CHECKPOINT_COMPLETE_MARKER}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("complete\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _resolve_resume_checkpoint(
    resume_from_checkpoint: str,
    *,
    output_dir: str,
    require_complete_marker: bool = False,
) -> Path | None:
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
        latest = _find_latest_checkpoint(
            out,
            require_complete_marker=require_complete_marker,
        )
        if latest is None:
            has_checkpoint_dirs = out.is_dir() and any(child.is_dir() and _CHECKPOINT_DIR_RE.match(child.name)
                                                       for child in out.iterdir())
            if require_complete_marker and has_checkpoint_dirs:
                raise ValueError(f"No complete resumable checkpoint found under {out}; "
                                 "refusing to start from scratch in a non-empty training namespace")
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
        if not _is_complete_training_checkpoint(
                path,
                require_complete_marker=require_complete_marker,
        ):
            raise ValueError(f"Checkpoint is incomplete under the configured resume policy: {path}")
        return path

    # Treat as output_dir -> pick latest.
    latest = _find_latest_checkpoint(
        path,
        require_complete_marker=require_complete_marker,
    )
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
    # Full distributed state used to resume training.
    save_steps: int
    keep_last: int
    # Suppress periodic saves before this step (early checkpoints of a long
    # run are rarely useful and cost ~100 GiB each). 0 disables the gate.
    start_step: int = 0
    # Deployable model-only checkpoints are written for every validation event
    # and are never removed by ``keep_last``.
    save_inference_on_validation: bool = False
    inference_role: str = "student"
    inference_dtype: str = "bfloat16"
    # Require the post-RNG completion marker when resolving resumable state.
    # False preserves checkpoints written before that marker was introduced.
    require_complete_training_checkpoint: bool = False
    # Independent opt-in export limit; appended to preserve positional callers.
    inference_checkpoints_total_limit: int = 0

    def __post_init__(self) -> None:
        if type(self.inference_checkpoints_total_limit) is not int or self.inference_checkpoints_total_limit < 0:
            raise ValueError("inference_checkpoints_total_limit must be a non-negative integer")


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
        save_inference = bool(config.save_inference_on_validation)
        inference_role = str(config.inference_role or "")
        if save_inference and (not inference_role or "." in inference_role):
            raise ValueError("inference_role must be a non-empty DCP key segment when inference saving is enabled")
        if save_inference and str(config.inference_dtype) not in {"bfloat16", "float16", "float32"}:
            raise ValueError("inference_dtype must be bfloat16, float16, or float32")
        if config.require_complete_training_checkpoint:
            metadata = {"config": raw_config}
            if _saved_checkpoint_world_size(metadata) is None:
                raise ValueError("require_complete_training_checkpoint needs a positive "
                                 "training.distributed.num_gpus value in the saved raw config")
        self._callbacks = callbacks
        self._raw_config = raw_config
        # Training-state and inference checkpoints have independent policies
        # and deduplication.
        self._last_saved_step: int | None = None
        self._last_inference_saved_step: int | None = None

    def _build_states(self) -> dict[str, Any]:
        states: dict[str, Any] = self.method.checkpoint_state()

        # Dataloader (optional but recommended for exact resume).
        if _is_stateful(self.dataloader):
            states["dataloader"] = self.dataloader

        # Callback state (e.g. EMA shadow weights, validation RNG).
        if self._callbacks is not None and _is_stateful(self._callbacks):
            states["callbacks"] = _CallbackStateWrapper(self._callbacks, )

        return states

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / f"checkpoint-{step}"

    def _dcp_dir(self, step: int) -> Path:
        return self._checkpoint_dir(step) / "dcp"

    def _inference_checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / "inference" / f"checkpoint-{step}"

    def _inference_staging_dir(self, step: int) -> Path:
        return Path(self.output_dir) / ".inference-staging" / f"checkpoint-{step}"

    def maybe_save(self, step: int) -> None:
        if step < int(self.config.start_step or 0):
            return

        save_steps = int(self.config.save_steps or 0)
        if save_steps > 0 and step % save_steps == 0 and self._last_saved_step != step:
            self.save(step)

    def maybe_save_inference(self, step: int, *, validation_scheduled: bool) -> None:
        """Save the model evaluated by one scheduled validation event.

        This event-driven policy includes step-zero validation and deliberately
        ignores the start gate and cadence used for resumable training state.
        """
        if not validation_scheduled or not bool(self.config.save_inference_on_validation):
            return
        if self._last_inference_saved_step == step:
            return
        self.save_inference(step)

    def save_final(self, step: int) -> None:
        if int(self.config.save_steps or 0) > 0 and self._last_saved_step != step:
            self.save(step)

    def save(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        dcp_dir = self._dcp_dir(step)
        os.makedirs(dcp_dir, exist_ok=True)

        # A retry may target a directory whose previous DCP save completed but
        # whose RNG snapshots did not. Remove the publication marker before
        # overwriting any state so strict readers can never select stale data.
        if _rank() == 0:
            with contextlib.suppress(FileNotFoundError):
                (checkpoint_dir / _TRAINING_CHECKPOINT_COMPLETE_MARKER).unlink()
        _barrier()

        states = self._build_states()
        if _rank() == 0:
            logger.info(
                "Saving resumable training checkpoint to %s",
                checkpoint_dir,
            )
            self._write_metadata(checkpoint_dir, step)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()

        # Save RNG state AFTER dcp.save so it captures the
        # exact state the continuous run continues with.
        # dcp.save triggers FSDP all-gather ops that can
        # advance the RNG between when DCP captures it and
        # when the save completes.
        self._save_rng_snapshot(checkpoint_dir)
        _barrier()

        if _rank() == 0:
            _publish_training_checkpoint_complete(checkpoint_dir)
        _barrier()

        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def save_inference(self, step: int) -> None:
        """Save one deployable inference checkpoint for the configured role.

        DCP is used only as a temporary, distributed staging format so FSDP2
        ranks never gather the full fp32 model into one process. Rank zero then
        streams bounded tensor groups into a bf16/fp16/fp32 modular model
        directory and publishes it atomically.
        """
        role = str(self.config.inference_role or "student")
        modules = self.method.inference_checkpoint_modules(role)
        base_model_path = self.method.inference_checkpoint_base_model_path(role)
        checkpoint_dir = self._inference_checkpoint_dir(step)

        # A method may declare the sampling contract its export must carry
        # (FastH3 PDD students). Rank zero resolves it from checkpoint files;
        # every rank then fails together when it cannot be built.
        inference_contract: dict[str, Any] | None = None
        contract_error: str | None = None
        contract_hook = getattr(self.method, "inference_checkpoint_contract", None)
        if callable(contract_hook):
            if _rank() == 0:
                try:
                    inference_contract = contract_hook(role)
                except Exception as error:
                    contract_error = f"{type(error).__name__}: {error}"
            if dist.is_available() and dist.is_initialized():
                contract_payload: list[Any] = [inference_contract, contract_error]
                dist.broadcast_object_list(contract_payload, src=0)
                inference_contract, contract_error = contract_payload
            if contract_error is not None:
                raise RuntimeError(f"Cannot build the inference contract at step {step}: {contract_error}")

        already_complete: bool | None = None
        existing_error: str | None = None
        if _rank() == 0:
            try:
                from fastvideo.train.utils.inference_checkpoint import (
                    module_export_hooks,
                    validate_complete_inference_checkpoint,
                )

                if self.config.inference_checkpoints_total_limit and (Path(self.output_dir).is_symlink()
                                                                      or checkpoint_dir.parent.is_symlink()
                                                                      or checkpoint_dir.is_symlink()):
                    raise ValueError(f"Unsafe inference retention namespace: {checkpoint_dir}")
                # Models with a structurally different export (PDD-widened
                # heads) declare their expected shapes; validate against them.
                if len(modules) != 1:
                    raise ValueError("Inference export requires exactly one model component")
                module_name, module = next(iter(modules.items()))
                shape_hook, config_overrides = module_export_hooks(module)
                existing = validate_complete_inference_checkpoint(
                    checkpoint_dir,
                    step=step,
                    expected_shape_override=shape_hook,
                    expected_dtype=self.config.inference_dtype,
                    expected_config_overrides=config_overrides,
                    expected_role=role,
                    expected_module=module_name,
                    expected_inference_contract=inference_contract,
                )
                already_complete = existing is not None
            except Exception as error:
                existing_error = f"{type(error).__name__}: {error}"
        if dist.is_available() and dist.is_initialized():
            complete_payload: list[Any] = [already_complete, existing_error]
            dist.broadcast_object_list(complete_payload, src=0)
            already_complete = bool(complete_payload[0])
            existing_error = complete_payload[1]
        if existing_error is not None:
            raise RuntimeError(f"Existing inference checkpoint failed validation at step {step}: {existing_error}")
        if already_complete:
            if _rank() == 0:
                logger.info("Inference checkpoint already complete at %s; skipping", checkpoint_dir)
            self._cleanup_old_inference_checkpoints(step, modules=modules, base_model_path=base_model_path)
            self._last_inference_saved_step = step
            return

        staging_dir = self._inference_staging_dir(step)
        dcp_dir = staging_dir / "dcp"
        export_status_path = staging_dir / "export-status.json"
        if _rank() == 0:
            # A prior failed save is never a valid source: DCP writes
            # ``.metadata`` last, and the exporter publishes independently.
            shutil.rmtree(staging_dir, ignore_errors=True)
            os.makedirs(dcp_dir, exist_ok=True)
        _barrier()

        states = {f"roles.{role}.{module_name}": _FullModelState(module) for module_name, module in modules.items()}
        if not states:
            raise ValueError(f"Inference checkpoint role {role!r} exposes no modules")

        # Saving weights must not perturb the training trajectory. The regular
        # resumable checkpoint intentionally snapshots its post-save RNG state;
        # this model-only staging save instead restores the pre-save state.
        torch_rng = torch.get_rng_state()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        generator = getattr(self.method, "cuda_generator", None)
        generator_rng = generator.get_state() if generator is not None else None
        try:
            if _rank() == 0:
                logger.info("Staging inference role %s with DCP at %s", role, dcp_dir)
            dcp.save(states, checkpoint_id=str(dcp_dir))
            _barrier()
        finally:
            torch.set_rng_state(torch_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            if generator is not None and generator_rng is not None:
                generator.set_state(generator_rng)

        export_error: str | None = None
        if _rank() == 0:
            try:
                from fastvideo.train.utils.inference_checkpoint import (
                    export_inference_checkpoint, )

                export_inference_checkpoint(
                    dcp_dir=dcp_dir,
                    output_dir=checkpoint_dir,
                    base_model_path=base_model_path,
                    role=role,
                    modules=modules,
                    dtype=str(self.config.inference_dtype),
                    step=step,
                    raw_config=self._raw_config,
                    inference_contract=inference_contract,
                )
                if self.config.inference_checkpoints_total_limit:
                    validated = validate_complete_inference_checkpoint(checkpoint_dir,
                                                                       step=step,
                                                                       expected_shape_override=shape_hook,
                                                                       expected_inference_contract=inference_contract)
                    if validated is None:
                        raise RuntimeError("Inference exporter did not publish a complete replacement")
            except Exception as error:  # propagate the rank-zero failure collectively
                logger.exception("Inference checkpoint export failed at step %s", step)
                export_error = f"{type(error).__name__}: {error}"
            status_tmp = export_status_path.with_suffix(".tmp")
            status_tmp.write_text(
                json.dumps({
                    "complete": export_error is None,
                    "error": export_error
                }) + "\n",
                encoding="utf-8",
            )
            os.replace(status_tmp, export_status_path)
        else:
            # Do not enter a collective while rank zero performs a multi-minute
            # CPU/shared-filesystem export: an outstanding NCCL operation can trip the
            # process-group watchdog. The atomically published shared-FS result
            # gives every rank the same terminal outcome before any barrier.
            last_log = time.monotonic()
            while not export_status_path.is_file():
                time.sleep(2.0)
                now = time.monotonic()
                if now - last_log >= 60.0:
                    logger.info("Waiting for rank-zero inference export at step %s", step)
                    last_log = now
            try:
                status = json.loads(export_status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Invalid inference export status at step {step}: {export_status_path}") from error
            if status.get("complete") is not True:
                export_error = str(status.get("error") or "rank-zero export failed without an error message")
        if export_error is not None:
            raise RuntimeError(f"Inference checkpoint export failed at step {step}: {export_error}")

        _barrier()
        if _rank() == 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_root = staging_dir.parent
            with contextlib.suppress(OSError):
                staging_root.rmdir()
        _barrier()
        self._cleanup_old_inference_checkpoints(step, modules=modules, base_model_path=base_model_path)
        self._last_inference_saved_step = step

    def _cleanup_old_inference_checkpoints(self, published_step: int, *, modules: dict[str, Any],
                                           base_model_path: str) -> None:
        """Rotate only recognized exports in this run after collective publication.

        Foreign/incomplete entries and input roots are not retention candidates.
        Shared base-component symlinks inside an export are unlinked, never followed.
        A retry of an older step must never evict the newer, last-good export.
        """
        limit = self.config.inference_checkpoints_total_limit
        if not limit:
            return
        error_message = None
        if _rank() == 0:
            try:
                self._prune_inference_checkpoints(published_step, modules=modules, base_model_path=base_model_path)
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
        if dist.is_available() and dist.is_initialized():
            payload: list[Any] = [error_message]
            dist.broadcast_object_list(payload, src=0)
            error_message = payload[0]
        if error_message is not None:
            raise RuntimeError(f"Inference checkpoint retention failed: {error_message}")

    def _prune_inference_checkpoints(self, published_step: int, *, modules: dict[str, Any],
                                     base_model_path: str) -> None:
        from fastvideo.train.utils.inference_checkpoint import module_export_hooks, validate_complete_inference_checkpoint

        output = Path(self.output_dir).resolve()
        root = output / "inference"
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Unsafe inference retention namespace: {root}")
        protected: set[Path] = set()
        input_roots: set[Path] = set()
        snapshots: dict[Path, tuple[int, ...]] = {}
        try:
            inputs = [base_model_path]
            inputs.extend(model["init_from"] for model in (self._raw_config or {}).get("models", {}).values()
                          if model.get("init_from"))
            for source in inputs:
                dependencies, resolved = _trace_inference_dependency(Path(source), snapshots)
                protected.update(dependencies)
                input_roots.add(resolved)
            # Trace every hop, including aliases outside this namespace and
            # symlinks in directory components. An export's own link is not an
            # external dependency, but every other checkpoint owner is.
            for child in root.iterdir():
                if child.is_symlink() or not child.is_dir():
                    continue
                for component in child.iterdir():
                    identity = _inference_dependency_identity(component)
                    if component in snapshots and snapshots[component] != identity:
                        raise RuntimeError(f"Inference dependency changed during traversal: {component}")
                    snapshots[component] = identity
                    if stat.S_ISLNK(identity[2]):
                        dependencies, _ = _trace_inference_dependency(component, snapshots)
                        protected.update(path for path in dependencies if path != child and child not in path.parents)
        except (OSError, RuntimeError) as error:
            logger.warning("Skipping inference retention: uncertain dependency: %s", error)
            return
        shape_hook = None
        if len(modules) == 1:
            shape_hook, _ = module_export_hooks(next(iter(modules.values())))
        candidates = []
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir() or not _CHECKPOINT_DIR_RE.fullmatch(child.name):
                continue
            if (any(child == source or child in source.parents for source in protected)
                    or any(source == child or source in child.parents for source in input_roots)):
                logger.info("Retaining inference dependency (may exceed keep=%s): %s",
                            self.config.inference_checkpoints_total_limit, child)
                continue
            try:
                if (child / "metadata.json").is_symlink() or (child / ".complete").is_symlink():
                    continue
                metadata = json.loads((child / "metadata.json").read_text(encoding="utf-8"))
                saved_output = metadata["config"]["training"]["checkpoint"]["output_dir"]
                if Path(saved_output).resolve() != output or metadata.get("role") != self.config.inference_role:
                    continue
                module_name = metadata.get("module")
                if module_name not in modules or (child / module_name).is_symlink():
                    continue
                step = _parse_step_from_dir(child)
                if validate_complete_inference_checkpoint(child, step=step, expected_shape_override=shape_hook) is None:
                    continue
                candidates.append((step, child, child.stat()))
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                # Unrecognized/invalid exports are diagnostic debris, not ours to delete.
                continue
        candidates.sort(key=lambda item: item[0])
        for step, path, export_identity in candidates[:-self.config.inference_checkpoints_total_limit]:
            try:
                if any(_inference_dependency_identity(node) != saved for node, saved in snapshots.items()):
                    raise RuntimeError("Inference dependency changed before pruning")
            except (OSError, RuntimeError) as error:
                logger.warning("Skipping inference retention: uncertain dependency: %s", error)
                return
            if step >= published_step or path.is_symlink():
                continue
            current = path.stat()
            if (current.st_dev, current.st_ino) != (export_identity.st_dev, export_identity.st_ino):
                continue
            logger.info("Removing old inference checkpoint (keep=%s): %s",
                        self.config.inference_checkpoints_total_limit, path)
            shutil.rmtree(path)
            snapshots = {node: saved for node, saved in snapshots.items() if node != path and path not in node.parents}

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
            require_complete_marker=self.config.require_complete_training_checkpoint,
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
            require_complete_marker=self.config.require_complete_training_checkpoint,
        )
        if resolved is None:
            return None
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
            # In strict mode, a directory without the post-RNG publication
            # marker is diagnostic debris, not one of the rolling resumable
            # checkpoints. It must not consume ``keep_last`` and displace an
            # older checkpoint that can actually be resumed.
            if (self.config.require_complete_training_checkpoint
                    and not _is_complete_training_checkpoint(child, require_complete_marker=True)):
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
