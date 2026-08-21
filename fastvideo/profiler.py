# SPDX-License-Identifier: Apache-2.0
"""Utilities for managing the PyTorch profiler within FastVideo.

The profiler is shared across the process; this module adds a light-weight
controller that gates collection based on named *regions*. Regions may be
enabled through dedicated environment variables (e.g.
``FASTVIDEO_TORCH_PROFILE_MODEL_LOADING=1``) or via the consolidated
``FASTVIDEO_TORCH_PROFILE_REGIONS`` comma-separated list. Short names work
(``FASTVIDEO_TORCH_PROFILE_REGIONS=model_loading,training_train`` resolves the
``profiler_region_`` prefix automatically).

Typical usage from client code::

    controller = TorchProfilerController(profiler, activities)
    with controller.region("training_dit"):
        run_training_step()

To introduce a new region, register it via :func:`register_profiler_region`
and wrap the corresponding code in :meth:`TorchProfilerController.region`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
import functools
from collections.abc import Iterable

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_GLOBAL_CONTROLLER: TorchProfilerController | None = None


@dataclass(frozen=True)
class ProfilerRegion:
    """Metadata describing a profiler region."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError(f"Profiler region name must be non-empty without surrounding whitespace: {self.name!r}")
        if not self.name.islower():
            raise ValueError(f"Profiler region name must be lower-case: {self.name!r}")


_REGISTERED_REGIONS: dict[str, ProfilerRegion] = {}


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def register_profiler_region(
    name: str,
    description: str,
) -> None:
    """Register a profiler region so configuration can validate inputs."""

    canonical = _normalize_token(name)
    if canonical in _REGISTERED_REGIONS:
        raise ValueError(f"Profiler region {name!r} is already registered")

    region = ProfilerRegion(
        name=canonical,
        description=description,
    )
    _REGISTERED_REGIONS[canonical] = region


def resolve_profiler_region(name: str) -> ProfilerRegion | None:
    """Return the registered region for ``name`` (long or short form).

    Accepts both the canonical name and the short form without the
    ``profiler_region_`` prefix, so ``REGIONS=inference_denoising`` works.
    """

    canonical = _normalize_token(name)
    region = _REGISTERED_REGIONS.get(canonical)
    if region is None and not canonical.startswith("profiler_region_"):
        region = _REGISTERED_REGIONS.get(f"profiler_region_{canonical}")
    return region


def list_profiler_regions() -> list[ProfilerRegion]:
    """Return all registered profiler regions sorted by canonical name."""

    return [_REGISTERED_REGIONS[name] for name in sorted(_REGISTERED_REGIONS)]


_DEFAULT_ACTIVITIES: tuple[torch.profiler.ProfilerActivity, ...] = (
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
)


def get_global_controller() -> TorchProfilerController | None:
    return _GLOBAL_CONTROLLER


def set_global_controller(controller: TorchProfilerController | None) -> None:
    global _GLOBAL_CONTROLLER
    _GLOBAL_CONTROLLER = controller


register_profiler_region(
    name="profiler_region_model_loading",
    description="Module/model loading during pipeline initialization.",
)
# register_profiler_region(
#     name="profiler_region_inference_pre_denoising",
#     description="Pre-denoising inference steps (conditioning, preprocessing).",
# )
register_profiler_region(
    name="profiler_region_inference_denoising",
    description="The main inference denoising loop.",
)
# register_profiler_region(
#     name="profiler_region_inference_post_denoising",
#     description=
#     "Post-processing after denoising (decoder, conditioning restores).",
# )
register_profiler_region(
    name="profiler_region_training_save_checkpoint",
    description="Training save checkpoint operations.",
)

# general training related regions
register_profiler_region(
    name="profiler_region_training_validation",
    description="Validation loop during training.",
)
register_profiler_region(
    name="profiler_region_training_train_one_step",
    description="Single optimizer step including forward/backward passes.",
)
register_profiler_region(
    name="profiler_region_training_train",
    description="High-level step orchestration in the training loop.",
)

# distillation specific regions
register_profiler_region(
    name="profiler_region_distillation_teacher_forward",
    description="Teacher model forward pass in distillation pipelines.",
)
register_profiler_region(
    name="profiler_region_distillation_student_forward",
    description="Student model forward pass in distillation pipelines.",
)
register_profiler_region(
    name="profiler_region_distillation_loss",
    description="Distillation loss computation and aggregation.",
)
register_profiler_region(
    name="profiler_region_distillation_update",
    description="Parameter updates specific to distillation workflows.",
)


def get_or_create_profiler(trace_dir: str | None) -> TorchProfilerController:
    """Create or reuse the process-wide torch profiler controller."""

    existing = get_global_controller()
    if existing is not None:
        if trace_dir:
            logger.info("Reusing existing global torch profiler controller")
        return existing

    if not trace_dir:
        logger.info("Torch profiler disabled; returning no-op controller")
        return TorchProfilerController(None, _DEFAULT_ACTIVITIES, disabled=True)

    logger.info("Profiling enabled. Traces will be saved to: %s", trace_dir)
    logger.info(
        "Profiler config: record_shapes=%s, profile_memory=%s, with_stack=%s, with_flops=%s",
        envs.FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES,
        envs.FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY,
        envs.FASTVIDEO_TORCH_PROFILER_WITH_STACK,
        envs.FASTVIDEO_TORCH_PROFILER_WITH_FLOPS,
    )
    logger.info("FASTVIDEO_TORCH_PROFILE_REGIONS=%s", envs.FASTVIDEO_TORCH_PROFILE_REGIONS)

    profiler = torch.profiler.profile(
        activities=_DEFAULT_ACTIVITIES,
        record_shapes=envs.FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES,
        profile_memory=envs.FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY,
        with_stack=envs.FASTVIDEO_TORCH_PROFILER_WITH_STACK,
        with_flops=envs.FASTVIDEO_TORCH_PROFILER_WITH_FLOPS,
        # No schedule: nothing in the codebase calls profiler.step(), so a
        # wait/warmup schedule never advances and the profiler records nothing.
        # Region toggling gates collection; the single trace exports at stop().
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=True),
    )
    controller = TorchProfilerController(profiler, _DEFAULT_ACTIVITIES)
    controller._trace_dir = trace_dir
    controller.start()
    # The trace only exports at stop(); inference paths have no shutdown hook
    # that calls it, so register one. stop() is idempotent.
    import atexit
    atexit.register(controller.stop)
    logger.info("Torch profiler started")
    return controller


@dataclass
class TorchProfilerConfig:
    """Configuration for torch profiler region control.

    Use :meth:`from_env` to construct an instance with defaults inherited from
    registered regions and optional overrides from the
    ``FASTVIDEO_TORCH_PROFILE_REGIONS`` environment variable. The resulting
    ``regions`` map is consumed by :class:`TorchProfilerController` to decide
    when collection should be enabled.
    """

    regions: dict[str, bool]

    @classmethod
    def from_env(cls) -> TorchProfilerConfig:
        """Build a configuration from process environment variables."""

        requested_regions = {
            token.strip()
            for token in (getattr(envs, "FASTVIDEO_TORCH_PROFILE_REGIONS", "") or "").split(",") if token.strip()
        }

        if not requested_regions:
            available = ", ".join(region.name for region in list_profiler_regions())
            raise ValueError("FASTVIDEO_TORCH_PROFILE_REGIONS must list at least one region; "
                             f"available regions: {available}")

        regions: dict[str, bool] = {}
        available_regions = list_profiler_regions()
        available_names = ", ".join(region.name for region in available_regions)

        for token in requested_regions:
            resolved = resolve_profiler_region(token)
            if resolved is None:
                logger.warning("Unknown profiler region '%s'; available regions: %s", token, available_names)
                continue
            regions[resolved.name] = True

        if not regions:
            raise ValueError("FASTVIDEO_TORCH_PROFILE_REGIONS did not match any known regions; "
                             f"requested={sorted(requested_regions)}, available={available_names}")

        return cls(regions=regions)

    def __str__(self) -> str:
        return f"TorchProfilerConfig(regions={self.regions})"


class TorchProfilerController:
    """Helper that toggles torch profiler collection for named regions.

    Parameters
    ----------
    profiler:
        The shared :class:`torch.profiler.profile` instance, or ``None`` if
        profiling is disabled.
    activities:
        Iterable of :class:`torch.profiler.ProfilerActivity` recorded by the
        profiler.
    config:
        Optional :class:`TorchProfilerConfig`. If omitted, :meth:`from_env`
        constructs one during initialization.

    Examples
    --------
    Enabling an existing region from the command line::

        FASTVIDEO_TORCH_PROFILE_REGIONS=model_loading,training_dit \
        python fastvideo/training/wan_training_pipeline.py ...

    Wrapping a code block in a registered region::

        from fastvideo.profiler import profiler_region

        with profiler_region("inference_denoising"):
            run_denoising_loop()

    Adding a new region requires two steps:
      1. Register it with ``register_profiler_region`` in this module.
      2. Wrap the target code in :func:`profiler_region` using the new name.
    """

    def __init__(
        self,
        profiler: Any,
        activities: Iterable[torch.profiler.ProfilerActivity],
        config: TorchProfilerConfig | None = None,
        disabled: bool = False,
    ) -> None:
        activities_tuple = tuple(activities)
        existing = get_global_controller()
        if existing is not None and not disabled:
            raise RuntimeError("TorchProfilerController already initialized globally. Use get_global_controller().")
        if disabled:
            self._profiler = None
            return

        self._profiler = profiler
        self._activities = activities_tuple
        self._config = config or TorchProfilerConfig.from_env()
        # torch.profiler collects from start(); reflect that so the initial
        # _set_collection(False) in start() actually toggles it off instead of
        # short-circuiting (which captured everything before the first region).
        self._collection_enabled = True
        self._active_region_depth = 0
        self._trace_dir: str | None = None
        logger.info("PROFILER: TorchProfilerController initialized with config: %s", self._config)
        set_global_controller(self)

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when the underlying profiler is collecting."""

        if self._profiler is None:
            return False
        return self._collection_enabled

    def is_region_enabled(self, region: str) -> bool:
        """Return ``True`` if ``region`` should be collected."""

        if self._profiler is None:
            return False
        resolved = resolve_profiler_region(region)
        if resolved is None:
            return False
        return self._config.regions.get(resolved.name, False)

    def _set_collection(self, enabled: bool) -> None:
        if self._profiler is None:
            return
        if self._collection_enabled == enabled:
            return
        event = ("fastvideo.profiler.enable_collection" if enabled else "fastvideo.profiler.disable_collection")
        with torch.profiler.record_function(event):
            self._profiler.toggle_collection_dynamic(enabled, self._activities)
        self._collection_enabled = enabled

    _warned_unregistered: set[str] = set()

    @contextlib.contextmanager
    def region(self, region: str):
        """Context manager that enables profiling for ``region`` if configured."""

        if self._profiler is None:
            yield
            return

        if resolve_profiler_region(region) is None:
            # a typo here would otherwise silently profile nothing, forever
            if region not in self._warned_unregistered:
                self._warned_unregistered.add(region)
                logger.warning("PROFILER: region %r is not registered (typo?); available: %s", region,
                               ", ".join(r.name for r in list_profiler_regions()))
            yield
            return

        if not self.is_region_enabled(region):
            yield
            return

        # NVTX range so the same region names are visible in nsys timelines
        nvtx = torch.cuda.is_available()
        if nvtx:
            torch.cuda.nvtx.range_push(f"fastvideo.region::{region}")
        self._active_region_depth += 1
        if self._active_region_depth == 1:
            logger.info("PROFILER: Setting collection to True (depth=%s) for region %s", self._active_region_depth,
                        region)
            self._set_collection(True)
        try:
            # record_function opens after collection is enabled so the region
            # marker itself lands in the trace.
            with torch.profiler.record_function(f"fastvideo.region::{region}"):
                yield
        finally:
            self._active_region_depth -= 1
            logger.info("PROFILER: Decreasing active region depth to %s", self._active_region_depth)
            if self._active_region_depth == 0:
                logger.info("PROFILER: Setting collection to False upon exiting region %s", region)
                self._set_collection(False)
            if nvtx:
                torch.cuda.nvtx.range_pop()

    def start(self) -> None:
        """Start the profiler and pause collection until a region is entered."""

        logger.info("PROFILER: Starting profiler...")
        if self._profiler is None:
            return
        self._profiler.start()
        logger.info("PROFILER: Profiler started")
        # Profiler starts with collection disabled by default.
        logger.info("PROFILER: Setting collection to False")
        self._set_collection(False)
        logger.info("PROFILER: Profiler started with collection disabled")

    def _write_summary(self) -> None:
        """Compact per-rank op summary next to the trace: a key_averages table
        and a JSON with input shapes, so operator-split analysis does not
        require parsing multi-GB chrome traces."""
        if self._profiler is None or not self._trace_dir:
            return
        try:
            import json as _json
            import os as _os
            rank = _os.environ.get("RANK", "0")
            averages = self._profiler.key_averages(group_by_input_shape=True)
            stem = _os.path.join(self._trace_dir, f"summary_rank{rank}")
            with open(f"{stem}.txt", "w") as fh:
                fh.write(averages.table(sort_by="self_device_time_total", row_limit=60))
            rows = [{
                "name": e.key,
                "shapes": str(e.input_shapes),
                "self_device_us": e.self_device_time_total,
                "device_us": e.device_time_total,
                "count": e.count,
            } for e in averages]
            with open(f"{stem}.json", "w") as fh:
                _json.dump(rows, fh)
            if rank == "0":
                logger.info("PROFILER: summary written to %s.txt", stem)
        except Exception:  # noqa: BLE001 -- summaries must never break shutdown
            logger.exception("PROFILER: summary generation failed")

    def stop(self) -> None:
        """Stop the profiler after disabling collection and clearing state."""

        if self._profiler is None:
            return

        logger.info("PROFILER: Stopping profiler...")
        self._profiler.stop()
        self._write_summary()
        self._profiler = None  # makes stop() idempotent (atexit may re-enter)
        logger.info("PROFILER: Profiler stopped")
        self._active_region_depth = 0
        set_global_controller(None)

    @property
    def has_profiler(self) -> bool:
        """Return ``True`` when a profiler instance is available."""

        return self._profiler is not None

    @property
    def activities(self) -> tuple[torch.profiler.ProfilerActivity, ...]:
        return tuple(self._activities)


@contextlib.contextmanager
def profiler_region(region: str):
    """Module-level region context manager bound to the global controller.

    A no-op when no profiler is configured — stages can use this without the
    get_global_controller()/nullcontext dance.
    """
    controller = get_global_controller()
    if controller is None:
        yield
        return
    with controller.region(region):
        yield


def profile_region(region: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a bound method so it runs inside a profiler region if available."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            controller = getattr(self, "profiler_controller", None)
            if controller is None or not controller.has_profiler:
                return fn(self, *args, **kwargs)
            with controller.region(region):
                return fn(self, *args, **kwargs)

        return wrapped

    return decorator
