# SPDX-License-Identifier: Apache-2.0
"""Utilities for managing the PyTorch profiler within FastVideo.

The profiler is shared across the process; this module adds a light-weight
controller that gates collection based on named *regions*. Regions are enabled
through the ``FASTVIDEO_TORCH_PROFILE_REGIONS`` comma-separated list. Short names work
(``FASTVIDEO_TORCH_PROFILE_REGIONS=model_loading,training_train`` resolves the
``profiler_region_`` prefix automatically).

Typical usage from client code::

    controller = get_or_create_profiler("/tmp/fastvideo-traces")
    with controller.region("training_train"):
        run_training_step()
    controller.stop()

To introduce a new region, register it via :func:`register_profiler_region`
and wrap the corresponding code in :meth:`TorchProfilerController.region`.
"""

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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
    name="profiler_region_training_forward",
    description="Training method forward pass and loss computation.",
)
register_profiler_region(
    name="profiler_region_training_dataloader",
    description="Fetch the next training batch in the trainer process.",
)
register_profiler_region(
    name="profiler_region_training_backward",
    description="Training backward pass.",
)
register_profiler_region(
    name="profiler_region_training_optimizer",
    description="Gradient clipping, optimizer/scheduler steps, and zero_grad.",
)
register_profiler_region(
    name="profiler_region_training_callbacks",
    description="End-of-step training callbacks such as EMA updates.",
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

# DMD2 method regions. These sit inside ``training_forward`` and make the
# method's multi-model forward path distinguishable in a single trace.
register_profiler_region(
    name="profiler_region_dmd2_student_rollout",
    description="DMD2 student rollout, including its simulated prefix steps.",
)
register_profiler_region(
    name="profiler_region_dmd2_generator_loss",
    description="DMD2 generator loss, including teacher and critic scoring.",
)
register_profiler_region(
    name="profiler_region_dmd2_critic_loss",
    description="DMD2 critic flow-matching loss, including its student rollout.",
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

    def profiler_factory() -> Any:
        return torch.profiler.profile(
            activities=_DEFAULT_ACTIVITIES,
            record_shapes=envs.FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES,
            profile_memory=envs.FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY,
            with_stack=envs.FASTVIDEO_TORCH_PROFILER_WITH_STACK,
            with_flops=envs.FASTVIDEO_TORCH_PROFILER_WITH_FLOPS,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=True),
        )

    controller = TorchProfilerController(
        None,
        _DEFAULT_ACTIVITIES,
        profiler_factory=profiler_factory,
        trace_dir=trace_dir,
    )
    controller.start()
    # Region exit normally exports each trace segment. Keep an atexit hook for
    # exceptions or process shutdown while a region is still active.
    import atexit
    atexit.register(controller.stop)
    logger.info("Torch profiler armed; collection starts at the first enabled region")
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
    """Create complete torch-profiler trace segments for named regions.

    PyTorch's dynamic CUDA collection toggle can fail to re-enable CUPTI on
    some supported stacks. In that failure mode it emits CPU operators while
    silently dropping every CUDA kernel. This controller therefore starts a
    fresh profiler at each outermost enabled region and stops it when that
    region exits. Nested enabled regions become annotations in the same
    CPU/CUDA trace segment.

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
    profiler_factory:
        Factory for fresh profiler instances. Required to profile more than
        one outermost region invocation.
    trace_dir:
        Directory for per-segment summaries.

    Examples
    --------
    Enabling an existing region from the command line::

        FASTVIDEO_TORCH_PROFILE_REGIONS=model_loading,training_train \
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
        profiler_factory: Callable[[], Any] | None = None,
        trace_dir: str | None = None,
    ) -> None:
        activities_tuple = tuple(activities)
        existing = get_global_controller()
        if existing is not None and not disabled:
            raise RuntimeError("TorchProfilerController already initialized globally. Use get_global_controller().")
        self._profiler = profiler
        self._profiler_factory = profiler_factory
        self._activities = activities_tuple
        self._trace_dir = trace_dir
        self._segment_index = 0
        self._segment_region: str | None = None
        self._active_region_depth = 0
        self._collection_enabled = False
        if disabled:
            self._configured = False
            self._armed = False
            return

        self._config = config or TorchProfilerConfig.from_env()
        self._configured = True
        self._armed = False
        logger.info("PROFILER: TorchProfilerController initialized with config: %s", self._config)
        set_global_controller(self)

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when the underlying profiler is collecting."""

        return self._collection_enabled

    def is_region_enabled(self, region: str) -> bool:
        """Return ``True`` if ``region`` should be collected."""

        if not self.has_profiler:
            return False
        resolved = resolve_profiler_region(region)
        if resolved is None:
            return False
        return self._config.regions.get(resolved.name, False)

    def _new_profiler(self) -> Any:
        if self._profiler is not None:
            profiler = self._profiler
            self._profiler = None
            return profiler
        if self._profiler_factory is None:
            raise RuntimeError("Torch profiler cannot start another trace segment without a profiler_factory")
        return self._profiler_factory()

    def _start_segment(self, region: str) -> None:
        if self._collection_enabled:
            return
        self._profiler = self._new_profiler()
        logger.info(
            "PROFILER: Starting segment %d for region %s",
            self._segment_index,
            region,
        )
        self._profiler.start()
        self._segment_region = region
        self._collection_enabled = True

    def _finish_segment(self) -> None:
        if self._profiler is None or not self._collection_enabled:
            return
        profiler = self._profiler
        segment_index = self._segment_index
        segment_region = self._segment_region or "unknown"
        logger.info(
            "PROFILER: Stopping segment %d for region %s",
            segment_index,
            segment_region,
        )
        profiler.stop()
        self._write_summary(
            profiler,
            segment_index=segment_index,
            segment_region=segment_region,
        )
        self._profiler = None
        self._collection_enabled = False
        self._segment_region = None
        self._segment_index += 1

    _warned_unregistered: set[str] = set()

    @contextlib.contextmanager
    def region(self, region: str):
        """Context manager that enables profiling for ``region`` if configured."""

        if not self.has_profiler:
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

        if self._active_region_depth == 0:
            self._start_segment(region)

        # NVTX range so the same region names are visible in nsys timelines.
        # Push after profiler startup so Kineto also records the annotation.
        nvtx = torch.cuda.is_available()
        if nvtx:
            torch.cuda.nvtx.range_push(f"fastvideo.region::{region}")
        self._active_region_depth += 1
        try:
            with torch.profiler.record_function(f"fastvideo.region::{region}"):
                yield
        finally:
            self._active_region_depth -= 1
            try:
                if nvtx:
                    torch.cuda.nvtx.range_pop()
            finally:
                # Close NVTX before stopping Kineto so both profilers see a
                # balanced outermost range in the exported segment.
                if self._active_region_depth == 0:
                    self._finish_segment()

    def start(self) -> None:
        """Arm the controller; collection begins at an enabled region."""

        if not self._configured:
            return
        self._armed = True
        logger.info("PROFILER: Controller armed")

    def _write_summary(
        self,
        profiler: Any,
        *,
        segment_index: int,
        segment_region: str,
    ) -> None:
        """Compact per-rank op summary next to the trace: a key_averages table
        and a JSON with input shapes, so operator-split analysis does not
        require parsing multi-GB chrome traces."""
        if not self._trace_dir:
            return
        try:
            import json as _json
            rank = os.environ.get("RANK", "0")
            short_region = segment_region.removeprefix("profiler_region_")
            stem = os.path.join(
                self._trace_dir,
                f"summary_rank{rank}_segment{segment_index:04d}_{short_region}",
            )
            averages = profiler.key_averages(group_by_input_shape=envs.FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES, )
            with open(f"{stem}.txt", "w", encoding="utf-8") as fh:
                fh.write(averages.table(sort_by="self_device_time_total", row_limit=60))
            rows = [{
                "name": e.key,
                "shapes": str(e.input_shapes),
                "self_cpu_us": e.self_cpu_time_total,
                "cpu_us": e.cpu_time_total,
                "self_device_us": e.self_device_time_total,
                "device_us": e.device_time_total,
                "count": e.count,
            } for e in averages]
            with open(f"{stem}.json", "w", encoding="utf-8") as fh:
                _json.dump(rows, fh)
            if rank == "0":
                logger.info("PROFILER: summary written to %s.txt", stem)
        except Exception:  # noqa: BLE001 -- summaries must never break shutdown
            logger.exception("PROFILER: summary generation failed")

    def stop(self) -> None:
        """Flush any active segment and disable this controller."""

        if not self._configured:
            return

        logger.info("PROFILER: Stopping profiler...")
        self._finish_segment()
        self._profiler = None
        self._configured = False
        self._armed = False
        logger.info("PROFILER: Profiler stopped")
        self._active_region_depth = 0
        set_global_controller(None)

    @property
    def has_profiler(self) -> bool:
        """Return ``True`` when this controller is configured and armed."""

        return self._configured and self._armed

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
    """Wrap a bound method so it runs inside a profiler region if available.

    Prefer a controller attached to the instance, then fall back to the
    process-wide controller. The fallback lets lightweight owners such as the
    modular trainer and its callbacks add regions without threading profiler
    plumbing through their public constructors.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            controller = getattr(self, "profiler_controller", None)
            if controller is None:
                controller = get_global_controller()
            if controller is None or not controller.has_profiler:
                return fn(self, *args, **kwargs)
            with controller.region(region):
                return fn(self, *args, **kwargs)

        return wrapped

    return decorator
