# SPDX-License-Identifier: Apache-2.0
"""Zero-overhead-when-off activation trace mode for FastVideo pipelines.

Enable by setting FASTVIDEO_TRACE_ACTIVATIONS=1. When off, this module
adds zero overhead — no hooks are registered, no branches exist in the
production forward path. When on, registers forward hooks on modules
whose name matches FASTVIDEO_TRACE_LAYERS, computes the requested stats
(FASTVIDEO_TRACE_STATS) on each output tensor, and writes JSONL records
to FASTVIDEO_TRACE_OUTPUT.

Useful for parity debugging across model ports — log on both the
FastVideo path and the upstream reference, diff the two JSONL files
to find the first divergent layer.

Example:

    FASTVIDEO_TRACE_ACTIVATIONS=1 \
    FASTVIDEO_TRACE_LAYERS="^block\\.layers\\.[0-9]+$" \
    FASTVIDEO_TRACE_STATS="abs_mean,max,shape" \
    FASTVIDEO_TRACE_OUTPUT="/tmp/fv_trace.jsonl" \
    python examples/inference/basic/basic_magi_human.py
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterator

import torch
from torch import nn

from fastvideo import envs
from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_TRACE_STATE = threading.local()


def current_step_idx() -> int | None:
    return getattr(_TRACE_STATE, "step_idx", None)


@contextmanager
def trace_step(step_idx: int) -> Iterator[None]:
    """Context manager that sets the current denoise step for trace records."""
    prev = getattr(_TRACE_STATE, "step_idx", None)
    _TRACE_STATE.step_idx = step_idx
    try:
        yield
    finally:
        _TRACE_STATE.step_idx = prev


_STAT_FNS: dict[str, Callable[[torch.Tensor], Any]] = {
    "abs_mean": lambda t: float(t.detach().float().abs().mean().item()),
    "sum": lambda t: float(t.detach().float().sum().item()),
    "min": lambda t: float(t.detach().float().min().item()),
    "max": lambda t: float(t.detach().float().max().item()),
    "mean": lambda t: float(t.detach().float().mean().item()),
    "std": lambda t: float(t.detach().float().std().item()),
    "shape": lambda t: list(t.shape),
    "dtype": lambda t: str(t.dtype),
}


def _resolve_stats(spec: str) -> list[tuple[str, Callable[[torch.Tensor], Any]]]:
    stats = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        stat_fn = _STAT_FNS.get(name)
        if stat_fn is None:
            logger.warning(
                "FASTVIDEO_TRACE_STATS contains unknown stat %r; valid: %s",
                name,
                sorted(_STAT_FNS),
            )
            continue
        stats.append((name, stat_fn))
    return stats


def _resolve_output_path(template: str) -> Path:
    return Path(template.replace("<pid>", str(os.getpid())))


def _parse_step_filter(spec: str) -> set[int] | None:
    if not spec.strip():
        return None
    return {int(s.strip()) for s in spec.split(",") if s.strip()}


class JsonlSink:
    """Buffered JSONL writer with thread-safe append."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # noqa: SIM115
        self._lock = threading.Lock()
        logger.info("Activation trace JSONL sink: %s", self.path)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            self._fh.write(line)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class ActivationStatHook(ForwardHook):
    """Forward hook that emits per-tensor stats to a JSONL sink."""

    def __init__(
        self,
        module_name: str,
        stats: list[tuple[str, Callable[[torch.Tensor], Any]]],
        sink: JsonlSink,
        step_filter: set[int] | None,
    ) -> None:
        self.module_name = module_name
        self.stats = stats
        self.sink = sink
        self.step_filter = step_filter

    def name(self) -> str:
        return "ActivationStatHook"

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        step_idx = current_step_idx()
        if self.step_filter is not None and step_idx not in self.step_filter:
            return output
        for tensor_label, tensor in _flatten_tensors(output):
            record: dict[str, Any] = {
                "module": self.module_name,
                "tensor": tensor_label,
                "step": step_idx,
            }
            for stat_name, stat_fn in self.stats:
                try:
                    record[stat_name] = stat_fn(tensor)
                except Exception as exc:  # pragma: no cover - defensive logging
                    record[stat_name] = f"<error: {exc!r}>"
            self.sink.write(record)
        return output


def _flatten_tensors(obj: Any, prefix: str = "out") -> list[tuple[str, torch.Tensor]]:
    """Yield (label, tensor) pairs from arbitrarily-nested forward outputs."""
    if isinstance(obj, torch.Tensor):
        return [(prefix, obj)]
    if isinstance(obj, tuple | list):
        out = []
        for idx, item in enumerate(obj):
            out.extend(_flatten_tensors(item, f"{prefix}[{idx}]"))
        return out
    if isinstance(obj, dict):
        out = []
        for key, value in obj.items():
            out.extend(_flatten_tensors(value, f"{prefix}.{key}"))
        return out
    return []


class ActivationTraceManager:

    def __init__(self, managers: list[ModuleHookManager], sink: JsonlSink) -> None:
        self.managers = managers
        self.sink = sink

    def remove_from_manager(self) -> None:
        for manager in self.managers:
            if manager.get_forward_hook("ActivationStatHook") is not None:
                manager.remove_forward_hook("ActivationStatHook")
            if not manager.forward_hooks:
                ModuleHookManager.remove_from_manager(manager.module)
        self.sink.close()


def attach_activation_trace(model: nn.Module | None) -> ActivationTraceManager | None:
    """Attach activation-stat hooks to model. Returns None if trace is off."""
    if not envs.FASTVIDEO_TRACE_ACTIVATIONS or model is None:
        return None

    pattern_spec = envs.FASTVIDEO_TRACE_LAYERS
    pattern = re.compile(pattern_spec) if pattern_spec else re.compile(".*")
    stats = _resolve_stats(envs.FASTVIDEO_TRACE_STATS)
    if not stats:
        logger.warning("FASTVIDEO_TRACE_STATS yielded no valid stats; trace disabled.")
        return None

    sink = JsonlSink(_resolve_output_path(envs.FASTVIDEO_TRACE_OUTPUT))
    step_filter = _parse_step_filter(envs.FASTVIDEO_TRACE_STEPS)
    managers = []
    for name, module in model.named_modules():
        if not name or not pattern.search(name):
            continue
        manager = ModuleHookManager.get_from_or_default(module)
        manager.append_forward_hook(ActivationStatHook(name, stats, sink, step_filter))
        managers.append(manager)

    logger.info(
        "Activation trace attached to %d modules (pattern=%r, stats=%s)",
        len(managers),
        pattern_spec,
        [stat_name for stat_name, _ in stats],
    )
    return ActivationTraceManager(managers, sink)


def detach_activation_trace(mgr: ActivationTraceManager | None) -> None:
    if mgr is not None:
        mgr.remove_from_manager()
