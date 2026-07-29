# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture for the causal Wan denoising step.

Ported from NVIDIA FlashDreams (https://github.com/NVIDIA/flashdreams,
``flashdreams/infra/cuda_graph.py``), trimmed to what this repo needs: no
``drain()`` step, since capture here isn't yet combined with
``torch.compile`` (see FlashDreams.md item 2 for why that's deferred).
"""
from typing import Any
from collections.abc import Callable

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class CausalCudaGraphWrapper:
    """Capture a stateful CUDA callable into a replayable graph.

    The wrapped callable runs eagerly for ``warmup_iters`` calls so kernels
    JIT-load and the allocator stabilizes. The next call captures the whole
    forward into a ``torch.cuda.CUDAGraph`` against static input buffers;
    every same-shape call after that copies inputs into those buffers and
    replays the graph, returning clones of the captured outputs.

    Only usable where the wrapped callable reads/writes memory through
    stable pointers on every call (e.g. a KV cache updated by slice
    assignment) and where any plain-Python arguments (ints, bools, ``None``)
    don't change which memory the callable touches -- for the causal Wan
    denoising step, this holds once the KV cache window has filled and
    ``rope_cache_policy`` is ``"relativistic"``; see ``causal_denoising.py``
    for the gating that ensures that before this wrapper is ever used.

    Input staging: only top-level tensor positional args and kwargs are
    copied into static buffers. Everything else (ints, ``None``, dicts,
    lists) passes through to the callable verbatim on the eager/warmup path
    and is simply not re-passed on replay, since a replay only reruns the
    kernels recorded during capture -- it never calls the wrapped function
    again. This is why the KV/cross-attention caches (mutated in place
    through existing tensor storage) work correctly here, while a fresh
    per-call value like the current chunk's noisy latents needs staging.
    """

    def __init__(self, fn: Callable[..., Any], warmup_iters: int = 2) -> None:
        self.fn = fn
        self.warmup_iters = warmup_iters
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_args: list[Any] = []
        self._static_kwargs: dict[str, Any] = {}
        self._static_out_leaves: list[Any] | None = None
        self._out_spec: Any = None
        self._warmup_remaining = warmup_iters

    def reset(self) -> None:
        """Drop the captured graph and staged buffers, restarting warmup."""
        self._graph = None
        self._static_args = []
        self._static_kwargs = {}
        self._static_out_leaves = None
        self._out_spec = None
        self._warmup_remaining = self.warmup_iters

    @staticmethod
    def _slot_compatible(slot: Any, fresh: Any) -> bool:
        """Return whether ``slot``'s static buffer can absorb ``fresh``."""
        if isinstance(slot, torch.Tensor):
            return (isinstance(fresh, torch.Tensor) and slot.shape == fresh.shape and slot.dtype == fresh.dtype)
        return not isinstance(fresh, torch.Tensor)

    def _slots_compatible_with(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if len(self._static_args) != len(args):
            return False
        if set(self._static_kwargs) != set(kwargs):
            return False
        for slot, fresh in zip(self._static_args, args, strict=False):
            if not self._slot_compatible(slot, fresh):
                return False
        return all(self._slot_compatible(slot, kwargs[name]) for name, slot in self._static_kwargs.items())

    @staticmethod
    def _make_slot(value: Any) -> Any:
        """Return a static buffer for a tensor, or ``value`` itself otherwise."""
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value).contiguous()
        return value

    def _stage(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Copy top-level tensors into static buffers; forward everything else as-is.

        Reallocates buffers (and drops any captured graph) if the staged
        tensor signature changes shape or dtype.
        """
        if not self._slots_compatible_with(args, kwargs):
            self.reset()
            self._static_args = [self._make_slot(a) for a in args]
            self._static_kwargs = {k: self._make_slot(v) for k, v in kwargs.items()}

        staged_args: list[Any] = []
        for slot, fresh in zip(self._static_args, args, strict=False):
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_args.append(slot)
            else:
                staged_args.append(fresh)

        staged_kwargs: dict[str, Any] = {}
        for name, fresh in kwargs.items():
            slot = self._static_kwargs[name]
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_kwargs[name] = slot
            else:
                staged_kwargs[name] = fresh

        return tuple(staged_args), staged_kwargs

    def _clone_output(self) -> Any:
        assert self._static_out_leaves is not None and self._out_spec is not None
        cloned = [leaf.clone() if isinstance(leaf, torch.Tensor) else leaf for leaf in self._static_out_leaves]
        return tree_unflatten(cloned, self._out_spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = self._stage(args, kwargs)

        if self._graph is not None:
            self._graph.replay()
            # Clone: the next replay would overwrite the static output buffer.
            return self._clone_output()

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return self.fn(*args, **kwargs)

        # Capture: trace one full forward against the static buffers.
        # cudaStreamBeginCapture only records kernels -- it doesn't execute
        # them -- so the static outputs and in-place cache writes are no-ops
        # here. Replay once immediately to actually compute this call's
        # output and advance the cache.
        #
        # Keep the graph local until capture and the first replay both
        # succeed. If capture fails and the caller catches the exception, a
        # stored-but-invalid CUDAGraph would make the next call fail with
        # "replay without a preceding successful capture", hiding the real
        # capture error.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.fn(*args, **kwargs)
        out_leaves, out_spec = tree_flatten(out)
        graph.replay()
        self._graph = graph
        self._out_spec = out_spec
        self._static_out_leaves = out_leaves
        logger.info("CausalCudaGraphWrapper: captured CUDA graph for %s", getattr(self.fn, "__qualname__", self.fn))
        return self._clone_output()


class CausalCudaGraphDispatch:
    """Route a transformer call through the right CUDA graph once steady state is reached.

    A chunk-start step evicts the oldest cached frames before writing the
    new ones; every other step just overwrites the existing slot. Those are
    two different fixed operation sequences, so each needs its own captured
    graph -- replaying one for the other would silently apply the wrong kind
    of cache update. Both graphs are only ever engaged once the KV cache has
    reached steady state, where each kind of step's sequence stops changing
    from call to call (see FlashDreams.md item 2).

    Construct one of these per ``forward()`` call, not once per stage
    instance: a captured graph's static buffers are tied to that call's
    specific KV cache tensors, and reusing one across generations would
    replay against stale memory once a fresh KV cache is allocated.
    """

    def __init__(self, transformer: Callable[..., Any], *, enabled: bool, warmup_iters: int = 2) -> None:
        self.enabled = enabled
        self._chunk_start = CausalCudaGraphWrapper(transformer, warmup_iters=warmup_iters) if enabled else None
        self._continuation = CausalCudaGraphWrapper(transformer, warmup_iters=warmup_iters) if enabled else None

    def call(self, transformer: Callable[..., Any], *args: Any, is_chunk_start: bool | None, is_steady_state: bool,
             **kwargs: Any) -> Any:
        """Call ``transformer(*args, is_chunk_start=is_chunk_start, **kwargs)``, through a graph if steady state allows it."""
        kwargs = dict(kwargs, is_chunk_start=is_chunk_start)
        if self.enabled and is_steady_state:
            wrapper = self._chunk_start if is_chunk_start else self._continuation
            assert wrapper is not None, "wrappers are built whenever enabled is True"
            return wrapper(*args, **kwargs)
        return transformer(*args, **kwargs)
