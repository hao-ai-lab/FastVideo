# SPDX-License-Identifier: Apache-2.0
"""Request-local CUDA graphs for fixed-window causal Wan inference.

See docs/inference/optimizations.md for the cache invariant and restrictions.
Warmup calls are real sampling calls: never run extra forwards on the live cache.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _signature(value: Any, *, reference: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        layout = (value.shape, value.dtype, value.device, value.stride())
        return (*layout, value.data_ptr()) if reference else layout
    if isinstance(value, list | tuple):
        return (type(value), tuple(_signature(item, reference=reference) for item in value))
    if isinstance(value, dict):
        # Logical positions advance outside replay; cache storage stays fixed.
        return tuple((key, _signature(item, reference=reference)) for key, item in value.items()
                     if key not in {"global_end_index", "local_end_index"})
    return (type(value), value)


class CausalCudaGraphWrapper:
    """Stage changing tensors while retaining conditioning and cache storage.

    Positional argument 1 is immutable conditioning. KV/cross-attention caches
    are references, not copies. Logical frame/token positions may change only
    because the caller has established constant physical offsets and RoPE.
    """

    def __init__(self, fn: Callable[..., Any], *, warmup_iters: int = 2, clone_output: bool = True) -> None:
        self.fn = fn
        self.warmup_iters = warmup_iters
        self.clone_output = clone_output
        self.capture_count = 0
        self.replay_count = 0
        self.reset_count = 0
        self.reset()

    def reset(self) -> None:
        self._graph: torch.cuda.CUDAGraph | None = None
        self._input_signature: Any = None
        self._static_args: list[Any] = []
        self._static_kwargs: dict[str, Any] = {}
        self._out_leaves: list[Any] = []
        self._out_spec: Any = None
        self._warmup_remaining = self.warmup_iters
        self._warmup_stream: torch.cuda.Stream | None = None

    @staticmethod
    def _make_slot(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value, memory_format=torch.contiguous_format)
        if isinstance(value, list):
            return [CausalCudaGraphWrapper._make_slot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(CausalCudaGraphWrapper._make_slot(item) for item in value)
        if isinstance(value, dict):
            return {key: CausalCudaGraphWrapper._make_slot(item) for key, item in value.items()}
        return value

    @staticmethod
    def _copy_slot(slot: Any, fresh: Any) -> None:
        if isinstance(slot, torch.Tensor):
            slot.copy_(fresh)
        elif isinstance(slot, list | tuple):
            for item, new_item in zip(slot, fresh, strict=True):
                CausalCudaGraphWrapper._copy_slot(item, new_item)
        elif isinstance(slot, dict):
            for key, item in slot.items():
                CausalCudaGraphWrapper._copy_slot(item, fresh[key])

    def _stage(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        references = {"kv_cache", "crossattn_cache"}
        positions = {"current_start", "cache_start", "start_frame"}
        signature = (
            tuple(_signature(item, reference=index == 1) for index, item in enumerate(args)),
            tuple((key, type(item) if key in positions else _signature(item, reference=key in references))
                  for key, item in kwargs.items()),
        )
        if signature != self._input_signature:
            if self._input_signature is not None:
                self.reset_count += 1
            self.reset()
            self._input_signature = signature
            self._static_args = [item if index == 1 else self._make_slot(item) for index, item in enumerate(args)]
            self._static_kwargs = {
                key: item if key in references else self._make_slot(item)
                for key, item in kwargs.items()
            }
        for index, (slot, fresh) in enumerate(zip(self._static_args, args, strict=True)):
            if index != 1:
                self._copy_slot(slot, fresh)
        for key, fresh in kwargs.items():
            if key not in references:
                self._copy_slot(self._static_kwargs[key], fresh)

    def _output(self) -> Any:
        if not self.clone_output:
            return None  # Context forwards update the cache; their output is unused.
        return tree_unflatten([item.clone() if isinstance(item, torch.Tensor) else item for item in self._out_leaves],
                              self._out_spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args or not isinstance(args[0], torch.Tensor) or args[0].device.type != "cuda":
            raise ValueError("Causal CUDA graphs require CUDA latent inputs")
        self._stage(args, kwargs)
        if self._graph is not None:
            self._graph.replay()
            self.replay_count += 1
            return self._output()
        if self._warmup_remaining:
            current_stream = torch.cuda.current_stream(args[0].device)
            if self._warmup_stream is None:
                self._warmup_stream = torch.cuda.Stream(device=args[0].device)
            self._warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._warmup_stream):
                # Use fresh logical positions during real warmup calls.
                call_kwargs = dict(self._static_kwargs)
                for key in ("current_start", "cache_start", "start_frame"):
                    if key in kwargs:
                        call_kwargs[key] = kwargs[key]
                output = self.fn(*self._static_args, **call_kwargs)
            current_stream.wait_stream(self._warmup_stream)
            for item in tree_flatten(output)[0]:
                if isinstance(item, torch.Tensor):
                    item.record_stream(current_stream)
            self._warmup_remaining -= 1
            return output if self.clone_output else None

        graph = torch.cuda.CUDAGraph()
        # Record the current call's logical positions. They no longer affect
        # physical slices, and Python counters are finalized by the stage.
        for key in ("current_start", "cache_start", "start_frame"):
            if key in kwargs:
                self._static_kwargs[key] = kwargs[key]
        with torch.cuda.graph(graph):
            output = self.fn(*self._static_args, **self._static_kwargs)
        self._out_leaves, self._out_spec = tree_flatten(output)
        # Capture records kernels without executing them; this executes once.
        graph.replay()
        self._graph = graph
        self.capture_count += 1
        self.replay_count += 1
        logger.info("Captured causal Wan CUDA graph (%s)", "denoising" if self.clone_output else "context")
        return self._output()


class CausalCudaGraphDispatch:
    """One dispatcher per request, with distinct eviction, overwrite and context graphs."""

    def __init__(self, transformer: Callable[..., Any], *, enabled: bool, warmup_iters: int = 2) -> None:
        self.transformer = transformer
        self.enabled = enabled
        self.wrappers = {
            name: CausalCudaGraphWrapper(transformer, warmup_iters=warmup_iters, clone_output=name != "context")
            for name in ("chunk_start", "continuation", "context")
        } if enabled else {}

    def call(self,
             transformer: Callable[..., Any],
             *args: Any,
             is_chunk_start: bool | None,
             is_steady_state: bool,
             is_context: bool = False,
             **kwargs: Any) -> Any:
        if self.enabled and is_steady_state:
            if transformer is not self.transformer:
                raise ValueError("Causal CUDA graph dispatcher received a different transformer")
            if is_chunk_start is None:
                raise ValueError("Graph replay requires an explicit chunk-start decision")
            name = "context" if is_context else "chunk_start" if is_chunk_start else "continuation"
            return self.wrappers[name](*args, is_chunk_start=is_chunk_start, **kwargs)
        if is_chunk_start is not None:
            kwargs["is_chunk_start"] = is_chunk_start
        return transformer(*args, **kwargs)

    def statistics(self) -> dict[str, int]:
        return {
            metric: sum(getattr(wrapper, metric) for wrapper in self.wrappers.values())
            for metric in ("capture_count", "replay_count", "reset_count")
        }
