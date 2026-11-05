# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-aware recovery from Dynamo cache exhaustion for H3 training."""
from __future__ import annotations

import sys
from collections import Counter
from contextvars import ContextVar
from functools import partial, wraps
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def canonicalize_singleton_batch(x: Any) -> Any:
    """Normalize the unused batch stride without copying large H3 activations.

    A B=1 sequence slice can be contiguous yet retain its allocation's batch
    stride. ``contiguous()`` then does nothing. Only the stride of the singleton
    dimension changes here; storage, addresses, and all non-singleton strides
    remain identical, including for genuinely noncontiguous inputs.
    """
    if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[0] == 1:
        stride = (x.shape[1] * x.stride(1), *x.stride()[1:])
        if x.stride() != stride:
            return x.as_strided(x.shape, stride)
    return x


class _CheckpointPhase:
    """Reusable contexts also support repeated backwards on a retained graph."""

    def __init__(self, active: ContextVar, modes: dict, replay: bool) -> None:
        self.active, self.modes, self.replay = active, modes, replay
        self.tokens: list[Any] = []

    def __enter__(self) -> None:
        self.tokens.append(self.active.set((self.modes, self.replay)))

    def __exit__(self, *exc: Any) -> None:
        self.active.reset(self.tokens.pop())


class CheckpointCompileCache:
    """Bound new compilations; preserve each checkpoint's original execution mode.

    Only FailOnRecompileLimitHit triggers eager execution. Cached graphs continue
    to run, and ordinary compiler/kernel errors propagate. Replaying an already
    compiled checkpoint must never switch to eager: AOTAutograd and eager save
    different tensors. Replay therefore has an emergency allowance to finish
    outstanding compiled forwards, even if a new guard appears only in backward.
    New forward cache misses still obey the configured limits. No cache is reset,
    no graph guards are skipped, and parameters/checkpoint/FSDP modules are kept.
    """

    def __init__(self) -> None:
        self.calls: Counter = Counter()
        self.active: ContextVar = ContextVar("h3_compile_checkpoint", default=None)
        self.originals: list[tuple[Any, str, Any, Any]] = []

    def checkpoint_contexts(self) -> tuple[_CheckpointPhase, _CheckpointPhase]:
        modes: dict[object, bool] = {}
        return _CheckpointPhase(self.active, modes, False), _CheckpointPhase(self.active, modes, True)

    def install_checkpoints(self, transformer: Any) -> None:
        for module in transformer.modules():
            if isinstance(module, CheckpointWrapper):
                original = module.checkpoint_fn
                if "context_fn" in original.keywords:
                    raise ValueError("Compile cache recovery requires an unmodified checkpoint context")
                installed = partial(original, context_fn=self.checkpoint_contexts)
                self.originals.append((module, "checkpoint_fn", original, installed))
                module.checkpoint_fn = installed

    def wrap(self, eager: Any, compiled: Any, compile_kwargs: dict[str, Any], fqn: str) -> Any:
        replay = torch.compile(eager, **{**compile_kwargs, "recompile_limit": sys.maxsize})
        region = object()

        @wraps(eager)
        def forward(*args: Any, **kwargs: Any) -> Any:
            # Do this outside Dynamo so unused singleton strides do not become
            # guards in either dense or sparse H3 block graphs.
            if args:
                args = (canonicalize_singleton_batch(args[0]), *args[1:])
            elif "hidden_states" in kwargs:
                kwargs = {**kwargs, "hidden_states": canonicalize_singleton_batch(kwargs["hidden_states"])}
            context = self.active.get()
            modes, is_replay = context if context is not None else ({}, False)
            if is_replay:
                if region not in modes:
                    raise RuntimeError(f"Checkpoint replay has no compile execution receipt for {fqn}")
                if modes[region]:
                    self.calls["eager_replays"] += 1
                    return eager(*args, **kwargs)
                self.calls["compiled_replays"] += 1
                with torch._dynamo.config.patch(accumulated_recompile_limit=sys.maxsize):
                    return replay(*args, **kwargs)
            modes[region] = False
            try:
                self.calls["compiled_attempts"] += 1
                return compiled(*args, **kwargs)
            except torch._dynamo.exc.FailOnRecompileLimitHit:
                modes[region] = True
                self.calls["eager_fallbacks"] += 1
                if self.calls["eager_fallbacks"] == 1:
                    logger.warning(
                        "H3 compile cache exhausted at %s; new uncached layouts execute eagerly. "
                        "Cached graphs and native attention remain active; checkpoint replay retains "
                        "its original execution mode. recompile_limit=%s accumulated_recompile_limit=%s", fqn,
                        compile_kwargs.get("recompile_limit"), torch._dynamo.config.accumulated_recompile_limit)
                return eager(*args, **kwargs)

        return forward

    def install_dense(self, transformer: Any, compile_kwargs: dict[str, Any], role: str) -> int:
        count = 0
        for name, module in transformer.named_modules():
            compiled = vars(module).get("forward")
            eager = getattr(compiled, "_torchdynamo_orig_callable", None)
            if eager is None:
                continue
            installed = self.wrap(eager, compiled, compile_kwargs, f"{role}.{name}")
            self.originals.append((module, "forward", compiled, installed))
            module.forward = installed
            count += 1
        return count

    def close(self) -> None:
        pending = []
        failures = []
        for module, name, original, installed in reversed(self.originals):
            try:
                if getattr(module, name) is installed:
                    setattr(module, name, original)
            except BaseException as error:
                pending.append((module, name, original, installed))
                failures.append(error)
        self.originals = list(reversed(pending))
        if failures:
            primary = failures[0]
            vars(primary)["_fastvideo_checkpoint_compile_cache"] = self
            vars(primary).setdefault("_fastvideo_compile_cleanup_errors", []).extend(failures[1:])
            raise primary
        logger.info("H3 compile cache recovery counters: %s", dict(self.calls))
