# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import cast

import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.abstract import AttentionBackend
from fastvideo.logger import init_logger
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import STR_BACKEND_ENV_VAR, resolve_obj_by_qualname

logger = init_logger(__name__)


def backend_name_to_enum(backend_name: str) -> AttentionBackendEnum | None:
    """
    Convert a string backend name to a _Backend enum value.

    Returns:
    * _Backend: enum value if backend_name is a valid in-tree type
    * None: otherwise it's an invalid in-tree type or an out-of-tree platform is
            loaded.
    """
    assert backend_name is not None
    return AttentionBackendEnum[backend_name] if backend_name in AttentionBackendEnum.__members__ else \
          None


def coerce_attn_backend(attn_backend: AttentionBackendEnum | str | None, ) -> AttentionBackendEnum | None:
    """Normalize an explicit backend selection.

    Environment-variable parsing remains permissive via
    :func:`backend_name_to_enum`, but typed/config-driven call sites should
    fail fast on typos instead of silently falling back to another backend.
    """
    if attn_backend is None or isinstance(attn_backend, AttentionBackendEnum):
        return attn_backend
    if not isinstance(attn_backend, str) or not attn_backend.strip():
        raise ValueError("attention backend must be a non-empty string, "
                         f"an AttentionBackendEnum, or None; got {attn_backend!r}")

    backend_name = attn_backend.strip().upper()
    backend = backend_name_to_enum(backend_name)
    if backend is None:
        raise ValueError(f"Unknown attention backend {attn_backend!r}. "
                         f"Expected one of {sorted(AttentionBackendEnum.__members__)}")
    return backend


def get_env_variable_attn_backend() -> AttentionBackendEnum | None:
    '''
    Get the backend override specified by the FastVideo attention
    backend environment variable, if one is specified.

    Returns:

    * _Backend enum value if an override is specified
    * None otherwise
    '''
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return (None if backend_name is None else backend_name_to_enum(backend_name))


class _NoRequest:
    """Type of the "caller expressed no opinion" sentinel.

    ``None`` cannot carry that meaning: it is a real answer — "this component
    resolved to automatic selection" — and a caller passing it wants automatic
    selection, not a fallback to whatever the environment says.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<no request>"


NO_REQUEST = _NoRequest()


@dataclass(frozen=True)
class _BackendScope:
    backend: AttentionBackendEnum | None
    component: str | None
    consult_env: bool


_SCOPE: ContextVar[_BackendScope | None] = ContextVar("fastvideo_attention_backend_scope", default=None)


@contextmanager
def _component_attention_backend_scope(
    attn_backend: AttentionBackendEnum | str | None,
    component: str | None = None,
    *,
    consult_env: bool = False,
) -> Generator[None, None, None]:
    """Carry a component's resolved request from its loader down to its layers.

    **Transitional plumbing, not a feature.** The decision is resolved once at
    load time and written onto the component's own config
    (``ModelConfig._resolved_attention_backend``); this scope exists only
    because the layer constructors nested inside a model do not yet receive
    that config. They are handed a bare ``supported_attention_backends`` tuple,
    threaded through block constructors up to four levels deep.

    Deletion path: thread the request alongside that tuple, one model family at
    a time, starting with the families that carry per-role requests (Wan,
    LTX-2, Kandinsky5). When the last family reads its request from its own
    config, this scope, ``_SCOPE`` and ``_BackendScope`` all go away and
    ``get_attn_backend`` takes the request as a plain argument.

    Nothing switches backends at runtime and nothing should: every caller wraps
    *construction*. Process-local, exception-safe, nestable, and part of the
    resolution cache key, so it never needs a cache flush.

    ``_component_attention_backend_scope(None)`` means "automatic selection,
    ignore any process-wide request" (dense teacher/critic roles built
    alongside a quantized student); ``consult_env=True`` re-enables the env
    fallback inside a scope.
    """
    token = _SCOPE.set(_BackendScope(coerce_attn_backend(attn_backend), component, consult_env))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _active_component_attention_backend_scope() -> "_BackendScope | None":
    """The request governing the component being constructed right now, or None."""
    return _SCOPE.get()


def record_resolved_attention_backend(config: object) -> AttentionBackendEnum | None:
    """Write the request governing construction *right now* onto a component config.

    Called by each loader while its component is being built, so the decision
    ends up on the object the component keeps and is readable from the component
    after load. The loader is the right place rather than the caller above it:
    a loader may narrow the request for one component (the DMD teacher/critic
    transformers build dense inside a nested scope), and only it knows that.
    """
    scope = _SCOPE.get()
    resolved = scope.backend if scope is not None else None
    config._resolved_attention_backend = resolved  # type: ignore[attr-defined]
    return resolved


def component_attention_backend(component: object) -> AttentionBackendEnum | _NoRequest:
    """Read back the decision :func:`record_resolved_attention_backend` wrote.

    Returns ``NO_REQUEST`` unless the component recorded a *concrete* backend,
    so a caller that passes this through only overrides the ambient fallback
    when there is a real decision to override it with.

    ``NO_REQUEST`` rather than ``None`` for the no-decision case is deliberate,
    and cannot be derived from the attribute's presence: ``ModelConfig`` declares
    ``_resolved_attention_backend`` as a field defaulting to ``None``, so the
    attribute always exists and a ``getattr`` default can never fire. Worse,
    ``record_resolved_attention_backend`` writes ``None`` whenever no scope is
    active, so "resolved to automatic selection" and "never recorded" are the
    same stored value. Neither state should suppress the environment variable at
    a call site that previously honoured it, and collapsing both to
    ``NO_REQUEST`` keeps that behavior identical.
    """
    resolved = getattr(getattr(component, "config", None), "_resolved_attention_backend", None)
    return NO_REQUEST if resolved is None else resolved


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum, ...]
    | None = None,
    default_backend: AttentionBackendEnum | None = None,
    *,
    requested: AttentionBackendEnum | None | _NoRequest = NO_REQUEST,
) -> type[AttentionBackend]:
    """Resolve the attention backend class for one call site.

    ``requested`` is the decision made for the component this call site belongs
    to — read from ``ModelConfig._resolved_attention_backend``. Passing it means
    the caller knows the answer, so nothing ambient is consulted:

    * ``requested=SOME_BACKEND`` — use it (subject to the layer's declared
      support), ignoring the construction scope and the environment;
    * ``requested=None`` — that component resolved to *automatic selection*.
      That is an answer, not an absence, so the environment is not consulted
      behind it;
    * ``requested`` omitted (``NO_REQUEST``) — the caller has no opinion; fall
      back to the construction scope, then the environment. This is the path
      every layer built inside a loader still takes.
    """
    # Resolve every selection-affecting input BEFORE the cache so all of them
    # live in the cache key: no mutation (env, scope) ever needs a cache_clear
    # again, and components with different requests get distinct cache entries
    # (per-component resolution).
    if not isinstance(requested, _NoRequest):
        # Explicit: the component's own decision outranks anything ambient.
        component = None
        env_backend = None
    else:
        scope = _SCOPE.get()
        if scope is not None:
            requested = scope.backend
            component = scope.component
            env_backend = envs.FASTVIDEO_ATTENTION_BACKEND if scope.consult_env else None
        else:
            requested = None
            component = None
            env_backend = envs.FASTVIDEO_ATTENTION_BACKEND
    # The active device is a real selection input, not bookkeeping: the
    # platform's backend resolution runs capability probes against the
    # *current* device (e.g. AttnQatInferBackend's per-arch capability sets
    # decide sm_12x CUTLASS vs sm_100/sm_103 FP4 FA4 vs FlashAttention
    # fallback). Keying on it means a resolution taken for one device is
    # never handed to another.
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
    return _cached_get_attn_backend(
        head_size,
        dtype,
        supported_attention_backends,
        default_backend,
        requested=requested,
        env_backend=env_backend,
        component=component,
        device_index=device_index,
    )


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum, ...]
    | None = None,
    default_backend: AttentionBackendEnum | None = None,
    *,
    requested: AttentionBackendEnum | None = None,
    env_backend: str | None = None,
    component: str | None = None,
    device_index: int | None = None,
) -> type[AttentionBackend]:
    # Pure resolution: every selection input arrives as an argument (and is
    # therefore part of the functools cache key). Only get_attn_backend calls
    # this; it performs the scope/env/force reads.
    if not supported_attention_backends:
        raise ValueError("supported_attention_backends is empty")
    # Cache-key-only inputs: component identity separates two components that
    # agree on shape/dtype but not on request; device_index separates
    # resolutions taken under different device capabilities (the platform
    # probes the current device below).
    del component, device_index
    # Precedence: the per-component request (explicit argument, else the
    # construction scope), then the env var (already resolved to None by the
    # wrapper whenever the caller supplied a request), then the layer-declared
    # default. Every one of these arrived as an argument, so all of them are in
    # the cache key.
    selected_backend = requested
    if selected_backend is None and env_backend is not None:
        selected_backend = backend_name_to_enum(env_backend)

    # Layer-level default (e.g. a checkpoint that requires a specific sparse
    # backend). Lower precedence than the force/scope/env requests, so
    # users can still override it.
    if selected_backend is None and default_backend is not None:
        selected_backend = default_backend

    # get device-specific attn_backend
    from fastvideo.platforms import current_platform

    if (selected_backend is not None and selected_backend not in supported_attention_backends):
        fallback_backend = (default_backend if default_backend in supported_attention_backends else None)
        logger.warning(
            "Requested attention backend %s is not supported by this "
            "layer; supported backends are %s. Falling back to %s.",
            selected_backend.name,
            [b.name for b in supported_attention_backends],
            fallback_backend.name if fallback_backend is not None else "automatic selection",
        )
        selected_backend = fallback_backend
    attention_cls = current_platform.get_attn_backend_cls(selected_backend, head_size, dtype)
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {current_platform.device_name}")
    return cast(type[AttentionBackend], resolve_obj_by_qualname(attention_cls))
