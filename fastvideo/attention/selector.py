# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py

import os
from collections.abc import Generator
from contextlib import contextmanager
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


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# FASTVIDEO ATTENTION BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: AttentionBackendEnum | None = None


def global_force_attn_backend(attn_backend: AttentionBackendEnum | None) -> None:
    '''
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    '''
    global forced_attn_backend
    forced_attn_backend = attn_backend
    _cached_get_attn_backend.cache_clear()


def get_global_forced_attn_backend() -> AttentionBackendEnum | None:
    '''
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    '''
    return forced_attn_backend


def get_selected_attn_backend() -> AttentionBackendEnum | None:
    '''
    The attention backend selected before any per-layer filtering.

    A global force (see `global_force_attn_backend`) takes precedence over the
    `FASTVIDEO_ATTENTION_BACKEND` env var; returns None when neither is set
    (automatic selection). This is the single source of the force-over-env
    precedence rule, shared by `_cached_get_attn_backend` and
    `check_attn_backend_requirement`.
    '''
    forced_backend = get_global_forced_attn_backend()
    if forced_backend is not None:
        return forced_backend
    backend_by_env_var = envs.FASTVIDEO_ATTENTION_BACKEND
    return backend_name_to_enum(backend_by_env_var) if backend_by_env_var is not None else None


def check_attn_backend_requirement(
    required_backend: AttentionBackendEnum | None,
    *,
    model_name: str = "This model",
) -> AttentionBackendEnum | None:
    '''
    Resolve the selected attention backend, enforcing a model's hard requirement.

    Some model families are only numerically correct with a specific backend
    (e.g. FastWan is sparse-distilled with VSA and produces wrong outputs
    otherwise). Rather than silently forcing the backend -- which would only
    reach some construction sites and leave others (e.g. the denoising stage)
    resolving a different backend -- we require the user to select it explicitly
    and fail loudly otherwise, mirroring how the platform layer hard-fails on
    missing explicitly-requested backends.

    Arguments:

    * required_backend: backend the model requires, or None for no requirement
    * model_name: name used in the error message

    Returns:

    * the selected attention backend (`get_selected_attn_backend`), or None if
      unset and unrequired

    Raises:

    * ValueError if `required_backend` is set but is not the selected backend
    '''
    selected_backend = get_selected_attn_backend()
    if required_backend is not None and selected_backend != required_backend:
        selected_name = selected_backend.name if selected_backend is not None else "unset"
        raise ValueError(f"{model_name} requires the {required_backend.name} attention backend, but the effective "
                         f"FASTVIDEO_ATTENTION_BACKEND is {selected_name}. This checkpoint is only correct with "
                         f"{required_backend.name}; set FASTVIDEO_ATTENTION_BACKEND={required_backend.name} before "
                         f"loading the model.")
    return selected_backend


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum, ...]
    | None = None,
) -> type[AttentionBackend]:
    return _cached_get_attn_backend(head_size, dtype, supported_attention_backends)


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum, ...]
    | None = None,
) -> type[AttentionBackend]:
    if not supported_attention_backends:
        raise ValueError("supported_attention_backends is empty")
    # A global force overrides the FASTVIDEO_ATTENTION_BACKEND env var.
    selected_backend = get_selected_attn_backend()

    # get device-specific attn_backend
    from fastvideo.platforms import current_platform

    if (selected_backend is not None and selected_backend not in supported_attention_backends):
        logger.warning(
            "Requested attention backend %s is not supported by this "
            "layer; supported backends are %s. Falling back to automatic "
            "selection.",
            selected_backend.name,
            [b.name for b in supported_attention_backends],
        )
        selected_backend = None
    attention_cls = current_platform.get_attn_backend_cls(selected_backend, head_size, dtype)
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {current_platform.device_name}")
    return cast(type[AttentionBackend], resolve_obj_by_qualname(attention_cls))


@contextmanager
def global_force_attn_backend_context_manager(attn_backend: AttentionBackendEnum) -> Generator[None, None, None]:
    '''
    Globally force a FastVideo attention backend override within a
    context manager, reverting the global attention backend
    override to its prior state upon exiting the context
    manager.

    Arguments:

    * attn_backend: attention backend to force

    Returns:

    * Generator
    '''

    # Save the current state of the global backend override (if any)
    original_value = get_global_forced_attn_backend()

    # Globally force the new backend override
    global_force_attn_backend(attn_backend)

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the original global backend override, if any
        global_force_attn_backend(original_value)
