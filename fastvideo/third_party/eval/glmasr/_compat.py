"""Compatibility shims so the vendored GLM-ASR code runs on
fastvideo's pinned ``transformers==4.57.3``.

The vendored code was extracted from ``transformers==5.8.0``. Five
symbols it imports do not exist in 4.57.3:

- ``PreTrainedConfig`` (renamed in 5.x; was ``PretrainedConfig``).
- ``utils.generic.maybe_autocast``     — autocast context helper.
- ``utils.generic.merge_with_config_defaults`` — config-default decorator.
- ``integrations.use_kernelized_func``  — kernel-dispatch decorator.
- ``utils.output_capturing.capture_outputs`` — hook-based outputs decorator.

The first three we vendor so semantics match 5.x. The last two are
optimization / introspection helpers — stubbing them as identity
decorators keeps the standard forward path intact, and our ASR caller
(``model.generate``) never opts into the captured outputs anyway.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import MISSING, fields
from functools import wraps
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

# 5.x renamed the base config class. Alias keeps the vendored config
# files unchanged.
PreTrainedConfig = PretrainedConfig


def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
) -> Any:
    """Vendored from ``transformers.utils.generic`` (5.x).

    Only autocasts if autocast is already enabled for this device, or
    if the caller passed ``enabled=True``. Avoids spurious autocast
    context insertion under ``torch.compile``.
    """
    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    return nullcontext()


def merge_with_config_defaults(func: Any) -> Any:
    """Vendored from ``transformers.utils.generic`` (5.x).

    Decorator: for a small allow-list of args (vision_feature_layer,
    use_cache, etc.), fall back to ``self.config.<arg>`` when the
    caller didn't pass them. GLM-ASR uses this for ``use_cache``.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        args_with_config_defaults = (
            "use_cache",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "vision_aspect_ratio",
        )
        for arg_name in args_with_config_defaults:
            arg_index = None
            if arg_name in func.__code__.co_varnames:
                arg_index = func.__code__.co_varnames.index(arg_name) - 1
            if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:
                continue
            if kwargs.get(arg_name) is not None:
                continue
            cfg_value = getattr(getattr(self, "config", None), arg_name, None)
            if cfg_value is not None:
                kwargs[arg_name] = cfg_value
        return func(self, *args, **kwargs)

    return wrapper


def use_kernelized_func(*_a: Any, **_kw: Any) -> Any:
    """Stub: identity decorator factory.

    5.x uses this to dispatch to optional optimized kernels (Triton
    flash variants, etc.). Without it, the model still runs via the
    standard PyTorch path — we just lose the kernel speed-up. Safe to
    drop for correctness.
    """

    def decorator(fn: Any) -> Any:
        return fn

    return decorator


def capture_outputs(func: Any = None, **_kw: Any) -> Any:
    """Stub: identity decorator.

    5.x uses this to hook into intermediate-layer outputs when the
    caller passes ``output_attentions=True`` / ``output_hidden_states=True``
    on a forward call. Our ASR caller (``model.generate(...)``) never
    sets either, and the stubbed-out decorator preserves the normal
    forward signature.
    """
    if func is None:
        return lambda f: f
    return func


def _patch_attention_interface() -> None:
    """Add 5.x's ``.get_interface(...)`` to 4.57's ``ALL_ATTENTION_FUNCTIONS``.

    4.57's ``AttentionInterface`` is dict-like (``.get``, ``.keys``,
    subscript) but lacks the 5.x-only ``.get_interface(impl, default)``
    method that 5.x's modeling code expects. The shim falls back to the
    default callable when ``impl`` is ``None``, ``"eager"``, or absent
    from the registry — matching the 5.x semantics.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cls = type(ALL_ATTENTION_FUNCTIONS)
    if hasattr(cls, "get_interface"):
        return

    def get_interface(self: Any, attn_implementation: str | None, default: Any) -> Any:
        if attn_implementation is None or attn_implementation == "eager":
            return default
        return self.get(attn_implementation, default)

    cls.get_interface = get_interface


_patch_attention_interface()


def auto_docstring(*args: Any, **_kw: Any) -> Any:
    """Stub: identity decorator (no docstring synthesis).

    transformers 4.57's ``auto_docstring`` chokes on PEP 604 unions
    (e.g. ``int | None``) used in the vendored 5.x signatures —
    ``types.UnionType`` has no ``__name__``. The decorator is purely
    cosmetic (auto-builds a help docstring from the signature), so
    stubbing it preserves behaviour and avoids the parser crash.
    """
    if len(args) == 1 and callable(args[0]) and not _kw:
        return args[0]

    def decorator(fn: Any) -> Any:
        return fn

    return decorator


# Dataclass helpers re-exported for the vendored config to depend on
# stable names regardless of where they live in transformers' API.
__all__ = [
    "PreTrainedConfig",
    "maybe_autocast",
    "merge_with_config_defaults",
    "use_kernelized_func",
    "capture_outputs",
    "MISSING",
    "fields",
]
