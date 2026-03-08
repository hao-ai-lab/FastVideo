# SPDX-License-Identifier: Apache-2.0
"""``_target_``-based instantiation utilities.

These helpers resolve a dotted Python path to a class and instantiate it,
filtering constructor kwargs through ``inspect.signature`` so that only
recognized parameters are forwarded.  Unrecognized keys emit a warning
rather than raising — this keeps YAML configs forward-compatible when
a class drops a parameter in a later version.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any


def resolve_target(target: str) -> type:
    """Import and return the class (or callable) at *target*.

    *target* must be a fully-qualified dotted path, e.g.
    ``"fastvideo.train.models.wan.wan.WanModel"``.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"_target_ must be a non-empty dotted path string, "
                         f"got {target!r}")
    target = target.strip()
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"_target_ must contain at least one dot "
                         f"(module.ClassName), got {target!r}")
    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Cannot import module {module_path!r} "
                          f"(from _target_={target!r})") from exc
    try:
        cls = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Module {module_path!r} has no attribute "
                          f"{attr_name!r} (from _target_={target!r})") from exc
    return cls


def instantiate(cfg: dict[str, Any], **extra: Any) -> Any:
    """Instantiate the class specified by ``cfg["_target_"]``.

    All remaining keys in *cfg* (minus ``_target_``) plus any *extra*
    keyword arguments are forwarded to the constructor.  Keys that do
    not match an ``__init__`` parameter are silently warned about and
    dropped, so callers can safely pass a superset.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"instantiate() expects a dict with '_target_', "
                        f"got {type(cfg).__name__}")
    target_str = cfg.get("_target_")
    if target_str is None:
        raise KeyError("Config dict is missing '_target_' key")

    cls = resolve_target(str(target_str))
    kwargs: dict[str, Any] = {k: v for k, v in cfg.items() if k != "_target_"}
    kwargs.update(extra)

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if not has_var_keyword:
        valid_names = {
            name
            for name, p in params.items() if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        valid_names.discard("self")
        unrecognized = set(kwargs) - valid_names
        if unrecognized:
            warnings.warn(
                f"instantiate({target_str}): dropping unrecognized "
                f"kwargs {sorted(unrecognized)}",
                stacklevel=2,
            )
            for key in unrecognized:
                del kwargs[key]

    return cls(**kwargs)
