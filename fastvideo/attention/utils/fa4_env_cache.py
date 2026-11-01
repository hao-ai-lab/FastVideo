# SPDX-License-Identifier: Apache-2.0
"""Drop FA4's JIT kernel caches when its FA4_*/FA_* tuning environment changes.

FA4 reads these knobs while tracing a kernel but does not include them in its compile-cache key.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from fastvideo.logger import init_logger

logger = init_logger(__name__)

FA4_ENV_PREFIXES = ("FA4_", "FA_")

_JIT_CACHE_ATTRS = ("compile_cache", "compile_cache_pre", "compile_cache_post")

_last_fingerprint: tuple[tuple[str, str], ...] | None = None


def fa4_env_fingerprint(environ: Mapping[str, str] | None = None, ) -> tuple[tuple[str, str], ...]:
    """Return the sorted ``(name, value)`` pairs of FA4 tuning env vars."""
    source = os.environ if environ is None else environ
    pairs = [(name, source[name]) for name in source if name.startswith(FA4_ENV_PREFIXES)]
    pairs.sort()
    return tuple(pairs)


def clear_fa4_jit_caches(kernel_objects: Iterable[object]) -> int:
    """Drop every FA4 JIT cache on ``kernel_objects``; return entries dropped."""
    cleared = 0
    for kernel_object in kernel_objects:
        for attribute in _JIT_CACHE_ATTRS:
            cache = getattr(kernel_object, attribute, None)
            if isinstance(cache, dict) and cache:
                cleared += len(cache)
                cache.clear()
    return cleared


def reset_fa4_env_fingerprint() -> None:
    """Forget the last observed fingerprint (tests / fresh-process semantics)."""
    global _last_fingerprint
    _last_fingerprint = None


def invalidate_fa4_jit_caches_on_env_change(
    kernel_objects: Iterable[object],
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Clear FA4's JIT caches if the FA4 tuning env changed since the last call (always on the first call)."""
    global _last_fingerprint
    fingerprint = fa4_env_fingerprint(environ)
    if fingerprint == _last_fingerprint:
        return False
    previous, _last_fingerprint = _last_fingerprint, fingerprint
    cleared = clear_fa4_jit_caches(kernel_objects)
    if previous is not None:
        changed = sorted(name for name in {k
                                           for k, _ in previous} | {k
                                                                    for k, _ in fingerprint}
                         if dict(previous).get(name) != dict(fingerprint).get(name))
        if cleared:
            logger.info(
                "FA4 tuning env changed (%s); dropped %d cached JIT kernel(s) "
                "so the new setting is not served stale code", ", ".join(changed), cleared)
        else:
            logger.debug("FA4 tuning env changed (%s); FA4 JIT cache was empty", ", ".join(changed))
    return True
