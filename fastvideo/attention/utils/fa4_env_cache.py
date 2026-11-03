# SPDX-License-Identifier: Apache-2.0
"""Drop FA4's JIT kernel caches when its FA4_*/FA_* tuning environment changes.

FA4 reads these knobs while tracing a kernel but does not include them in its compile-cache key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

FA4_ENV_PREFIXES = ("FA4_", "FA_")

_JIT_CACHE_ATTRS = ("compile_cache", "compile_cache_pre", "compile_cache_post")

_last_fingerprint: tuple[tuple[str, str], ...] | None = None


def fa4_env_fingerprint(environ: Mapping[str, str] | None = None, ) -> tuple[tuple[str, str], ...]:
    """Return the sorted ``(name, value)`` pairs of FA4 tuning env vars."""
    source = envs.external_with_prefixes("FA4_", "FA_") if environ is None else environ
    pairs = [(name, source[name]) for name in source if name.startswith(FA4_ENV_PREFIXES)]
    pairs.sort()
    return tuple(pairs)


def _cached_entries(cache: object) -> dict | None:
    """The in-memory entries of an FA4 JIT cache, or None if ``cache`` is not one.

    Older FA4 builds keep a plain dict; the pinned ref wraps it in ``cache_utils.JITCache``
    (entries under ``.cache``), which is not a dict, so checking only for dicts cleared nothing.
    """
    if isinstance(cache, dict):
        return cache
    entries = getattr(cache, "cache", None)
    if isinstance(entries, dict) and callable(getattr(cache, "clear", None)):
        return entries
    return None


def clear_fa4_jit_caches(kernel_objects: Iterable[object]) -> int:
    """Drop every FA4 JIT cache on ``kernel_objects``; return entries dropped.

    Clearing goes through the cache's own ``clear()``. For FA4's opt-in persistent cache
    (FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1) that also purges its on-disk kernels, whose keys
    omit the tuning env as well; it only happens when compiled kernels are held in memory.
    """
    cleared = 0
    for kernel_object in kernel_objects:
        for attribute in _JIT_CACHE_ATTRS:
            cache = getattr(kernel_object, attribute, None)
            entries = _cached_entries(cache)
            if entries:
                cleared += len(entries)
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
