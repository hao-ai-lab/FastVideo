# SPDX-License-Identifier: Apache-2.0
"""Keep FA4's JIT kernel cache consistent with the FA4 tuning environment.

``flash_attn.cute`` (FA4) reads its tuning knobs straight from the environment
while it *traces* a kernel -- ``FA4_FP8_PV_USE_FUSED_PACK``,
``FA4_MXFP8_USE_INLINE_PTX``, ``FA4_SFQK_TMEM_SLOT``, ``FA4_E2E_FREQ``,
``FA_DISABLE_2CTA`` and friends -- but the key it indexes
``_flash_attn_fwd.compile_cache`` with only covers shapes, dtypes and call
flags. Changing a knob inside one process therefore silently reuses the kernel
that was compiled for the previous value: the new setting is a no-op, and any
in-process A/B sweep (or a tuning run that reuses a worker) reports numbers for
the *first* arm. Only clearing the cache by hand, or restarting the process,
gets the new kernel.

This module fingerprints the FA4 tuning environment (everything under ``FA4_``
plus the ``FA_`` prefix the same kernels read) and drops the external JIT
caches whenever the fingerprint moves, so each knob value compiles its own
kernel. The authoritative fix is to fold the knobs into FA4's compile key; this
is the in-tree guard for the FA4 build FastVideo pins today.

Only Python-level state is touched, so this is safe to call from a
``torch.library.custom_op`` implementation (where the real FA4 call lives and
where dynamo cannot trace it anyway).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from fastvideo.logger import init_logger

logger = init_logger(__name__)

#: Environment prefixes that select or retune FA4 kernel code paths.
FA4_ENV_PREFIXES = ("FA4_", "FA_")

#: JIT caches hanging off the FA4 kernel entry points (fwd + backward).
_JIT_CACHE_ATTRS = ("compile_cache", "compile_cache_pre", "compile_cache_post")

# Fingerprint observed on the previous call; ``None`` until the first one.
_last_fingerprint: tuple[tuple[str, str], ...] | None = None


def fa4_env_fingerprint(environ: Mapping[str, str] | None = None, ) -> tuple[tuple[str, str], ...]:
    """Return the sorted ``(name, value)`` pairs of FA4 tuning env vars."""
    source = os.environ if environ is None else environ
    # Iterate keys and look values up by name: reading every value of the
    # process environment costs ~2x more, and this runs per attention call.
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
    """Clear FA4's JIT caches when the FA4 tuning env moved since the last call.

    The first call of a process also clears: the caches may already hold a
    kernel compiled by other code (or by an earlier import) under a different
    environment, and clearing an empty cache is free.

    Returns ``True`` when the caches were dropped.
    """
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
