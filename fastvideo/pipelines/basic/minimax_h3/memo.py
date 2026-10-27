# SPDX-License-Identifier: Apache-2.0
"""Content-keyed memo for MiniMax-H3 reference encodes.

Serving the same reference images again (a re-roll, a new seed, the next turn
of a session) re-runs the Qwen3-VL presentation and the VAE keyframe encode on
identical inputs. Both are pure functions of their inputs — the keyframe
posterior sample uses a fixed-seed generator — so their outputs can be reused
exactly. Keys hash the pixel bytes, never object identity, so a mutated or
re-decoded image misses. Entries are cloned on the way in and out, so callers
may mutate what they get back.

Every sequence-parallel rank prepares the same references in the same order,
so hits and misses are uniform across ranks and an encode that enters a
collective is either skipped or run on all of them.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

import fastvideo.envs as envs


def image_key(image: Any) -> tuple:
    """Shape, dtype and a 128-bit digest of an image's pixel bytes."""
    pixels = np.ascontiguousarray(np.asarray(image))
    return pixels.shape, str(pixels.dtype), hashlib.blake2b(pixels.tobytes(), digest_size=16).hexdigest()


def _clone(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value.clone()


class ContentMemo:
    """A small LRU of cloned tensor (or tuple-of-tensor) results; capacity 0 disables it."""

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = max(0, envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.get() if capacity is None else capacity)
        self._items: OrderedDict[Any, Any] = OrderedDict()

    @property
    def enabled(self) -> bool:
        # Training needs the graph through the encoders; a memo would cut it.
        return self.capacity > 0 and not torch.is_grad_enabled()

    def __len__(self) -> int:
        return len(self._items)

    def get_or_compute(self, key: Any, compute) -> Any:
        if not self.enabled:
            return compute()
        cached = self._items.get(key)
        if cached is None:
            value = compute()
            self._items[key] = _clone(value)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return value
        self._items.move_to_end(key)
        return _clone(cached)

    def clear(self) -> None:
        self._items.clear()
