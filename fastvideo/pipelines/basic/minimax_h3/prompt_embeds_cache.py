# SPDX-License-Identifier: Apache-2.0
"""Exact cross-request storage for the MiniMax-H3 Qwen3-VL conditioning output.

The text encoder is the largest component in the MiniMax-H3 stack, and on a
unified-memory box (GB10 / Spark) it cannot stay materialized next to the DiT,
so the pipeline releases it after every conditioning stage.  A benchmark or
server that serves the same prompt presentation again then pays the full
encoder reload (tens of seconds) plus the encode, even though the output is a
pure function of the prompt presentation.

This module stores that output keyed by a digest built in the conditioning
stage.  Entries are cloned on the way in and out, for the same reason as
``memo.ContentMemo``: callers may mutate what they get back.

Off by default. ``FASTVIDEO_H3_PROMPT_EMBEDS_CACHE=memory`` keeps an in-process LRU; a directory
also persists entries across processes.  The cached tensor is the
encoder's own output, bit-identical, so this is an exact-tier optimization.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

FORMAT = 1
_DISABLED = {"0", "off", "false", "no", "none", "disable", "disabled"}
_MEMORY_ONLY = "memory"


def _setting() -> str | None:
    raw = envs.FASTVIDEO_H3_PROMPT_EMBEDS_CACHE.get()
    if not raw:
        return None
    return str(raw).strip()


def _disabled() -> bool:
    """Off unless opted in: unset, empty, or an off-word disables the cache."""
    value = _setting()
    return value is None or value.lower() in _DISABLED


def _cache_root() -> Path | None:
    value = _setting()
    if value is None or value.lower() in _DISABLED or value.lower() == _MEMORY_ONLY:
        return None
    return Path(value).expanduser()


def _entry_path(root: Path, key: str) -> Path:
    return root / f"{key}.pt"


class PromptEmbedsCache:
    """In-process LRU plus an optional on-disk directory keyed by the stage's digest."""

    def __init__(self, capacity: int = 8) -> None:
        self._root = _cache_root()
        self._capacity = 0 if _disabled() else max(0, capacity)
        self._items: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def get(self, key: str | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if key is None or not self.enabled:
            return None
        entry = self._items.pop(key, None)
        if entry is not None:
            self._items[key] = entry
            return self._clone(entry)
        root = self._root
        if root is None:
            return None
        path = _entry_path(root, key)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("Ignoring unreadable MiniMax-H3 prompt-embeds cache entry %s", path)
            return None
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            return None
        entry = (payload["prompt_embeds"], payload["text_token_tags"])
        self._remember(key, entry)
        return self._clone(entry)

    def put(self, key: str | None, prompt_embeds: torch.Tensor, text_token_tags: torch.Tensor) -> None:
        if key is None or not self.enabled:
            return
        entry = (
            prompt_embeds.detach().to("cpu", copy=True).contiguous(),
            text_token_tags.detach().to("cpu", copy=True).contiguous(),
        )
        self._remember(key, entry)
        root = self._root
        if root is None:
            return
        try:
            root.mkdir(parents=True, exist_ok=True)
            target = _entry_path(root, key)
            tmp = target.with_name(f"{target.name}.tmp")
            torch.save({"format": FORMAT, "prompt_embeds": entry[0], "text_token_tags": entry[1]}, tmp)
            os.replace(tmp, target)
        except Exception:
            logger.warning("Could not persist MiniMax-H3 prompt-embeds cache entry %s", key, exc_info=True)

    def _remember(self, key: str, entry: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._items[key] = entry
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    @staticmethod
    def _clone(entry: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return entry[0].clone(), entry[1].clone()


__all__ = ["PromptEmbedsCache"]
