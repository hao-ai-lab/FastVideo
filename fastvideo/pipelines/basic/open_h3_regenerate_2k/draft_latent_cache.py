# SPDX-License-Identifier: Apache-2.0
"""Exact cross-request storage for the Open-H3-Regenerate-2K draft-latent encode.

The official Regenerate-2K benchmark feeds a 768p draft video, and the latent
preparation stage encodes it with the fp32 H3 video VAE before upscaling.  On
the Spark the compiled encode is still ~82 s of the warm request (~26% of the
end-to-end wall), and it is a pure function of the draft pixels, the encoder
tiling, the fixed posterior seed and the model bundle.  A benchmark or server
that serves the same draft presentation again therefore pays it repeatedly for
no reason.

This module stores the *normalized native latent* returned by
``_encode_draft`` keyed by a digest of the input file and encode settings.
Entries are cloned on the way in and out (callers may mutate what they get
back).  Off by default. ``FASTVIDEO_H3_DRAFT_LATENT_CACHE=memory`` keeps an in-process LRU; a
directory also persists entries across processes.  The cached tensor
is the encoder's own output, so this is an exact-tier optimization.  The
product path (a native latent supplied by the draft server) never reaches this
cache.
"""

from __future__ import annotations

import hashlib
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
    raw = envs.FASTVIDEO_H3_DRAFT_LATENT_CACHE.get()
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


# Content digests are memoized per (absolute path, size, mtime_ns) so the same
# draft file is hashed at most once per process.
_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: str) -> str:
    absolute = os.path.realpath(path)
    stat = os.stat(absolute)
    memo_key = (absolute, int(stat.st_size), int(stat.st_mtime_ns))
    digest = _FILE_DIGESTS.get(memo_key)
    if digest is not None:
        return digest
    hasher = hashlib.blake2b(digest_size=16)
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _FILE_DIGESTS[memo_key] = digest
    return digest


def draft_latent_cache_key(video_path: str | None, draft_size: object, tile_spec: object, model_path: object,
                           seed: int) -> str | None:
    """Stable digest of everything ``_encode_draft`` consumes."""
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        digest = _file_digest(video_path)
    except OSError:
        return None
    parts = [
        "h3-draft-latent",
        str(FORMAT),
        str(model_path),
        str(draft_size),
        str(tile_spec),
        str(int(seed)),
        digest,
    ]
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()


class DraftLatentCache:
    """In-process LRU plus an optional on-disk directory keyed by the stage's digest."""

    def __init__(self, capacity: int = 4) -> None:
        self._root = _cache_root()
        self._capacity = 0 if _disabled() else max(0, capacity)
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def get(self, key: str | None) -> torch.Tensor | None:
        if key is None or not self.enabled:
            return None
        entry = self._items.pop(key, None)
        if entry is not None:
            self._items[key] = entry
            return entry.clone()
        root = self._root
        if root is None:
            return None
        path = _entry_path(root, key)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("Ignoring unreadable MiniMax-H3 draft-latent cache entry %s", path)
            return None
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            return None
        entry = payload["draft"]
        self._remember(key, entry)
        return entry.clone()

    def put(self, key: str | None, draft: torch.Tensor) -> None:
        if key is None or not self.enabled:
            return
        entry = draft.detach().to("cpu", copy=True).contiguous()
        self._remember(key, entry)
        root = self._root
        if root is None:
            return
        try:
            root.mkdir(parents=True, exist_ok=True)
            target = _entry_path(root, key)
            tmp = target.with_name(f"{target.name}.tmp")
            torch.save({"format": FORMAT, "draft": entry}, tmp)
            os.replace(tmp, target)
        except Exception:
            logger.warning("Could not persist MiniMax-H3 draft-latent cache entry %s", key, exc_info=True)

    def _remember(self, key: str, entry: torch.Tensor) -> None:
        self._items[key] = entry
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)


__all__ = ["DraftLatentCache", "draft_latent_cache_key"]
