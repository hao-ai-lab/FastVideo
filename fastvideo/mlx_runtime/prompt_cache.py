# SPDX-License-Identifier: Apache-2.0
"""Best-effort prompt-embedding cache shared by the MLX entrypoints."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def fingerprint_digest(fingerprint: dict[str, object]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_encoder_fingerprint(model_root: Path) -> dict[str, object]:
    """Return a cheap identity for the tokenizer and text-encoder files."""
    root = model_root.resolve()
    components = [path for name in ("tokenizer", "text_encoder") if (path := root / name).is_dir()]
    scan_roots = components or [root]
    files: list[list[object]] = []
    complete = True
    try:
        for scan_root in scan_roots:
            for path in sorted(scan_root.rglob("*")):
                try:
                    if not path.is_file():
                        continue
                    stat = path.stat()
                    files.append([
                        path.relative_to(root).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    ])
                except OSError:
                    complete = False
    except OSError:
        complete = False
    # ponytail: metadata avoids hashing multi-GB weights; use a model manifest
    # if supported workflows ever preserve size, mtime, and ctime while mutating.
    return {"root": str(root), "files": files, "complete": complete}


def prompt_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _fingerprint_is_complete(fingerprint: dict[str, object]) -> bool:
    text_encoder = fingerprint.get("text_encoder")
    return not isinstance(text_encoder, dict) or text_encoder.get("complete") is not False


def load_prompt_cache(
    cache_path: Path | None,
    fingerprint: dict[str, object],
) -> np.ndarray | None:
    """Load a matching cache entry, treating every cache failure as a miss."""
    if cache_path is None or not _fingerprint_is_complete(fingerprint):
        return None
    try:
        metadata = json.loads(prompt_cache_meta_path(cache_path).read_text())
        if not isinstance(metadata, dict):
            return None
        if metadata.get("fingerprint_sha256") != fingerprint_digest(fingerprint):
            return None
        payload = cache_path.read_bytes()
        if metadata.get("data_sha256") != hashlib.sha256(payload).hexdigest():
            return None
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        return array if isinstance(array, np.ndarray) else None
    except (EOFError, OSError, UnicodeError, ValueError) as exc:
        logger.info("Prompt cache read skipped for %s: %s", cache_path, exc)
        return None


def _atomic_write(path: Path, payload: bytes) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_prompt_cache(
    cache_path: Path | None,
    embeds: np.ndarray,
    fingerprint: dict[str, object],
) -> bool:
    """Atomically publish an integrity-bound cache entry when possible."""
    if cache_path is None or not _fingerprint_is_complete(fingerprint):
        return False
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(embeds), allow_pickle=False)
        payload = buffer.getvalue()
        metadata = (json.dumps(
            {
                "fingerprint_sha256": fingerprint_digest(fingerprint),
                "data_sha256": hashlib.sha256(payload).hexdigest(),
                "fingerprint": fingerprint,
            },
            indent=2) + "\n").encode("utf-8")
        # Publish data first. Until metadata follows, old metadata's data digest
        # makes the torn pair a harmless miss rather than a stale cache hit.
        _atomic_write(cache_path, payload)
        _atomic_write(prompt_cache_meta_path(cache_path), metadata)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.info("Prompt cache write skipped for %s: %s", cache_path, exc)
        return False
