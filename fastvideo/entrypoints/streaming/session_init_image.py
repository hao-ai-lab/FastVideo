# SPDX-License-Identifier: Apache-2.0
"""Persist the initial-image blob attached to a streaming session."""
from __future__ import annotations

import base64
import binascii
import contextlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

_ACCEPTED_MIMES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

_MAX_IMAGE_BYTES = 32 * 1024 * 1024  # 32 MiB cap
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$")


@dataclass(frozen=True)
class SessionInitImage:
    """Location of the persisted init image.

    Callers pass ``path`` to ``InputConfig.image_path``; ``display_name``
    is only used for logs.
    """

    path: str
    display_name: str
    mime: str

    @property
    def file_path(self) -> Path:
        """Compatibility alias used by the migrated Dreamverse controller."""

        return Path(self.path)

    @property
    def temp_dir(self) -> Path:
        """Compatibility alias for cleanup of the temporary image directory."""

        return Path(self.path).parent

    @property
    def mime_type(self) -> str:
        """Compatibility alias for the UI payload's MIME field."""

        return self.mime


def persist_session_init_image(
    payload: Any,
    *,
    output_dir: str | None = None,
) -> SessionInitImage | None:
    """Decode a client init-image blob and persist it to disk.

    ``payload`` shape (matches the internal UI protocol)::

        {
            "mime": "image/png",
            "name": "ref.png",
            "data": "<base64 bytes>",
        }

    Returns ``None`` when ``payload`` is falsy (no init image). Raises
    :class:`ValueError` on schema / size / decode errors so the caller
    can surface a user-facing ``error`` frame.
    """
    if not payload:
        return None
    if not isinstance(payload, dict):
        raise ValueError("session init image must be an object")

    data_url = payload.get("data_url")
    if isinstance(data_url, str) and data_url.strip():
        match = _DATA_URL_RE.match(data_url.strip())
        if match is None:
            raise ValueError("session init image data URL is not valid base64")
        mime = payload.get("mime_type") or match.group("mime").strip().lower()
        data_b64 = match.group("data")
    else:
        mime = payload.get("mime")
        data_b64 = payload.get("data")
    if mime not in _ACCEPTED_MIMES:
        raise ValueError(f"session init image mime {mime!r} is not one of "
                         f"{sorted(_ACCEPTED_MIMES)}")
    if not isinstance(data_b64, str):
        raise ValueError("session init image data must be a base64 string")
    try:
        data = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"session init image data is not valid base64: {exc}") from exc
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError(f"session init image is {len(data)} bytes; limit is "
                         f"{_MAX_IMAGE_BYTES}")
    if len(data) == 0:
        raise ValueError("session init image data is empty")

    ext = _ACCEPTED_MIMES[mime]
    display_name = _sanitize_display_name(payload.get("name")) or f"init{ext}"
    fd, path = tempfile.mkstemp(prefix="fastvideo-init-", suffix=ext, dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        raise
    return SessionInitImage(path=path, display_name=display_name, mime=mime)


def cleanup_session_init_image(session_image: SessionInitImage | None) -> None:
    """Remove a persisted session init image and its temporary directory."""

    if session_image is None:
        return
    shutil.rmtree(session_image.temp_dir, ignore_errors=True)


def _sanitize_display_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    # Strip any path components — we only keep the leaf for logging.
    return os.path.basename(name)


__all__ = [
    "SessionInitImage",
    "cleanup_session_init_image",
    "persist_session_init_image",
]
