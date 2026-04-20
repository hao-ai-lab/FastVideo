# SPDX-License-Identifier: Apache-2.0
"""Persist the initial-image blob attached to a streaming session.

The client may include an ``initial_image`` field on
``session_init_v2`` to seed an i2v session with a starting frame. The
payload is base64-encoded PNG / JPEG bytes. We decode, validate, and
persist to a temp file so the downstream pipeline can read it as an
ordinary ``image_path``.
"""
from __future__ import annotations

import base64
import binascii
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

_ACCEPTED_MIMES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

_MAX_IMAGE_BYTES = 32 * 1024 * 1024  # 32 MiB cap


@dataclass(frozen=True)
class SessionInitImage:
    """Location of the persisted init image.

    Callers pass ``path`` to
    ``InputConfig.image_path``; ``display_name`` is only used for logs.
    """

    path: str
    display_name: str
    mime: str


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

    mime = payload.get("mime")
    if mime not in _ACCEPTED_MIMES:
        raise ValueError(f"session init image mime {mime!r} is not one of "
                         f"{sorted(_ACCEPTED_MIMES)}")
    data_b64 = payload.get("data")
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
    target_dir = output_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"fastvideo-init-{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return SessionInitImage(path=path, display_name=display_name, mime=mime)


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
    "persist_session_init_image",
]
