# pyright: reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

import base64
import io
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MAX_SESSION_INIT_IMAGE_BYTES = 15 * 1024 * 1024
SUPPORTED_SESSION_INIT_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$")


@dataclass(frozen=True)
class SessionInitImage:
    file_path: Path
    temp_dir: Path
    display_name: str
    mime_type: str


def _sanitize_display_name(raw_name: object) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return "uploaded-image"
    return Path(text).name or "uploaded-image"


def persist_session_init_image(
    payload: object,
    *,
    temp_root: Path | None = None,
) -> SessionInitImage | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("initial_image must be an object.")

    data_url = str(payload.get("data_url") or "").strip()
    if not data_url:
        return None

    data_url_match = _DATA_URL_RE.match(data_url)
    if data_url_match is None:
        raise ValueError("initial_image.data_url must be a base64 data URL.")

    data_url_mime_type = data_url_match.group("mime").strip().lower()
    mime_type = str(payload.get("mime_type") or data_url_mime_type).strip().lower()
    if mime_type != data_url_mime_type:
        raise ValueError("initial_image.mime_type must match the data URL mime type.")
    if mime_type not in SUPPORTED_SESSION_INIT_IMAGE_MIME_TYPES:
        raise ValueError("initial_image must be a PNG, JPEG, or WebP image.")

    encoded_bytes = re.sub(r"\s+", "", data_url_match.group("data"))
    try:
        raw_bytes = base64.b64decode(encoded_bytes, validate=True)
    except Exception as exc:
        raise ValueError("initial_image.data_url is not valid base64 data.") from exc

    if len(raw_bytes) == 0:
        raise ValueError("initial_image.data_url did not contain image bytes.")
    if len(raw_bytes) > MAX_SESSION_INIT_IMAGE_BYTES:
        raise ValueError("initial_image must be 15 MB or smaller.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            image.load()
            normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    except Exception as exc:
        raise ValueError("initial_image must decode as a valid image.") from exc

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="ltx2_session_init_",
            dir=str(temp_root) if temp_root is not None else None,
        ))
    file_path = temp_dir / "initial_frame.png"
    normalized.save(file_path, format="PNG")

    return SessionInitImage(
        file_path=file_path,
        temp_dir=temp_dir,
        display_name=_sanitize_display_name(payload.get("name")),
        mime_type=mime_type,
    )


def cleanup_session_init_image(session_image: SessionInitImage | None) -> None:
    if session_image is None:
        return
    shutil.rmtree(session_image.temp_dir, ignore_errors=True)
