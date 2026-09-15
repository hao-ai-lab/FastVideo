"""Bounded, runtime-local media library shared by the HTTP and generation APIs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_LIMIT = 15 * 1024 * 1024
MEDIA_LIMIT = 100 * 1024 * 1024
STORE_LIMIT = 2 * 1024 * 1024 * 1024
ASSET_LIMIT = 100
MAX_MEDIA_SECONDS = 30
MIME_TYPES = {
    "image/png": ("image", ".png"),
    "image/jpeg": ("image", ".jpg"),
    "image/webp": ("image", ".webp"),
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "video/webm": ("video", ".webm"),
    "audio/mpeg": ("audio", ".mp3"),
    "audio/mp4": ("audio", ".m4a"),
    "audio/wav": ("audio", ".wav"),
    "audio/x-wav": ("audio", ".wav"),
    "audio/flac": ("audio", ".flac"),
    "audio/ogg": ("audio", ".ogg"),
    "audio/webm": ("audio", ".webm"),
}


@dataclass(frozen=True)
class StoredAsset:
    asset_id: str
    kind: str
    path: str
    name: str
    mime_type: str
    size: int

    def public(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "name": self.name,
            "mime_type": self.mime_type,
            "size": self.size,
            "url": f"/assets/{self.asset_id}",
        }


def validate_media(path: Path, mime_type: str) -> None:
    """Inspect content, not filenames; refuse playlists and non-media uploads."""
    kind = MIME_TYPES[mime_type][0]
    if kind == "image":
        try:
            with Image.open(path) as img:
                expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[mime_type]
                if img.format != expected:
                    raise ValueError("The image content does not match its file type.")
                if img.width * img.height > 16_777_216:
                    raise ValueError("Images must contain at most 16 megapixels.")
                if getattr(img, "is_animated", False):
                    raise ValueError("Use a still image or upload the animation as a video.")
                img.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The image could not be decoded. Use PNG, JPEG, or WebP.") from exc
        return

    probe = shutil.which(os.getenv("FASTVIDEO_FFPROBE_BIN", "ffprobe"))
    if not probe:
        raise ValueError("This runtime needs ffprobe installed to accept video and audio assets.")
    try:
        result = subprocess.run(
            [
                probe, "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist",
                "mov,matroska,webm,mp3,wav,flac,ogg", "-show_format", "-show_streams", "-of", "json",
                str(path)
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        info = json.loads(result.stdout)
        formats = set(info.get("format", {}).get("format_name", "").split(","))
        if not formats.intersection({"mov", "mp4", "matroska", "webm", "mp3", "wav", "flac", "ogg"}):
            raise ValueError("Upload a media file, not a playlist or external reference.")
        streams = [stream for stream in info.get("streams", []) if stream.get("codec_type") == kind]
        if not streams:
            raise ValueError(f"The file contains no {kind} stream.")
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio" and int(stream.get("channels", 0)) not in (1, 2):
                raise ValueError("H3 references require mono or stereo audio, including video soundtracks.")
        duration = float(info.get("format", {}).get("duration", "nan"))
        if not math.isfinite(duration) or not 0 < duration <= MAX_MEDIA_SECONDS:
            raise ValueError(f"Reference video and audio must be between 0 and {MAX_MEDIA_SECONDS} seconds long.")
        for stream in streams:
            if kind == "video" and int(stream.get("width", 0)) * int(stream.get("height", 0)) > 8_294_400:
                raise ValueError("Reference videos must be 4K or smaller.")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        raise ValueError("The media file could not be decoded. Check its format and try again.") from exc


class AssetStore:
    """Assets live until deletion or runtime exit; pinned generation inputs cannot be deleted."""

    def __init__(self) -> None:
        self._directory: tempfile.TemporaryDirectory | None = None
        self._assets: dict[str, StoredAsset] = {}
        self._pins: dict[str, int] = {}
        self._lock = threading.RLock()

    def staging_path(self, mime_type: str) -> Path:
        with self._lock:
            if mime_type not in MIME_TYPES:
                raise ValueError("Unsupported media type. Use PNG/JPEG/WebP, MP4/WebM/MOV, or WAV/MP3/M4A/FLAC/OGG.")
            if len(self._assets) >= ASSET_LIMIT or sum(item.size for item in self._assets.values()) >= STORE_LIMIT:
                raise ValueError("The runtime asset library is full. Remove unused assets before uploading more.")
            if self._directory is None:
                self._directory = tempfile.TemporaryDirectory(prefix="dreamverse-assets-")
            return Path(self._directory.name) / f"{uuid.uuid4().hex}{MIME_TYPES[mime_type][1]}"

    def add(self, path: Path, name: str, mime_type: str) -> StoredAsset:
        validate_media(path, mime_type)
        size = path.stat().st_size
        if size == 0 or size > (IMAGE_LIMIT if MIME_TYPES[mime_type][0] == "image" else MEDIA_LIMIT):
            raise ValueError("The asset is empty or exceeds its upload size limit.")
        with self._lock:
            if len(self._assets) >= ASSET_LIMIT or size + sum(item.size
                                                              for item in self._assets.values()) > STORE_LIMIT:
                raise ValueError("The runtime asset library is full. Remove unused assets before uploading more.")
            if self._directory is None or path.parent != Path(self._directory.name):
                raise ValueError("The asset must be uploaded to this runtime.")
            display_name = re.sub(r"[\x00-\x1f\x7f/\\]", "_", name).strip()[:200] or "Untitled asset"
            asset = StoredAsset(path.stem, MIME_TYPES[mime_type][0], str(path), display_name, mime_type, size)
            self._assets[asset.asset_id] = asset
            return asset

    def get(self, asset_id: str) -> StoredAsset:
        with self._lock:
            if not isinstance(asset_id, str) or not re.fullmatch(r"[a-f0-9]{32}", asset_id):
                raise ValueError("Invalid asset ID. Upload or select an asset from the library.")
            asset = self._assets.get(asset_id)
            if asset is None or not Path(asset.path).is_file():
                raise ValueError("An asset is no longer available. Upload it again and reselect it.")
            return asset

    def pin(self, asset_ids: list[str]) -> None:
        with self._lock:
            for asset_id in asset_ids:
                self.get(asset_id)
            for asset_id in asset_ids:
                self._pins[asset_id] = self._pins.get(asset_id, 0) + 1

    def release(self, asset_ids: list[str]) -> None:
        with self._lock:
            for asset_id in asset_ids:
                count = self._pins.get(asset_id, 0)
                if count > 1:
                    self._pins[asset_id] = count - 1
                else:
                    self._pins.pop(asset_id, None)

    def delete(self, asset_id: str) -> None:
        with self._lock:
            asset = self.get(asset_id)
            if self._pins.get(asset_id, 0):
                raise ValueError("This asset is in use by a generation session. End the session before deleting it.")
            Path(asset.path).unlink(missing_ok=True)
            del self._assets[asset_id]


asset_store = AssetStore()
