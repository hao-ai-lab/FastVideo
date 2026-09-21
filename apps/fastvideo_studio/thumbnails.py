# SPDX-License-Identifier: Apache-2.0
"""Small first-frame posters; no video decoding is needed in job lists."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

_LOCK = threading.Lock()
MAX_CACHED_POSTERS = 50


def get_thumbnail(source: Path, cache_dir: Path) -> bytes:
    """Cache a JPEG bounded to 320px, invalidating when its source changes."""
    stat = source.stat()
    key = hashlib.sha256(f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    destination = cache_dir / f"{key}.jpg"
    # Serialize extraction to avoid parallel decoders and duplicate work.
    with _LOCK:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            destination.touch()
            return destination.read_bytes()
        with tempfile.TemporaryDirectory(prefix="poster-", dir=cache_dir) as temporary:
            frame = source
            if source.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
                executable = shutil.which("ffmpeg")
                if executable is None:
                    import imageio_ffmpeg
                    executable = imageio_ffmpeg.get_ffmpeg_exe()
                frame = Path(temporary) / "frame.jpg"
                subprocess.run(
                    [
                        executable, "-nostdin", "-v", "error", "-threads", "1", "-i",
                        str(source), "-frames:v", "1", "-vf", "scale=320:320:force_original_aspect_ratio=decrease",
                        "-threads", "1", "-y",
                        str(frame)
                    ],
                    check=True,
                    capture_output=True,
                    timeout=15,
                )
            with Image.open(frame) as original:
                poster = ImageOps.exif_transpose(original)
                poster.thumbnail((320, 320))
                staged = Path(temporary) / "poster.jpg"
                poster.convert("RGB").save(staged, "JPEG", quality=75)
            staged.replace(destination)
        posters = sorted(cache_dir.glob("*.jpg"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for expired in posters[MAX_CACHED_POSTERS:]:
            expired.unlink(missing_ok=True)
        # Read before releasing the lock so eviction cannot race a response.
        return destination.read_bytes()
