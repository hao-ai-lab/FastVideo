# pyright: reportMissingTypeArgument=false
import base64
import io
from pathlib import Path

from PIL import Image
import pytest

from dreamverse.session_init_image import (
    MAX_SESSION_INIT_IMAGE_BYTES,
    cleanup_session_init_image,
    persist_session_init_image,
)


def make_data_url(format_name: str = "PNG") -> str:
    image = Image.new("RGB", (2, 2), color=(16, 32, 64))
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if format_name == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def test_persist_session_init_image_saves_normalized_file(tmp_path: Path):
    session_image = persist_session_init_image(
        {
            "name": "frame.png",
            "mime_type": "image/png",
            "data_url": make_data_url("PNG"),
        },
        temp_root=tmp_path,
    )

    assert session_image is not None
    assert session_image.display_name == "frame.png"
    assert session_image.file_path.is_file()

    cleanup_session_init_image(session_image)
    assert not session_image.temp_dir.exists()


def test_persist_session_init_image_returns_none_when_missing_data():
    assert persist_session_init_image(None) is None
    assert persist_session_init_image({"name": "frame.png", "data_url": ""}) is None


def test_persist_session_init_image_rejects_unsupported_mime():
    with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
        persist_session_init_image(
            {
                "name": "frame.gif",
                "mime_type": "image/gif",
                "data_url": "data:image/gif;base64,R0lGODlhAQABAAAAACw=",
            }
        )


def test_persist_session_init_image_rejects_large_payload(monkeypatch):
    data_url = make_data_url("PNG")
    oversized = "a" * (MAX_SESSION_INIT_IMAGE_BYTES + 1)

    def fake_b64decode(value: str, validate: bool = True):
        return oversized.encode("ascii")

    monkeypatch.setattr(base64, "b64decode", fake_b64decode)

    with pytest.raises(ValueError, match="15 MB or smaller"):
        persist_session_init_image(
            {
                "name": "frame.png",
                "mime_type": "image/png",
                "data_url": data_url,
            }
        )
