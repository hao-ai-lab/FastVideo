# SPDX-License-Identifier: Apache-2.0
"""Tests for the session init-image persistence helper."""
from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

from fastvideo.entrypoints.streaming.session_init_image import (
    persist_session_init_image,
)


def _png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestPersistSessionInitImage:

    def test_none_payload_returns_none(self):
        assert persist_session_init_image(None) is None
        assert persist_session_init_image({}) is None

    def test_non_object_payload_rejected(self):
        with pytest.raises(ValueError):
            persist_session_init_image("not-a-dict")

    def test_png_payload_persists(self, tmp_path):
        data = _png_bytes()
        image = persist_session_init_image({
            "mime": "image/png",
            "name": "ref.png",
            "data": base64.b64encode(data).decode("ascii"),
        }, output_dir=str(tmp_path))
        assert image is not None
        assert os.path.exists(image.path)
        assert image.mime == "image/png"
        assert image.path.endswith(".png")
        with open(image.path, "rb") as f:
            assert f.read() == data

    def test_unknown_mime_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mime"):
            persist_session_init_image({
                "mime": "image/bmp",
                "data": "ignored",
            }, output_dir=str(tmp_path))

    def test_bad_base64_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="base64"):
            persist_session_init_image({
                "mime": "image/png",
                "data": "not!base64!",
            }, output_dir=str(tmp_path))

    def test_empty_data_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            persist_session_init_image({
                "mime": "image/png",
                "data": "",
            }, output_dir=str(tmp_path))

    def test_display_name_sanitized(self, tmp_path):
        image = persist_session_init_image({
            "mime": "image/png",
            "name": "../evil/../name.png",
            "data": base64.b64encode(_png_bytes()).decode("ascii"),
        }, output_dir=str(tmp_path))
        assert image is not None
        assert image.display_name == "name.png"

    def test_oversize_rejected(self, tmp_path):
        from fastvideo.entrypoints.streaming import session_init_image as mod

        original = mod._MAX_IMAGE_BYTES
        mod._MAX_IMAGE_BYTES = 100
        try:
            with pytest.raises(ValueError, match="limit"):
                persist_session_init_image({
                    "mime": "image/png",
                    "data": base64.b64encode(_png_bytes((512, 512))).decode(
                        "ascii"),
                }, output_dir=str(tmp_path))
        finally:
            mod._MAX_IMAGE_BYTES = original
