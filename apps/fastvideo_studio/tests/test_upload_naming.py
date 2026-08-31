# SPDX-License-Identifier: Apache-2.0
"""Uploaded files keep a readable basename under a unique directory."""
from __future__ import annotations

import pytest

from fastvideo_studio.server import _safe_upload_name


@pytest.mark.parametrize(
    ("filename", "ext", "expected"),
    [
        ("wukong_source.mp4", ".mp4", "wukong_source.mp4"),
        ("MonkeyKing_0.jpg", ".jpg", "MonkeyKing_0.jpg"),
        ("my clip (final).mp4", ".mp4", "my_clip_final.mp4"),
        ("../../etc/passwd.png", ".png", "passwd.png"),
        ("/abs/path/frame.png", ".png", "frame.png"),
        ("émoji✨.png", ".png", "moji.png"),
        ("", ".png", "upload.png"),
        (None, ".png", "upload.png"),
        ("...", ".png", "upload.png"),
    ],
)
def test_safe_upload_name(filename, ext, expected):
    assert _safe_upload_name(filename, ext) == expected


def test_long_names_are_capped():
    out = _safe_upload_name("x" * 300 + ".png", ".png")
    assert out == "x" * 80 + ".png"


def test_no_path_separators_survive():
    for bad in ("a/b.png", "a\\b.png", "../x.png"):
        assert "/" not in _safe_upload_name(bad, ".png")
        assert "\\" not in _safe_upload_name(bad, ".png")
