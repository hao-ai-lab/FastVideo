# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for load_image URL retry behavior.

A HF-CDN IncompleteRead mid-body previously killed SSIM runs (build 4202,
test_wan_i2v) because load_image handed PIL a partial ``.raw`` stream with no
retry. These tests mock requests.get: a transient failure must be retried and
a full-body success must yield a usable PIL image.
"""
import io
from unittest import mock

import PIL.Image
import pytest
import requests

from fastvideo.models import vision_utils

URL = "https://example.com/image.png"


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PIL.Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _good_response() -> mock.Mock:
    response = mock.Mock()
    response.content = _png_bytes()
    response.raise_for_status = mock.Mock()
    return response


def test_load_image_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(vision_utils.time, "sleep", lambda _s: None)
    get = mock.Mock(side_effect=[
        # The shape that broke build 4202: connection dropped mid-body.
        requests.exceptions.ChunkedEncodingError(
            "IncompleteRead(1024 bytes read, 2048 more expected)"),
        _good_response(),
    ])
    monkeypatch.setattr(vision_utils.requests, "get", get)

    image = vision_utils.load_image(URL)

    assert get.call_count == 2
    assert get.call_args.kwargs["timeout"] == 30.0
    assert isinstance(image, PIL.Image.Image)
    assert image.size == (4, 4)
    assert image.mode == "RGB"


def test_load_image_raises_after_exhausting_retries(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(vision_utils.time, "sleep", sleeps.append)
    get = mock.Mock(
        side_effect=requests.exceptions.ConnectionError("connection reset"))
    monkeypatch.setattr(vision_utils.requests, "get", get)

    with pytest.raises(requests.exceptions.ConnectionError):
        vision_utils.load_image(URL)

    assert get.call_count == 3
    assert sleeps == [1, 2]
