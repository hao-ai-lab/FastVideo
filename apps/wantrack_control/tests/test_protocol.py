from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from apps.wantrack_control.media import EncodedMediaSegment
from apps.wantrack_control.server import create_app


@dataclass
class _Prepared:
    image: Image.Image
    prompt: str


@dataclass
class _Block:
    block_index: int
    pixel_frames: np.ndarray
    applied_revision: int
    radius: float = 0.15
    active_handle_ids: tuple[str, ...] = ("h", )


class _FakeSession:
    def __init__(
        self,
        gates: list[threading.Event],
        *,
        fail_at: int | None = None,
    ) -> None:
        self.gates = gates
        self.fail_at = fail_at
        self.block_index = 0
        self.pending_revision = 0
        self.closed_reason = None

    def start(self, image, prompt, handles, sampling, *, radius):
        assert image.prompt == prompt
        assert handles
        assert sampling["steps"] > 0
        assert radius > 0

    def apply_control_revision(self, revision, **kwargs):
        del kwargs
        if revision <= self.pending_revision:
            return False
        self.pending_revision = revision
        return True

    def generate_next_block(self):
        index = self.block_index
        applied = self.pending_revision
        if index < len(self.gates):
            assert self.gates[index].wait(timeout=5)
        if self.fail_at == index:
            raise RuntimeError("fake generation failure")
        self.block_index += 1
        return _Block(
            block_index=index,
            pixel_frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
            applied_revision=applied,
        )

    def close(self, reason):
        self.closed_reason = reason


class _FakeRuntime:
    fps = 16.0
    chunk_size = 3

    def __init__(self, session):
        self.session = session

    def prepare(self, image_bytes, prompt):
        assert image_bytes
        return _Prepared(Image.new("RGB", (16, 12), "blue"), prompt)

    def create_session(self):
        return self.session


class _FakeWriter:
    def __init__(self, output_root, *, fps):
        assert fps == 16.0
        self.session_id = "fake-download"
        self.session_dir = Path(output_root) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.block_paths = []

    def encode_block(self, frames, block_index):
        assert frames.shape[-1] == 3
        path = self.session_dir / f"block-{block_index}.mp4"
        path.write_bytes(b"complete-block")
        self.block_paths.append(path)
        return EncodedMediaSegment(
            block_index=block_index,
            init_bytes=b"init",
            media_bytes=f"media-{block_index}".encode(),
            path=path,
        )

    def finalize(self):
        if not self.block_paths:
            return None
        path = self.session_dir / "wantrack_control.mp4"
        path.write_bytes(b"".join(item.read_bytes()
                                 for item in self.block_paths))
        return path


def _json(websocket):
    message = websocket.receive()
    assert message["type"] == "websocket.send"
    assert "text" in message
    import json
    return json.loads(message["text"])


def _bytes(websocket):
    message = websocket.receive()
    assert message["type"] == "websocket.send"
    return message["bytes"]


def _prepare_and_start(websocket):
    import base64
    import io

    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    websocket.send_json({
        "type": "prepare",
        "image": base64.b64encode(image.getvalue()).decode(),
        "prompt": "",
    })
    assert _json(websocket)["type"] == "prepared"
    websocket.send_json({
        "type": "start",
        "handles": [{
            "id": "h",
            "x": 0.5,
            "y": 0.5,
        }],
        "radius": 0.15,
        "steps": 2,
    })
    assert _json(websocket)["type"] == "session_started"


def test_two_block_binary_order_future_update_and_stop(tmp_path):
    gates = [threading.Event(), threading.Event()]
    session = _FakeSession(gates)
    app = create_app(
        _FakeRuntime(session),
        output_dir=tmp_path,
        writer_factory=_FakeWriter,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            _prepare_and_start(websocket)
            assert _json(websocket) == {
                "type": "block_started",
                "block_index": 0,
            }
            websocket.send_json({
                "type": "control_update",
                "revision": 1,
                "samples": [{
                    "id": "h",
                    "x": 0.7,
                    "y": 0.5,
                    "timestamp_ms": 10,
                }],
            })
            websocket.send_json({
                "type": "control_update",
                "revision": 1,
                "samples": [],
            })
            stale = _json(websocket)
            assert stale["status"] == "ignored_stale"
            gates[0].set()
            assert _json(websocket)["type"] == "media_init"
            assert _bytes(websocket) == b"init"
            assert _bytes(websocket) == b"media-0"
            assert _json(websocket)["type"] == "media_segment_complete"

            assert _json(websocket)["type"] == "block_started"
            websocket.send_json({"type": "stop"})
            gates[1].set()
            applied = _json(websocket)
            assert applied["type"] == "control_applied"
            assert applied["revision"] == 1
            assert _bytes(websocket) == b"media-1"
            assert _json(websocket)["type"] == "media_segment_complete"
            complete = _json(websocket)
            assert complete["type"] == "stream_complete"
            assert complete["blocks"] == 2
            assert complete["download_url"]
        response = client.get(complete["download_url"])
        assert response.status_code == 200
        assert response.content
        assert client.get("/healthz").json()["active"] is False


def test_error_releases_lock_and_preserves_completed_prefix(tmp_path):
    gates = [threading.Event(), threading.Event()]
    gates[0].set()
    gates[1].set()
    session = _FakeSession(gates, fail_at=1)
    app = create_app(
        _FakeRuntime(session),
        output_dir=tmp_path,
        writer_factory=_FakeWriter,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            _prepare_and_start(websocket)
            assert _json(websocket)["type"] == "block_started"
            assert _json(websocket)["type"] == "media_init"
            assert _bytes(websocket) == b"init"
            assert _bytes(websocket) == b"media-0"
            assert _json(websocket)["type"] == "media_segment_complete"
            assert _json(websocket)["type"] == "block_started"
            error = _json(websocket)
            assert error["type"] == "error"
            assert error["download_url"]
        assert client.get(error["download_url"]).content
        assert client.get("/healthz").json()["active"] is False
