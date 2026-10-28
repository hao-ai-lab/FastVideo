# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/streamv2/generate`` with a fake generator and a fake encoder.

CPU-only: no model weights and no ffmpeg (``test_video_stream.py`` covers the
real encoder).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from fastvideo.api.results import GenerationResult
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.streaming_v2.messages import GenerateRequest
from fastvideo.entrypoints.streaming_v2.routes import generate as generate_route
from fastvideo.entrypoints.streaming_v2.server import ServerState, create_app
from fastvideo.entrypoints.streaming_v2.video_stream import video_mime
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference

_FAKE_CHUNKS = [b"init-segment", b"fragment-1", b"fragment-2"]


def _png_base64(width: int = 32, height: int = 24) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _reference(media_type: str, data: bytes = b"media bytes") -> dict[str, str]:
    return {"media_type": media_type, "data": base64.b64encode(data).decode("ascii")}


def _body(**overrides: Any) -> dict[str, Any]:
    body = {
        "prompt": "a lighthouse at dusk",
        "references": [{"media_type": "image", "data": _png_base64()}],
        "width": 64,
        "height": 32,
        "num_frames": 124,
        "seed": 7,
        "return_last_frame": True,
    }
    body.update(overrides)
    return body


class _FakeGenerator:
    """Returns four distinct frames plus audio and records every request with its reference files' bytes."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[GenerationRequest] = []
        self.reference_bytes: list[list[bytes]] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        self.reference_bytes.append([Path(reference.source).read_bytes() for reference in request.inputs.references])
        if self.error is not None:
            raise self.error
        frames = [np.full((48, 64, 3), 40 * index, dtype=np.uint8) for index in range(4)]
        return GenerationResult(frames=frames, audio=np.zeros(1600, dtype=np.float32), audio_sample_rate=16000)


async def _fake_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
    for chunk in _FAKE_CHUNKS:
        yield chunk


async def _failing_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
    yield _FAKE_CHUNKS[0]
    raise RuntimeError("ffmpeg failed: boom")


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event name, JSON data) pairs."""
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_route, "encode_fmp4", _fake_encode)


@pytest.fixture
def temp_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``tempfile`` at an empty directory so a test can see the reference files the route writes."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_generate_streams_last_frame_video_and_done_in_order(fake_encoder: None) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [name for name, _ in events] == ["last_frame", "video_start", "video_chunk", "video_chunk", "video_chunk", "done"]

    last_frame = Image.open(io.BytesIO(base64.b64decode(events[0][1]["data"])))
    assert last_frame.format == "PNG"
    assert np.array_equal(np.asarray(last_frame), np.full((48, 64, 3), 120, dtype=np.uint8))
    assert events[1][1] == {"mime": video_mime(has_audio=True)}
    assert b"".join(base64.b64decode(data["data"]) for name, data in events if name == "video_chunk") == b"".join(_FAKE_CHUNKS)
    assert set(events[-1][1]["timings"]) == {"generation_ms", "e2e_latency_ms"}

    request = generator.requests[0]
    assert request.prompt == "a lighthouse at dusk"
    assert (request.sampling.width, request.sampling.height, request.sampling.num_frames, request.sampling.seed) == (64, 32, 124, 7)
    assert len(request.inputs.references) == 1 and request.inputs.references[0].media_type == "image"
    assert request.output.save_video is False and request.output.return_frames is True


def test_generate_renders_a_text_request_without_references(fake_encoder: None, temp_root: Path) -> None:
    """A body without references is text-to-video: nothing is validated as a reference and no file is written."""
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body(references=[]))

    assert _events(response.text)[-1][0] == "done"
    assert generator.requests[0].inputs.references == []
    assert list(temp_root.iterdir()) == []


def test_generate_hands_references_to_the_model_as_files_in_order(fake_encoder: None, temp_root: Path) -> None:
    """The route writes each reference to a file in request order, keeps its media type, and removes the files after generation."""
    references = [
        _reference("video", b"video one"),
        _reference("image", b"not decoded here"),
        _reference("audio", b"audio"),
    ]
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body(references=references))

    assert _events(response.text)[-1][0] == "done"
    media_types = [reference.media_type for reference in generator.requests[0].inputs.references]
    assert media_types == ["video", "image", "audio"]
    assert generator.reference_bytes == [[b"video one", b"not decoded here", b"audio"]]
    assert list(temp_root.iterdir()) == []


def test_generate_omits_last_frame_unless_requested(fake_encoder: None) -> None:
    with TestClient(create_app(_FakeGenerator())) as client:
        response = client.post("/v1/streamv2/generate", json=_body(return_last_frame=False))

    assert [name for name, _ in _events(response.text)][:2] == ["video_start", "video_chunk"]


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        json.dumps(_body(width=None)).encode(),
        json.dumps(_body(width=1, height=768)).encode(),
        json.dumps(_body(num_frames=4)).encode(),
        json.dumps(_body(num_frames=1000)).encode(),
        json.dumps({key: value for key, value in _body().items() if key != "num_frames"}).encode(),
        json.dumps(_body(prompt="")).encode(),
        json.dumps(_body(unknown_field=1)).encode(),
        json.dumps(_body(reference_images=[_png_base64()])).encode(),
        json.dumps(_body(references=[{**_reference("image"), "label": "Picture 1"}])).encode(),
        json.dumps(_body(references=[{"media_type": "image", "data": "%%% not base64 %%%"}])).encode(),
        json.dumps(_body(references=[_reference("")])).encode(),
        json.dumps(_body(references=[_reference("text")])).encode(),
        json.dumps(_body(references=[_reference("image")] * 10)).encode(),
        json.dumps(_body(references=[_reference("image")] + [_reference("video")] * 4)).encode(),
        json.dumps(_body(references=[_reference("image")] + [_reference("audio")] * 4)).encode(),
        json.dumps(_body(references=[_reference("image")] * 9 + [_reference("video")] * 3 +
                         [_reference("audio")])).encode(),
        json.dumps(_body(references=[_reference("audio")])).encode(),
    ],
)
def test_generate_rejects_invalid_requests_with_http_400(content: bytes, temp_root: Path) -> None:
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", content=content, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    assert response.json()["message"]
    assert generator.requests == []
    assert list(temp_root.iterdir()) == []  # a refused request writes no reference files


@pytest.mark.parametrize(
    ("error", "code"),
    [(ValueError("reference image outside 1:4..4:1"), "invalid_request"), (RuntimeError("CUDA OOM"), "generation_failed")],
)
def test_generate_maps_generation_failures_to_error_codes(fake_encoder: None, error: Exception, code: str) -> None:
    with TestClient(create_app(_FakeGenerator(error=error))) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    assert response.status_code == 200
    assert _events(response.text) == [("error", {"code": code, "message": str(error)})]


@pytest.mark.parametrize(
    ("error", "code"),
    [(ValueError("bad operator default"), "invalid_request"), (OSError("No space left on device"), "generation_failed")],
)
def test_generate_reports_setup_failure_and_removes_reference_files(monkeypatch: pytest.MonkeyPatch, temp_root: Path,
                                                                    error: Exception, code: str) -> None:
    """A failure while building the generation request ends the stream with an error event and removes the reference files."""
    def failing_build(*args: Any) -> GenerationRequest:
        raise error

    monkeypatch.setattr(generate_route, "build_generation_request", failing_build)
    generator = _FakeGenerator()
    with TestClient(create_app(generator)) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    assert response.status_code == 200
    assert _events(response.text) == [("error", {"code": code, "message": str(error)})]
    assert generator.requests == []
    assert list(temp_root.iterdir()) == []  # the reference files written before the failure are removed


def test_generate_reports_encoding_failure_after_video_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_route, "encode_fmp4", _failing_encode)
    with TestClient(create_app(_FakeGenerator())) as client:
        response = client.post("/v1/streamv2/generate", json=_body())

    events = _events(response.text)
    assert [name for name, _ in events] == ["last_frame", "video_start", "video_chunk", "error"]
    assert events[-1][1] == {"code": "generation_failed", "message": "ffmpeg failed: boom"}


class _GatedGenerator:
    """Blocks each ``generate()`` call until the test releases it and records which calls are running."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Semaphore(0)
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        self.started.release()
        self.release.wait(timeout=10)
        return GenerationResult(frames=[np.zeros((8, 8, 3), dtype=np.uint8)])


def test_lock_covers_generation_but_not_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """While request 1 is still streaming, request 2 already generates."""
    streaming = asyncio.Event()
    finish_streaming = asyncio.Event()

    async def blocking_encode(frames: list[np.ndarray], fps: int, audio: Any = None, sample_rate: Any = None) -> AsyncIterator[bytes]:
        streaming.set()
        await finish_streaming.wait()
        yield b"chunk"

    monkeypatch.setattr(generate_route, "encode_fmp4", blocking_encode)
    generator = _GatedGenerator()
    generator.release.set()

    async def scenario() -> None:
        state = ServerState(generator=generator)
        body = GenerateRequest(prompt="p", width=8, height=8, num_frames=1)
        first = generate_route.stream_segment(state, body, [MiniMaxH3Reference(source=b"image")], 0.0)
        assert (await first.__anext__()).startswith("event: video_start")  # request 1 finished generating
        chunk_task = asyncio.ensure_future(first.__anext__())  # request 1 now blocks while streaming
        await asyncio.wait_for(streaming.wait(), timeout=10)
        assert not state.lock.locked()
        second_result, _ = await asyncio.wait_for(state.generate(GenerationRequest(prompt="p"), reference_dir=tempfile.mkdtemp()),
                                                   timeout=10)
        assert generator.calls == 2 and second_result.frames is not None
        finish_streaming.set()
        assert (await chunk_task).startswith("event: video_chunk")
        await first.aclose()

    asyncio.run(scenario())


def test_lock_and_reference_files_stay_until_generation_returns_when_caller_is_cancelled(tmp_path: Path) -> None:
    """A cancelled caller leaves the lock and the reference directory in place until the generation call returns."""
    generator = _GatedGenerator()
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()

    async def scenario() -> None:
        state = ServerState(generator=generator)
        task = asyncio.ensure_future(state.generate(GenerationRequest(prompt="p"), reference_dir=str(reference_dir)))
        await asyncio.to_thread(generator.started.acquire)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.lock.locked()  # the uninterruptible call still runs
        assert reference_dir.exists()  # and still reads the reference files
        generator.release.set()
        for _ in range(100):
            if not state.lock.locked():
                break
            await asyncio.sleep(0.05)
        assert not state.lock.locked()
        assert not reference_dir.exists()

    asyncio.run(scenario())


def test_a_caller_cancelled_while_queued_never_generates(tmp_path: Path) -> None:
    """A request abandoned while another holds the lock leaves the queue and removes its reference files."""
    generator = _GatedGenerator()
    queued_dir = tmp_path / "queued"
    queued_dir.mkdir()

    async def scenario() -> None:
        state = ServerState(generator=generator)
        running = asyncio.ensure_future(state.generate(GenerationRequest(prompt="p")))
        await asyncio.to_thread(generator.started.acquire)
        queued = asyncio.ensure_future(state.generate(GenerationRequest(prompt="p"), reference_dir=str(queued_dir)))
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert not queued_dir.exists()
        generator.release.set()
        await asyncio.wait_for(running, timeout=10)
        await asyncio.sleep(0.05)
        assert generator.calls == 1
        assert not state.lock.locked()

    asyncio.run(scenario())
