"""Contract regressions independent of CUDA and model weights."""

import io
import asyncio
import shutil
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from dreamverse import assets, generation_inputs
from dreamverse.generation_inputs import resolve_generation_inputs
from dreamverse.routes import assets as asset_routes
from dreamverse.tests.test_mock_server import _FakeWebSocket


@pytest.fixture
def library(monkeypatch):
    store = assets.AssetStore()
    monkeypatch.setattr(asset_routes, "asset_store", store)
    monkeypatch.setattr(generation_inputs, "asset_store", store)
    app = FastAPI()
    app.include_router(asset_routes.router)
    with TestClient(app) as client:
        yield store, client


def upload_image(client, color="red"):
    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(image_bytes, format="PNG")
    response = client.post("/assets", content=image_bytes.getvalue(),
                           headers={"Content-Type": "image/png", "X-Asset-Name": "frame.png"})
    assert response.status_code == 201, response.text
    return response.json()


def conditioning(asset, role):
    return {"asset_id": asset["asset_id"], "role": role}


def test_assets_validate_content_and_support_head_range_and_delete(library):
    store, client = library
    asset = upload_image(client)
    assert set(asset) == {"asset_id", "kind", "name", "mime_type", "size", "url"}
    assert client.head(asset["url"]).status_code == 200
    response = client.get(asset["url"], headers={"Range": "bytes=0-7"})
    assert response.status_code == 206
    assert response.content == b"\x89PNG\r\n\x1a\n"
    assert client.post("/assets", content=b"not an image", headers={"Content-Type": "image/png"}).status_code == 400
    assert client.post("/assets", content=b"<svg/>", headers={"Content-Type": "image/svg+xml"}).status_code == 415
    assert client.post("/assets", content=b"", headers={"Content-Type": "image/png",
                                                      "Content-Length": str(assets.IMAGE_LIMIT + 1)}).status_code == 413
    with pytest.raises(ValueError, match="Invalid asset ID"):
        store.get("../../etc/passwd")
    assert client.delete(asset["url"]).status_code == 204
    assert client.head(asset["url"]).status_code == 404


def test_generation_pin_prevents_deletion_until_session_releases(library):
    _, client = library
    asset = upload_image(client)
    inputs = resolve_generation_inputs({"generation_mode": "fl2va", "conditioning_assets": [
        conditioning(asset, "first_frame")
    ]}, "full-h3")
    generation_inputs.pin_generation_inputs(inputs)
    generation_inputs.pin_generation_inputs(inputs)
    assert client.delete(asset["url"]).status_code == 409
    generation_inputs.release_generation_inputs(inputs)
    assert client.delete(asset["url"]).status_code == 409
    generation_inputs.release_generation_inputs(inputs)
    assert client.delete(asset["url"]).status_code == 204


def test_legacy_init_remains_compatible_but_explicit_t2va_is_text_only(library):
    _, client = library
    assert resolve_generation_inputs({"initial_image": {"old": "payload"}}, "fast-ltx2").mode is None
    assert resolve_generation_inputs({"generation_mode": "t2va"}, "fast-ltx2").mode == "t2va"
    with pytest.raises(ValueError, match="legacy initial_image"):
        resolve_generation_inputs({"generation_mode": "t2va", "initial_image": {}}, "full-h3")
    asset = upload_image(client)
    with pytest.raises(ValueError, match="text only"):
        resolve_generation_inputs({"generation_mode": "t2va", "conditioning_assets": [
            conditioning(asset, "reference")
        ]}, "full-h3")


@pytest.mark.parametrize("mode", ["unknown", None, 3, [], {}])
def test_unknown_mode_fails_before_assets_are_resolved(mode):
    with pytest.raises(ValueError, match="Unknown generation mode"):
        resolve_generation_inputs({"generation_mode": mode}, "full-h3")


@pytest.mark.parametrize("model_id", ["fast-h3", "fast-ltx2", "fast-ltx23"])
def test_preview_and_ltx_cannot_advertise_full_h3_modes(model_id):
    with pytest.raises(ValueError, match="Full H3"):
        resolve_generation_inputs({"generation_mode": "ref2va"}, model_id)


def test_fl2va_first_required_last_optional_and_roles_unique(library):
    _, client = library
    first = upload_image(client)
    last = upload_image(client, "blue")
    payload = {"generation_mode": "fl2va", "conditioning_assets": [conditioning(first, "first_frame")]}
    inputs = resolve_generation_inputs(payload, "full-h3")
    assert inputs.first_frame_path.endswith(".png")
    assert inputs.last_frame_path is None
    payload["conditioning_assets"].append(conditioning(last, "last_frame"))
    assert resolve_generation_inputs(payload, "full-h3").last_frame_path is not None
    payload["conditioning_assets"].append(conditioning(first, "first_frame"))
    with pytest.raises(ValueError, match="exactly one first-frame"):
        resolve_generation_inputs(payload, "full-h3")
    with pytest.raises(ValueError, match="exactly one first-frame"):
        resolve_generation_inputs({"generation_mode": "fl2va", "conditioning_assets": [
            conditioning(last, "last_frame")
        ]}, "full-h3")


def test_ref_order_is_preserved_and_limits_are_enforced(library):
    _, client = library
    first, second = upload_image(client), upload_image(client, "blue")
    refs = [conditioning(second, "reference"), conditioning(first, "reference")]
    payload = {"generation_mode": "ref2va", "conditioning_assets": refs}
    inputs = resolve_generation_inputs(payload, "full-h3")
    assert [asset.asset_id for asset in inputs.references] == [second["asset_id"], first["asset_id"]]
    with pytest.raises(ValueError, match="at most 9 image"):
        resolve_generation_inputs({**payload, "conditioning_assets": refs * 5}, "full-h3")
    with pytest.raises(ValueError, match="without keyframe roles"):
        resolve_generation_inputs({**payload, "conditioning_assets": [conditioning(first, "first_frame")]}, "full-h3")
    with pytest.raises(ValueError, match="at most 12"):
        resolve_generation_inputs({**payload, "conditioning_assets": refs * 7}, "full-h3")


def test_ref_audio_requires_visual_reference(library, monkeypatch):
    store, _ = library
    monkeypatch.setattr(store, "get", lambda asset_id: assets.StoredAsset(asset_id, "audio", "/audio.wav", "audio",
                                                                         "audio/wav", 100))
    with pytest.raises(ValueError, match="audio alone"):
        resolve_generation_inputs({"generation_mode": "ref2va", "conditioning_assets": [
            {"asset_id": "a" * 32, "role": "reference"}
        ]}, "full-h3")


def test_ref_rejects_extreme_image_aspect_before_gpu(library):
    _, client = library
    content = io.BytesIO()
    Image.new("RGB", (500, 50), "blue").save(content, format="PNG")
    response = client.post("/assets", content=content.getvalue(), headers={"Content-Type": "image/png"})
    assert response.status_code == 201
    with pytest.raises(ValueError, match="aspect ratios"):
        resolve_generation_inputs({"generation_mode": "ref2va", "conditioning_assets": [
            conditioning(response.json(), "reference")
        ]}, "full-h3")


def test_audio_upload_rejects_surround_sound(library, monkeypatch):
    import json
    _, client = library
    monkeypatch.setattr(assets.shutil, "which", lambda name: "/usr/bin/ffprobe")
    info = {"format": {"format_name": "wav", "duration": "1"},
            "streams": [{"codec_type": "audio", "channels": 6}]}
    monkeypatch.setattr(assets.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        [], 0, stdout=json.dumps(info).encode(), stderr=b""))
    response = client.post("/assets", content=b"surround wav", headers={"Content-Type": "audio/wav"})
    assert response.status_code == 400
    assert "mono or stereo" in response.json()["detail"]


@pytest.mark.parametrize("entries", [None, {}, "x", [{"path": "/etc/passwd", "role": "reference"}],
                                    [{"asset_id": "x", "role": "unknown"}]])
def test_malformed_conditioning_is_rejected(entries):
    with pytest.raises(ValueError):
        resolve_generation_inputs({"generation_mode": "ref2va", "conditioning_assets": entries}, "full-h3")


@pytest.mark.parametrize("mode", ["t2va", "fl2va", "ref2va"])
def test_mock_streams_all_valid_modes_and_releases_assets(library, monkeypatch, mode):
    from dreamverse import mock_server
    _, client = library
    monkeypatch.setattr(mock_server, "MOCK_SEGMENT_BYTES", b"mock-fmp4")
    monkeypatch.setattr(mock_server, "LATENCY_MS", 1)
    image = upload_image(client)
    refs = [] if mode == "t2va" else [conditioning(image, "first_frame" if mode == "fl2va" else "reference")]
    ws = _FakeWebSocket([
        (0, {"type": "session_init_v2", "generation_mode": mode, "conditioning_assets": refs,
             "curated_prompts": ["A bird flies over a lake."], "single_clip_mode": True,
             "enhancement_enabled": False}),
        (0.15, {"type": "leave"}),
    ])
    asyncio.run(mock_server.websocket_endpoint(ws))
    assert not [event for event in ws.sent_json if event["type"] == "error"]
    assert any(event["type"] == "media_segment_complete" for event in ws.sent_json)
    assert ws.sent_bytes
    assert client.delete(image["url"]).status_code == 204


def test_mock_rejects_invalid_mode_before_gpu_assignment(library):
    from dreamverse import mock_server
    ws = _FakeWebSocket([(0, {"type": "session_init_v2", "generation_mode": "fl2va"})])
    asyncio.run(mock_server.websocket_endpoint(ws))
    assert not any(event["type"] == "gpu_assigned" for event in ws.sent_json)
    errors = [event for event in ws.sent_json if event["type"] == "error"]
    assert errors[0]["error_code"] == "invalid_generation_input"
    assert "first-frame" in errors[0]["message"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg + ffprobe required")
@pytest.mark.parametrize("kind,mime,suffix", [("video", "video/mp4", ".mp4"), ("audio", "audio/wav", ".wav")])
def test_actual_video_and_audio_upload_validation(library, tmp_path, kind, mime, suffix):
    _, client = library
    media_path = tmp_path / f"sample{suffix}"
    source = "testsrc2=size=64x64:rate=24" if kind == "video" else "sine=frequency=440:sample_rate=24000"
    command = [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", source, "-t", "0.5", str(media_path)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    response = client.post("/assets", content=media_path.read_bytes(), headers={"Content-Type": mime})
    assert response.status_code == 201, response.text
    assert response.json()["kind"] == kind
    response = client.post("/assets", content=b"#EXTM3U\nhttp://example.com/stream", headers={"Content-Type": mime})
    assert response.status_code == 400
