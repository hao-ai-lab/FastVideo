"""T0: dreamverse protocol surface over a fake model (skips without fastapi)."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeCard:
    def digest(self):
        return "d"

    class sampling_defaults:
        fps = 16


class _FakeModel:
    model_id = "fake"
    card = _FakeCard()


def test_session_protocol_handshake(monkeypatch):
    import fastvideo2.dreamverse.server as S
    monkeypatch.setattr(S, "find_ffmpeg", lambda: "ffmpeg")
    app = S.build_app(_FakeModel())
    c = TestClient(app)
    assert c.get("/health").json()["model"] == "fake"
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "nope"})
        assert ws.receive_json()["code"] == "bad_init"
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "session_init_v2"})
        assert ws.receive_json()["type"] == "queue_status"
        assert ws.receive_json()["type"] == "gpu_assigned"
        assert ws.receive_json()["type"] == "stream_start"
        ws.send_json({"type": "segment_prompt_source", "prompt": ""})
        assert ws.receive_json()["code"] == "bad_prompt"
        ws.send_json({"type": "leave"})
        assert ws.receive_json()["type"] == "stream_complete"
