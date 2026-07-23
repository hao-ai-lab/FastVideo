"""T0: serving app construction + schema over a fake model (no torch/GPU;
skips when fastapi isn't installed)."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeCard:
    def digest(self):
        return "deadbeef"

    def to_dict(self):
        return {"model_id": "fake"}

    class sampling_defaults:
        fps = 16


class _FakeModel:
    model_id = "fake"
    card = _FakeCard()

    def describe(self):
        return self.card.to_dict()


def test_app_surfaces():
    from fastvideo2.serve import build_app
    app = build_app(_FakeModel())
    c = TestClient(app)
    assert c.get("/health").json()["model"] == "fake"
    assert c.get("/v1/models").json()["data"][0]["model_id"] == "fake"
    assert c.get("/v1/videos/nope").status_code == 404
    assert c.get("/v1/videos/nope/content").status_code == 404
