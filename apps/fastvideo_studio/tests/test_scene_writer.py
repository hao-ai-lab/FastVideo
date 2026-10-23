# SPDX-License-Identifier: Apache-2.0
"""Scene writer pieces that run without a model: settings, job forms, request validation."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastvideo_studio import mock_server
from fastvideo_studio.scene_writer import runner


@pytest.fixture
def client():
    return TestClient(mock_server.app)


def test_settings_file_values_apply_unless_the_environment_sets_them(tmp_path, monkeypatch):
    settings_file = tmp_path / "scene-writer.env"
    settings_file.write_text("# comment\nSCENE_WRITER_PROVIDER=cerebras\nexport SCENE_WRITER_MODEL='qwen-3.8-27b'\n"
                             "SCENE_WRITER_API_KEY=\n")
    monkeypatch.setenv("SCENE_WRITER_SETTINGS_FILE", str(settings_file))
    for name in ("SCENE_WRITER_PROVIDER", "SCENE_WRITER_MODEL", "SCENE_WRITER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    settings = runner.SceneWriterSettings()
    assert (settings.provider, settings.model, settings.api_key) == ("cerebras", "qwen-3.8-27b", "")

    monkeypatch.setenv("SCENE_WRITER_MODEL", "gpt-oss-120b")
    assert runner.SceneWriterSettings().model == "gpt-oss-120b"


def test_drafts_go_under_the_server_data_dir_unless_a_runs_root_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_WRITER_SETTINGS_FILE", str(tmp_path / "none.env"))
    monkeypatch.delenv("SCENE_WRITER_RUNS_ROOT", raising=False)
    monkeypatch.setattr(runner, "_server_runs_root", None)
    assert runner.SceneWriterSettings().runs_root == Path.home() / "h3_scene_writer"

    runner.use_server_data_dir(tmp_path / "ui_data")
    assert runner.SceneWriterSettings().runs_root == tmp_path / "ui_data" / "scene_writer"

    monkeypatch.setenv("SCENE_WRITER_RUNS_ROOT", str(tmp_path / "elsewhere"))
    assert runner.SceneWriterSettings().runs_root == tmp_path / "elsewhere"


def test_cerebras_defaults_to_its_model_and_routes_the_key_to_the_child():
    settings = runner.SceneWriterSettings(provider="cerebras", model=None, api_key="k", node_bin_dir=None)
    assert settings.resolved_model() == "qwen-3.8-27b"
    assert settings.child_env() == {runner.PI_AI_KEY_ENV: "k"}


def test_a_model_missing_from_the_harness_catalog_is_declared_with_image_input(tmp_path):
    def patch_for(model):
        settings = runner.SceneWriterSettings(provider="cerebras", model=model, api_key="k", node_bin_dir=None)
        runner._write_patch(tmp_path / "p.yml", tmp_path / "jobs", [], [], None, settings)
        return (tmp_path / "p.yml").read_text()

    assert '"id": "qwen-3.8-27b"' in patch_for("qwen-3.8-27b")
    assert '"input": ["text", "image"]' in patch_for("qwen-3.8-27b")
    assert f"requestImagePixelBudget: {runner.IMAGE_PIXEL_BUDGET}" in patch_for("qwen-3.8-27b")
    # A catalog model keeps the catalog's own entry.
    assert "models:" not in patch_for("gpt-oss-120b")


def test_take_stills_are_spread_over_the_clip_and_named_by_time(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    video = tmp_path / "take.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        for i in range(24 * 6):  # 6 s
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), i % 255, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())

    stills = runner.extract_frames(str(video), tmp_path / "frames", width=160)
    assert [p.name for p in stills] == ["t01.0s.jpg", "t03.0s.jpg", "t05.0s.jpg"]
    assert runner.extract_frames(str(tmp_path / "missing.mp4"), tmp_path / "none") == []


def test_studio_job_becomes_a_job_file_without_runtime_state():
    studio_job = {
        "id": "abc", "status": "completed", "output_path": "/out.mp4", "name": "s-clip-01",
        "model_id": "MiniMaxAI/MiniMax-H3", "prompt": "line one\nline two", "num_frames": 48, "fps": 24,
        "references": [{"source": "/a.png", "media_type": "image", "extra": 1}], "negative_prompt": None,
    }
    assert runner.job_file_form(studio_job) == {
        "name": "s-clip-01", "model_id": "MiniMaxAI/MiniMax-H3", "prompt": ["line one", "line two"],
        "num_frames": 48, "fps": 24, "references": [{"source": "/a.png", "media_type": "image"}],
    }


def test_a_draft_needs_references_or_permission_to_find_them(client):
    body = {"script": "x", "scene_name": "a", "references": [], "instructions": ""}
    assert client.post("/api/scene-writer/drafts", json=body).status_code == 422


def test_a_revision_needs_feedback_and_a_named_job(client):
    assert client.post("/api/scene-writer/revisions", json={"job": {"name": "a-clip-01"}, "feedback": ""}).status_code == 422
    assert client.post("/api/scene-writer/revisions", json={"job": {}, "feedback": "too many slaps"}).status_code == 422
    assert client.get("/api/scene-writer/revisions/missing").status_code == 404


def test_mock_duplicate_makes_a_pending_take_with_the_same_config(client):
    job = client.post("/api/jobs", json={"model_id": "MiniMaxAI/MiniMax-H3", "name": "t-clip-01", "prompt": "p",
                                         "num_frames": 48,
                                         "references": [{"source": "/x.png", "media_type": "image"}]}).json()
    take = client.post(f"/api/jobs/{job['id']}/duplicate").json()

    assert take["id"] != job["id"]
    assert (take["name"], take["status"], take["references"], take["num_frames"]) == (
        "t-clip-01", "pending", job["references"], 48)
    assert client.post("/api/jobs/missing/duplicate").status_code == 404


def test_a_provider_error_names_the_cause():
    settings = runner.SceneWriterSettings(provider="cerebras", api_key="k", node_bin_dir=None)

    def ended_with(message):
        return [{"type": "turn/end", "data": {"reason": {"kind": "error", "error": {"message": message}}}}]

    assert "402 Payment Required" in runner._turn_error_message(ended_with("402 status code (no body)"), settings)
    assert "rejected SCENE_WRITER_API_KEY" in runner._turn_error_message(ended_with("401 Unauthorized"), settings)
    assert runner._turn_error_message(ended_with("socket hang up"), settings) == "The model call failed: socket hang up"
