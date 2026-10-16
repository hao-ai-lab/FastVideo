# SPDX-License-Identifier: Apache-2.0
"""The Studio API with a FastH3-Omni export configured: catalogue, create, edit, duplicate, previews."""
import pytest
from fastapi.testclient import TestClient

from fastvideo_studio import fasth3_omni, server
from fastvideo_studio.database import Database
from fastvideo_studio.fasth3_omni import FastH3OmniRegistry
from fastvideo_studio.job_runner import JobRunner
from fastvideo_studio.tests.test_fasth3_omni import REFS, make_base, make_export


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(fasth3_omni, "hardware_status", lambda tile: {"supported": False, "reason": "sm_89"})
    export = make_export(tmp_path / "omni_export")
    registry = FastH3OmniRegistry([str(export), str(tmp_path / "missing")], str(make_base(tmp_path / "base")),
                                  str(tmp_path / "cache"))
    runner = JobRunner(output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"),
                       database=Database(tmp_path / "studio.db"), fasth3_omni=registry)
    monkeypatch.setattr(server, "fasth3_omni", registry)
    monkeypatch.setattr(server, "job_runner", runner, raising=False)
    monkeypatch.setattr(server, "database", None)
    monkeypatch.setattr(server, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(server, "_available_models", [{"id": "MiniMaxAI/MiniMax-H3", "label": "MiniMax H3"}])
    return TestClient(server.app)


def _omni_id(tmp_path):
    return str(tmp_path / "omni_export")


def _payload(tmp_path, **over):
    payload = {"model_id": _omni_id(tmp_path), "prompt": "p", "workload_type": "i2v", "references": REFS,
               "num_inference_steps": 50, "guidance_scale": 5.0, "negative_prompt": "blurry", "vsa_sparsity": 0.0}
    payload.update(over)
    return payload


def test_catalog_lists_omni_with_recipe_hardware_and_errors(client, tmp_path):
    models = {m["id"]: m for m in client.get("/api/models", params={"workload_type": "i2v"}).json()}
    omni = models[_omni_id(tmp_path)]
    assert omni["recipe"]["contract"]["num_inference_steps"] == 8
    assert omni["hardware"] == {"supported": False, "reason": "sm_89"}
    broken = models[str(tmp_path / "missing")]
    assert broken["recipe"] is None and broken["error"]


def test_create_locks_recipe_over_form_defaults(client, tmp_path):
    resp = client.post("/api/jobs", json=_payload(tmp_path))
    assert resp.status_code == 201, resp.text
    job = resp.json()
    assert (job["num_inference_steps"], job["guidance_scale"], job["negative_prompt"]) == (8, 1.0, "")
    assert job["vsa_sparsity"] == 0.9
    assert job["references"] == REFS
    assert job["recipe"]["source"] == "fastvideo_inference.json"


@pytest.mark.parametrize("over,status,match", [
    ({"references": [REFS[2]]}, 400, "audio alone"),
    ({"references": []}, 400, "at least one ordered reference"),
    ({"image_path": "/start.png"}, 400, "start or end frame"),
    ({"num_gpus": 3}, 400, "attention heads"),
])
def test_create_rejects_invalid_omni_requests(client, tmp_path, over, status, match):
    resp = client.post("/api/jobs", json=_payload(tmp_path, **over))
    assert resp.status_code == status and match in resp.json()["detail"]


def test_unusable_export_cannot_be_used(client, tmp_path):
    resp = client.post("/api/jobs", json=_payload(tmp_path, model_id=str(tmp_path / "missing")))
    assert resp.status_code == 400 and "unusable" in resp.json()["detail"]


def test_request_paths_are_not_models(client, tmp_path):
    """Only configured exports are models; an arbitrary directory in a request is rejected."""
    other = make_export(tmp_path / "unconfigured")
    resp = client.post("/api/jobs", json=_payload(tmp_path, model_id=str(other)))
    assert resp.status_code == 400 and "Unknown model_id" in resp.json()["detail"]


def test_references_need_a_ref2va_model(client, tmp_path):
    resp = client.post("/api/jobs", json=_payload(tmp_path, model_id="MiniMaxAI/MiniMax-H3"))
    assert resp.status_code == 201
    server._available_models.append({"id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "label": "Wan"})
    resp = client.post("/api/jobs", json=_payload(tmp_path, model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"))
    assert resp.status_code == 400 and "does not accept references" in resp.json()["detail"]


def test_edit_reorders_and_keeps_recipe_locked(client, tmp_path):
    job = client.post("/api/jobs", json=_payload(tmp_path)).json()
    reordered = [REFS[1], REFS[0], REFS[2]]
    resp = client.patch(f"/api/jobs/{job['id']}",
                        json=_payload(tmp_path, references=reordered, num_inference_steps=50, guidance_scale=7.0,
                                      recipe={"kind": "forged"}))
    assert resp.status_code == 200, resp.text
    edited = resp.json()
    assert edited["references"] == reordered
    assert (edited["num_inference_steps"], edited["guidance_scale"]) == (8, 1.0)
    assert edited["recipe"]["kind"] == "fasth3_omni_pdd"


def test_edit_validates_like_create(client, tmp_path):
    job = client.post("/api/jobs", json=_payload(tmp_path)).json()
    resp = client.patch(f"/api/jobs/{job['id']}", json={"references": [REFS[2]]})
    assert resp.status_code == 400 and "audio alone" in resp.json()["detail"]


def test_duplicate_keeps_references_and_recipe(client, tmp_path):
    job = client.post("/api/jobs", json=_payload(tmp_path)).json()
    dup = client.post(f"/api/jobs/{job['id']}/duplicate").json()
    assert dup["references"] == REFS and dup["recipe"] == job["recipe"]


def test_upload_preview_serves_only_uploads(client, tmp_path):
    resp = client.post("/api/upload-media", files={"file": ("ref.png", b"png-bytes", "image/png")})
    path = resp.json()["path"]
    preview = client.get("/api/uploads/preview", params={"path": path})
    assert preview.status_code == 200 and preview.content == b"png-bytes"
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"x")
    assert client.get("/api/uploads/preview", params={"path": str(secret)}).status_code == 404
    traversal = f"{tmp_path / 'uploads'}/../secret.png"
    assert client.get("/api/uploads/preview", params={"path": traversal}).status_code == 404
