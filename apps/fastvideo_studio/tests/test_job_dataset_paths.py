# SPDX-License-Identifier: Apache-2.0
"""Dataset references survive the real create/edit/duplicate API and persistence."""

import pytest
from fastapi.testclient import TestClient

from fastvideo_studio import server
from fastvideo_studio.database import Database
from fastvideo_studio.job_runner import JobRunner, JobStatus


@pytest.fixture
def api(tmp_path, monkeypatch):
    db = Database(tmp_path / "studio.db")
    runner = JobRunner(str(tmp_path / "outputs"), str(tmp_path / "logs"), db)
    datasets = tmp_path / "datasets"
    for dataset_id in ("train-one", "train-two", "validation-one", "validation-two"):
        (datasets / dataset_id).mkdir(parents=True)
    monkeypatch.setattr(server, "database", db)
    monkeypatch.setattr(server, "job_runner", runner, raising=False)
    monkeypatch.setattr(server, "datasets_upload_dir", str(datasets))
    monkeypatch.setattr(server, "_available_models", [{"id": "test-model"}])
    with TestClient(server.app) as client:
        yield client, runner, db, datasets
    runner._shutdown()


def create(client, job_type="finetuning", **overrides):
    payload = {
        "model_id": "test-model",
        "prompt": "training run",
        "job_type": job_type,
        "data_path": "train-one",
        "validation_dataset_file": "validation-one",
        **overrides,
    }
    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("job_type", ["finetuning", "lora", "distillation"])
def test_create_edit_duplicate_and_duplicate_edit_resolve_both_datasets(api, job_type):
    client, _, db, datasets = api
    job = create(client, job_type)
    assert job["data_path"] == str(datasets / "train-one")
    assert job["validation_dataset_file"] == str(datasets / "validation-one")

    updates = {"data_path": "train-two", "validation_dataset_file": "validation-two"}
    edited = client.patch(f"/api/jobs/{job['id']}", json=updates)
    assert edited.status_code == 200
    assert edited.json()["data_path"] == str(datasets / "train-two")
    assert edited.json()["validation_dataset_file"] == str(datasets / "validation-two")

    duplicate = client.post(f"/api/jobs/{job['id']}/duplicate").json()
    assert duplicate["id"] != job["id"]
    for field in updates:
        assert duplicate[field] == edited.json()[field]

    response = client.patch(f"/api/jobs/{duplicate['id']}", json={
        "data_path": "train-one",
        "validation_dataset_file": "validation-one",
    })
    assert response.status_code == 200
    for field in updates:
        assert response.json()[field] == job[field]
        assert db.get_job(duplicate["id"])[field] == job[field]
        # The original job remains independent of its duplicate.
        assert db.get_job(job["id"])[field] == edited.json()[field]


@pytest.mark.parametrize("field", ["data_path", "validation_dataset_file"])
@pytest.mark.parametrize("value", ["/external/processed/dataset", "relative/dataset", "unknown-id", ""])
def test_create_and_edit_preserve_paths_unknown_ids_and_empty_values(api, field, value):
    client, _, db, _ = api
    job = create(client, **{field: value})
    assert job[field] == value
    response = client.patch(f"/api/jobs/{job['id']}", json={field: value})
    assert response.status_code == 200
    assert response.json()[field] == value
    assert db.get_job(job["id"])[field] == value


def test_patch_preserves_omitted_fields_and_can_clear_validation(api):
    client, _, db, _ = api
    job = create(client)
    response = client.patch(f"/api/jobs/{job['id']}", json={"name": "renamed"})
    assert response.status_code == 200
    for field in ("data_path", "validation_dataset_file"):
        assert response.json()[field] == job[field]
    response = client.patch(f"/api/jobs/{job['id']}", json={"validation_dataset_file": ""})
    assert response.status_code == 200
    assert response.json()["validation_dataset_file"] == ""
    assert db.get_job(job["id"])["data_path"] == job["data_path"]


def test_inference_does_not_resolve_dataset_references(api):
    client, _, db, _ = api
    job = create(client, "inference")
    assert job["data_path"] == "train-one"
    response = client.patch(f"/api/jobs/{job['id']}", json={
        "data_path": "train-two",
        "validation_dataset_file": "validation-two",
    })
    assert response.status_code == 200
    assert db.get_job(job["id"])["data_path"] == "train-two"
    assert response.json()["validation_dataset_file"] == "validation-two"


def test_patch_uses_submitted_job_type(api):
    client, _, _, datasets = api
    job = create(client, "inference")
    response = client.patch(f"/api/jobs/{job['id']}", json={
        "job_type": "finetuning", "data_path": "train-two",
    })
    assert response.status_code == 200
    assert response.json()["data_path"] == str(datasets / "train-two")
    # Only submitted fields are normalized, even when changing job type.
    assert response.json()["validation_dataset_file"] == "validation-one"
    response = client.patch(f"/api/jobs/{job['id']}", json={
        "job_type": "inference", "data_path": "train-one",
    })
    assert response.status_code == 200
    assert response.json()["data_path"] == "train-one"


def test_patch_retains_not_found_editability_and_unknown_field_errors(api):
    client, runner, db, _ = api
    assert client.patch("/api/jobs/missing", json={"data_path": "train-one"}).status_code == 404
    job = create(client)
    response = client.patch(f"/api/jobs/{job['id']}", json={"status": "completed"})
    assert response.status_code == 400
    runner.get_job(job["id"]).status = JobStatus.RUNNING
    response = client.patch(f"/api/jobs/{job['id']}", json={"data_path": "train-two"})
    assert response.status_code == 400
    assert db.get_job(job["id"])["data_path"] == job["data_path"]


def test_all_editable_configuration_survives_database_reload(api):
    client, runner, _, datasets = api
    job = create(client, dmd_use_vsa=True)
    updates = {
        "model_id": "different-model",
        "name": "edited name",
        "prompt": "edited description",
        "workload_type": "dmd_t2v",
        "job_type": "distillation",
        "image_path": "/media/first.png",
        "last_image_path": "/media/last.png",
        "references": [{"source": "/media/reference.mp4", "media_type": "video"}],
        "negative_prompt": "negative",
        "num_inference_steps": 17,
        "num_frames": 49,
        "height": 720,
        "width": 1280,
        "guidance_scale": 0.0,
        "guidance_rescale": 0.25,
        "fps": 30,
        "seed": 0,
        "num_gpus": 2,
        "dit_cpu_offload": True,
        "dit_layerwise_offload": True,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": True,
        "image_encoder_cpu_offload": True,
        "use_fsdp_inference": True,
        "enable_torch_compile": True,
        "vsa_sparsity": 0.25,
        "tp_size": 2,
        "sp_size": 2,
        "data_path": "train-two",
        "max_train_steps": 2300,
        "train_batch_size": 3,
        "learning_rate": 0.0,
        "num_latent_t": 24,
        "validation_dataset_file": "",
        "lora_rank": 64,
        "dmd_use_vsa": False,
        "dmd_vsa_sparsity": 0.0,
        "dmd_denoising_steps": "900,500",
        "real_score_guidance_scale": 0.0,
        "generator_update_interval": 7,
        "real_score_model_path": "/models/teacher",
        "fake_score_model_path": "/models/critic",
    }
    assert set(updates) == set(JobRunner.CONFIG_FIELDS)
    response = client.patch(f"/api/jobs/{job['id']}", json=updates)
    assert response.status_code == 200
    expected = {**updates, "data_path": str(datasets / "train-two")}
    # Discard the live objects and reload through the normal startup path.
    runner._jobs.clear()
    runner._load_jobs()
    restored = client.get(f"/api/jobs/{job['id']}").json()
    for field, value in expected.items():
        assert restored[field] == value, field
    assert restored["created_at"] == job["created_at"]
    assert restored["status"] == "pending"
    duplicate = client.post(f"/api/jobs/{job['id']}/duplicate").json()
    for field, value in expected.items():
        assert duplicate[field] == value, field


def test_failed_configuration_write_does_not_change_live_job(api, monkeypatch):
    client, runner, db, _ = api
    job = create(client)

    def fail_write(*args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db, "update_job_config", fail_write)
    with pytest.raises(RuntimeError, match="database unavailable"):
        runner.update_job_config(job["id"], {"data_path": "unsaved"})
    assert runner.get_job(job["id"]).data_path == job["data_path"]


def test_failed_commit_rolls_back_before_another_write(api, monkeypatch):
    client, runner, db, _ = api
    job = create(client)

    def fail_commit():
        raise RuntimeError("commit failed")

    with monkeypatch.context() as patch:
        patch.setattr(db, "_commit", fail_commit)
        with pytest.raises(RuntimeError, match="commit failed"):
            runner.update_job_config(job["id"], {"data_path": "unsaved"})

    assert runner.get_job(job["id"]).data_path == job["data_path"]
    assert db.get_job(job["id"])["data_path"] == job["data_path"]
    db.update_job(job["id"], {"error": "a later runtime update"})
    assert db.get_job(job["id"])["data_path"] == job["data_path"]
    assert Database(db._path).get_job(job["id"])["data_path"] == job["data_path"]
