# SPDX-License-Identifier: Apache-2.0
"""The mock API's job edit and last-frame routes behave like the real server's."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fastvideo_studio import mock_server


@pytest.fixture(autouse=True)
def clean_state():
    mock_server._jobs.clear()
    yield
    mock_server._jobs.clear()


@pytest.fixture
def client():
    return TestClient(mock_server.app)


def _create(client, **over):
    body = {"model_id": "MiniMaxAI/MiniMax-H3", "prompt": "p", "workload_type": "i2v", **over}
    res = client.post("/api/jobs", json=body)
    assert res.status_code == 201
    return res.json()


def test_edits_a_pending_jobs_prompt_and_references(client):
    job = _create(client, name="clip-2", references=[{"source": "/a.png", "media_type": "image"}])
    refs = job["references"] + [{"source": "/b.png", "media_type": "image"}]

    res = client.patch(f"/api/jobs/{job['id']}", json={"prompt": "new prompt", "references": refs})

    assert res.status_code == 200
    body = res.json()
    assert body["prompt"] == "new prompt"
    assert [r["source"] for r in body["references"]] == ["/a.png", "/b.png"]
    assert body["name"] == "clip-2"  # untouched fields stay
    assert client.get(f"/api/jobs/{job['id']}").json()["prompt"] == "new prompt"


def test_a_started_job_cannot_be_edited(client):
    job = _create(client)
    mock_server._jobs[job["id"]]["status"] = "completed"
    res = client.patch(f"/api/jobs/{job['id']}", json={"prompt": "x"})
    assert res.status_code == 400
    assert "only pending, failed, stopped jobs can be edited" in res.json()["detail"]


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_failed_and_stopped_jobs_can_be_edited(client, status):
    job = _create(client)
    mock_server._jobs[job["id"]]["status"] = status
    assert client.patch(f"/api/jobs/{job['id']}", json={"prompt": "again"}).status_code == 200


def test_unknown_fields_are_rejected(client):
    job = _create(client)
    res = client.patch(f"/api/jobs/{job['id']}", json={"status": "completed", "nonsense": 1})
    assert res.status_code == 400
    assert "Not editable: nonsense, status" in res.json()["detail"]


def test_editing_an_unknown_job_is_404(client):
    assert client.patch("/api/jobs/nope", json={"prompt": "x"}).status_code == 404


def test_last_frame_of_a_completed_job(client):
    job = _create(client)
    mock_server._jobs[job["id"]].update(status="completed", output_path="/out/a.mp4")
    res = client.post(f"/api/jobs/{job['id']}/last-frame")
    assert res.status_code == 200
    assert res.json() == {"path": f"/mock/last_frames/last_frame_{job['id']}.png", "media_type": "image"}


@pytest.mark.parametrize("status", ["pending", "failed", "stopped"])
def test_last_frame_of_an_unfinished_job_is_404(client, status):
    job = _create(client)
    mock_server._jobs[job["id"]]["status"] = status
    res = client.post(f"/api/jobs/{job['id']}/last-frame")
    assert res.status_code == 404
    assert "No output" in res.json()["detail"]


def test_last_frame_of_an_unknown_job_is_404(client):
    assert client.post("/api/jobs/nope/last-frame").status_code == 404


# --- queue -------------------------------------------------------------------


def _frame_of(job):
    return [{"source": f"job-last-frame:{job['id']}", "media_type": "image"}]


def _statuses(client, *jobs):
    return [client.get(f"/api/jobs/{j['id']}").json()["status"] for j in jobs]


def _age(job, seconds):
    """Pretend `seconds` have passed since `job` started."""
    record = mock_server._jobs[job["id"]]
    record["started_at"] -= seconds


def test_queued_jobs_run_one_at_a_time_in_order(client):
    a, b, c = (_create(client, name=n) for n in "abc")
    res = client.post("/api/jobs/queue", json={"job_ids": [b["id"], c["id"], a["id"]]})

    assert res.status_code == 200
    assert [j["name"] for j in res.json()] == ["b", "c", "a"]
    assert _statuses(client, a, b, c) == ["queued", "running", "queued"]
    _age(b, 10)
    assert _statuses(client, a, b, c) == ["queued", "completed", "running"]


def test_a_long_idle_queue_is_all_done_on_the_next_poll(client):
    clips = [_create(client, name=f"c{i}") for i in range(3)]
    client.post("/api/jobs/queue", json={"job_ids": [c["id"] for c in clips]})
    for record in mock_server._jobs.values():  # everything happened a minute ago
        if record.get("queued_at"):
            record["queued_at"] -= 60
        if record.get("started_at"):
            record["started_at"] -= 60
    assert _statuses(client, *clips) == ["completed"] * 3
    starts = [mock_server._jobs[c["id"]]["started_at"] for c in clips]
    assert starts == sorted(starts)  # each took over when the previous finished


def test_a_clip_waits_for_the_clip_whose_last_frame_it_uses(client):
    one = _create(client, name="one")
    two = _create(client, name="two", references=_frame_of(one))
    client.post("/api/jobs/queue", json={"job_ids": [one["id"], two["id"]]})
    assert _statuses(client, one, two) == ["running", "queued"]
    _age(one, 10)
    assert _statuses(client, one, two) == ["completed", "running"]


def test_failure_of_a_frame_source_fails_the_clips_after_it(client):
    one = _create(client, name="Clip 1")
    two = _create(client, name="Clip 2", references=_frame_of(one))
    three = _create(client, name="Clip 3", references=_frame_of(two))
    mock_server._jobs[one["id"]]["status"] = "failed"
    client.post("/api/jobs/queue", json={"job_ids": [two["id"], three["id"]]})

    assert _statuses(client, two, three) == ["failed", "failed"]
    error = client.get(f"/api/jobs/{two['id']}").json()["error"]
    assert "Clip 1" in error and "failed" in error


def test_deleting_a_frame_source_fails_the_clips_waiting_on_it(client):
    blocker = _create(client)
    one = _create(client, name="one")
    two = _create(client, name="two", references=_frame_of(one))
    client.post("/api/jobs/queue", json={"job_ids": [blocker["id"], one["id"], two["id"]]})
    assert client.delete(f"/api/jobs/{one['id']}").status_code == 200
    assert _statuses(client, two) == ["failed"]
    assert "deleted" in client.get(f"/api/jobs/{two['id']}").json()["error"]


def test_dequeue_and_stop_return_a_queued_job_to_pending(client):
    blocker, a, b = _create(client), _create(client), _create(client)
    client.post("/api/jobs/queue", json={"job_ids": [blocker["id"], a["id"], b["id"]]})

    assert client.post(f"/api/jobs/{a['id']}/dequeue").json()["status"] == "pending"
    assert client.post(f"/api/jobs/{b['id']}/stop").json()["status"] == "pending"
    assert client.post(f"/api/jobs/{a['id']}/dequeue").status_code == 409
    assert client.post("/api/jobs/nope/dequeue").status_code == 404


def test_queueing_is_all_or_nothing_and_refuses_started_jobs(client):
    a, done = _create(client), _create(client)
    mock_server._jobs[done["id"]]["status"] = "completed"
    res = client.post("/api/jobs/queue", json={"job_ids": [a["id"], done["id"]]})
    assert res.status_code == 409 and "already completed" in res.json()["detail"]
    assert _statuses(client, a) == ["pending"]
    assert client.post("/api/jobs/queue", json={"job_ids": [a["id"], "nope"]}).status_code == 404
    assert client.post("/api/jobs/queue", json={"job_ids": [a["id"], a["id"]]}).status_code == 409

    assert client.post(f"/api/jobs/{a['id']}/queue").status_code == 200
    assert client.post(f"/api/jobs/{a['id']}/queue").status_code == 409  # running now


def test_starting_a_clip_before_its_frame_source_finished_is_refused(client):
    one = _create(client, name="Clip 1")
    two = _create(client, references=_frame_of(one))
    res = client.post(f"/api/jobs/{two['id']}/start")
    assert res.status_code == 409
    assert "Clip 1" in res.json()["detail"] and "hasn't finished" in res.json()["detail"]

    mock_server._jobs[one["id"]]["status"] = "completed"
    assert client.post(f"/api/jobs/{two['id']}/start").status_code == 200


def test_queued_jobs_cannot_be_edited(client):
    blocker, a = _create(client), _create(client)
    client.post("/api/jobs/queue", json={"job_ids": [blocker["id"], a["id"]]})
    assert client.patch(f"/api/jobs/{a['id']}", json={"prompt": "x"}).status_code == 400


# --- merging a scene -----------------------------------------------------------


def _completed(client, name):
    job = _create(client, name=name)
    record = mock_server._jobs[job["id"]]
    record.update(status="completed", output_path=f"/mock/outputs/{job['id']}/output.mp4", finished_at=1.0)
    return job


def test_merges_finished_clips_into_one_downloadable_video(client):
    clips = [_completed(client, f"wolf-{i}") for i in range(3)]
    res = client.post("/api/scenes/merge", json={"job_ids": [c["id"] for c in clips], "name": "wolf"})

    assert res.status_code == 200
    body = res.json()
    assert body["clips"] == 3 and body["filename"].startswith("wolf-") and body["seconds"] == pytest.approx(3.0, abs=0.1)
    video = client.get(body["url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 1000


def test_refuses_to_merge_a_scene_with_an_unfinished_clip(client):
    done, pending = _completed(client, "a"), _create(client, name="b")
    res = client.post("/api/scenes/merge", json={"job_ids": [done["id"], pending["id"]]})
    assert res.status_code == 409 and "'b' hasn't finished" in res.json()["detail"]


def test_merge_of_an_unknown_job_and_of_nothing(client):
    assert client.post("/api/scenes/merge", json={"job_ids": ["nope"]}).status_code == 404
    assert client.post("/api/scenes/merge", json={"job_ids": []}).status_code == 400


@pytest.mark.parametrize("filename", ["nope.mp4", "..%2Fetc%2Fpasswd", "x.txt"])
def test_an_unknown_merged_file_is_not_found(client, filename):
    assert client.get(f"/api/merged/{filename}").status_code == 404
