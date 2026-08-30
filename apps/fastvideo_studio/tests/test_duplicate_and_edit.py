# SPDX-License-Identifier: Apache-2.0
"""Duplicating a job's config, and editing one that has not started."""
import uuid

import pytest

from fastvideo_studio.database import Database
from fastvideo_studio.job_runner import JobRunner, JobStatus


@pytest.fixture
def runner(tmp_path):
    return JobRunner(
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        database=Database(tmp_path / "t.db"),
    )


def _make(runner, **over):
    kwargs = dict(
        job_id=str(uuid.uuid4()),
        model_id="MiniMaxAI/MiniMax-H3",
        prompt="p",
        workload_type="i2v",
        num_frames=141,
        guidance_scale=1.0,
        num_gpus=4,
        references=[{"source": "/x/clip.mp4", "media_type": "video"}],
    )
    kwargs.update(over)
    return runner.create_job(**kwargs)


def test_duplicate_copies_config_but_not_runtime_state(runner):
    src = _make(runner)
    src.status = JobStatus.COMPLETED
    src.output_path = "/x/out.mp4"

    dup = runner.duplicate_job(src.id, str(uuid.uuid4()))

    assert dup.id != src.id
    assert dup.status is JobStatus.PENDING
    assert dup.output_path is None
    for field in ("model_id", "prompt", "workload_type", "num_frames",
                  "guidance_scale", "num_gpus", "references"):
        assert getattr(dup, field) == getattr(src, field)


def test_duplicate_deep_copies_references(runner):
    src = _make(runner)
    dup = runner.duplicate_job(src.id, str(uuid.uuid4()))
    dup.references[0]["source"] = "/changed"
    assert src.references[0]["source"] == "/x/clip.mp4"


def test_duplicate_unknown_job(runner):
    with pytest.raises(ValueError, match="not found"):
        runner.duplicate_job("nope", str(uuid.uuid4()))


def test_edit_pending_job(runner):
    job = _make(runner)
    updated = runner.update_job_config(job.id, {"num_frames": 192, "seed": 7})
    assert updated.num_frames == 192
    assert updated.seed == 7


@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.STOPPED])
def test_edit_allows_restartable_jobs(runner, status):
    """Editable exactly when startable: neither has produced an output."""
    job = _make(runner)
    job.status = status
    assert runner.update_job_config(job.id, {"seed": 7}).seed == 7


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.RUNNING])
def test_edit_rejects_jobs_with_or_producing_a_result(runner, status):
    job = _make(runner)
    job.status = status
    with pytest.raises(ValueError, match="can be edited"):
        runner.update_job_config(job.id, {"seed": 7})


def test_edit_rejects_unknown_field(runner):
    job = _make(runner)
    with pytest.raises(ValueError, match="Not editable"):
        runner.update_job_config(job.id, {"status": "completed"})


def test_name_is_carried_by_duplicate(runner):
    src = _make(runner, name="wukong swap v2")
    dup = runner.duplicate_job(src.id, str(uuid.uuid4()))
    assert dup.name == "wukong swap v2"


def test_name_is_editable(runner):
    job = _make(runner, name="a")
    assert runner.update_job_config(job.id, {"name": "b"}).name == "b"
