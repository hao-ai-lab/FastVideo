# SPDX-License-Identifier: Apache-2.0
"""Studio records and serves the artifact actually saved by the generator."""

import io
from pathlib import Path
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo_studio import server
from fastvideo_studio.database import Database
from fastvideo_studio.job_runner import JobRunner, JobStatus


@pytest.fixture
def runner(tmp_path):
    instance = JobRunner(
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        database=Database(tmp_path / "studio.db"),
    )
    try:
        yield instance
    finally:
        instance._shutdown()


@pytest.mark.parametrize("workload, name, output_name, media_type", [
    ("t2i", "MPS handoff test", "MPS handoff test.png", "image/png"),
    ("t2i", "portrait.jpg", "portrait.png", "image/png"),
    ("t2i", "portrait.jpeg", "portrait.png", "image/png"),
    ("t2i", "portrait.webp", "portrait.png", "image/png"),
    ("t2v", "MPS handoff test", "MPS handoff test.mp4", "video/mp4"),
])
def test_named_job_serves_generated_artifact(runner, monkeypatch, workload, name, output_name, media_type):
    job = runner.create_job(
        job_id=str(uuid.uuid4()),
        model_id="test/model",
        name=name,
        prompt="a red dinosaur",
        workload_type=workload,
    )
    output_dir = Path(runner.output_dir) / job.id
    output_dir.mkdir()
    # Old artifacts and directories with media suffixes must not win discovery.
    (output_dir / "000-old.mp4").mkdir()
    (output_dir / "001-old.mp4").write_bytes(b"old video")
    path_context = SimpleNamespace(
        _is_image_workload=lambda: workload == "t2i",
        _is_audio_workload=lambda: False,
    )

    def generate_video(**kwargs):
        # Use the real filename/extension selection, replacing only model work.
        output = Path(VideoGenerator._prepare_output_path(
            path_context,
            kwargs["output_path"],
            kwargs["prompt"],
            output_video_name=kwargs.get("output_video_name"),
        ))
        if workload == "t2i":
            Image.new("RGB", (8, 8), "red").save(output)
        else:
            output.write_bytes(b"generated video fixture")
        return {"video_path": str(output)}

    monkeypatch.setattr(runner, "_get_or_create_generator",
                        lambda *args, **kwargs: SimpleNamespace(generate_video=generate_video))
    runner._run_inference_job(job)

    expected = output_dir / output_name
    assert job.status == JobStatus.COMPLETED, job.error
    assert job.output_path == str(expected)
    assert expected.is_file()
    assert runner._db.get_job(job.id)["output_path"] == str(expected)
    monkeypatch.setattr(server, "job_runner", runner, raising=False)
    with TestClient(server.app) as client:
        response = client.get(f"/api/jobs/{job.id}/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert response.content == expected.read_bytes()
    if workload == "t2i":
        with Image.open(io.BytesIO(response.content)) as image:
            image.load()
            assert image.format == "PNG"
            assert image.size == (8, 8)
            assert image.getpixel((0, 0)) == (255, 0, 0)


def test_directory_returned_as_output_fails_job(runner, monkeypatch):
    job = runner.create_job(job_id=str(uuid.uuid4()), model_id="test/model", prompt="test")
    directory = Path(runner.output_dir) / job.id / "not-a-video.mp4"
    directory.mkdir(parents=True)
    generator = SimpleNamespace(generate_video=lambda **kwargs: {"video_path": str(directory)})
    monkeypatch.setattr(runner, "_get_or_create_generator", lambda *args, **kwargs: generator)

    runner._run_inference_job(job)

    assert job.status == JobStatus.FAILED
    assert job.output_path is None
    assert job.error
