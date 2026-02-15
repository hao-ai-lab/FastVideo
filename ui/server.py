# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Job Runner — lightweight web UI backend.

Provides REST endpoints for creating, starting, stopping, and deleting
video-generation jobs powered by the FastVideo library.

Usage:
    python -m ui.server                       # from repo root
    python ui/server.py --output-dir ./videos # explicit output dir
"""

from __future__ import annotations

import argparse
import enum
import logging
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fastvideo.ui")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "ui_jobs"
)

# Well-known text-to-video models supported by FastVideo.
AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "label": "Wan 2.1 T2V 1.3B",
        "type": "t2v",
    },
    {
        "id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "label": "Wan 2.2 T2V A14B",
        "type": "t2v",
    },
    {
        "id": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "label": "FastWan 2.1 T2V 1.3B (Distilled)",
        "type": "t2v",
    },
    {
        "id": "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        "label": "FastWan 2.2 TI2V 5B (Distilled)",
        "type": "t2v",
    },
    {
        "id": "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers",
        "label": "TurboWan 2.1 T2V 1.3B",
        "type": "t2v",
    },
    {
        "id": "loayrashid/TurboWan2.1-T2V-14B-Diffusers",
        "label": "TurboWan 2.1 T2V 14B",
        "type": "t2v",
    },
    {
        "id": "FastVideo/LTX2-Distilled-Diffusers",
        "label": "LTX2 Distilled",
        "type": "t2v",
    },
    {
        "id": "Davids048/LTX2-Base-Diffusers",
        "label": "LTX2 Base",
        "type": "t2v",
    },
    {
        "id": (
            "hunyuanvideo-community/"
            "HunyuanVideo-1.5-Diffusers-480p_t2v"
        ),
        "label": "HunyuanVideo 1.5 480p",
        "type": "t2v",
    },
    {
        "id": "FastVideo/LongCat-Video-T2V-Diffusers",
        "label": "LongCat Video T2V",
        "type": "t2v",
    },
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Job:
    id: str
    model_id: str
    prompt: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_path: str | None = None
    num_inference_steps: int = 50
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 5.0
    seed: int = 1024
    num_gpus: int = 1
    # Internal
    _thread: threading.Thread | None = field(
        default=None, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "output_path": self.output_path,
            "num_inference_steps": self.num_inference_steps,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_gpus": self.num_gpus,
        }


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    model_id: str
    prompt: str
    num_inference_steps: int = 50
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 5.0
    seed: int = 1024
    num_gpus: int = 1


class JobResponse(BaseModel):
    id: str
    model_id: str
    prompt: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_path: str | None = None
    num_inference_steps: int = 50
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 5.0
    seed: int = 1024
    num_gpus: int = 1


# ---------------------------------------------------------------------------
# Job store (in-memory, guarded by a lock)
# ---------------------------------------------------------------------------
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_output_dir: str = DEFAULT_OUTPUT_DIR

# Cache of loaded generators keyed by model_id so that we only pay the
# model-loading cost once per model.
_generators: dict[str, Any] = {}
_generators_lock = threading.Lock()


def _get_or_create_generator(
    model_id: str, num_gpus: int
) -> Any:
    """Return a cached VideoGenerator, creating one on first use."""
    with _generators_lock:
        if model_id in _generators:
            return _generators[model_id]

    # Import lazily so starting the server is fast even without a GPU.
    from fastvideo import VideoGenerator

    logger.info("Loading model %s (num_gpus=%d) …", model_id, num_gpus)
    gen = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=num_gpus,
    )
    with _generators_lock:
        # Another thread may have created it while we were loading.
        if model_id not in _generators:
            _generators[model_id] = gen
        else:
            gen.shutdown()
            gen = _generators[model_id]
    return gen


# ---------------------------------------------------------------------------
# Worker — runs a single job in a background thread
# ---------------------------------------------------------------------------


def _run_job(job: Job) -> None:
    """Execute video generation for *job* (called in a daemon thread)."""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        if job._stop_event.is_set():
            job.status = JobStatus.STOPPED
            job.finished_at = time.time()
            return

        generator = _get_or_create_generator(
            job.model_id, job.num_gpus
        )

        job_output_dir = os.path.join(_output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)

        logger.info(
            "Starting generation for job %s (model=%s)", job.id, job.model_id
        )

        generator.generate_video(
            prompt=job.prompt,
            output_path=job_output_dir,
            save_video=True,
            num_inference_steps=job.num_inference_steps,
            num_frames=job.num_frames,
            height=job.height,
            width=job.width,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
        )

        # Find the generated video file
        video_files = sorted(Path(job_output_dir).glob("*.mp4"))
        if video_files:
            job.output_path = str(video_files[0])
        else:
            # Could be an image workload
            image_files = sorted(Path(job_output_dir).glob("*.png"))
            if image_files:
                job.output_path = str(image_files[0])

        if job._stop_event.is_set():
            job.status = JobStatus.STOPPED
        else:
            job.status = JobStatus.COMPLETED
        job.finished_at = time.time()
        logger.info("Job %s completed successfully", job.id)

    except Exception as exc:
        logger.exception("Job %s failed", job.id)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.finished_at = time.time()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FastVideo Job Runner",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Models ---------------------------------------------------------------


@app.get("/api/models")
def list_models() -> list[dict[str, str]]:
    """Return the catalogue of available video-generation models."""
    return AVAILABLE_MODELS


# ---- Jobs CRUD ------------------------------------------------------------


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """Return every job (newest first)."""
    with _jobs_lock:
        jobs = sorted(
            _jobs.values(), key=lambda j: j.created_at, reverse=True
        )
        return [j.to_dict() for j in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Get details for a single job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/jobs", status_code=201)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    """Create a new job (does **not** start it automatically)."""
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if req.model_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_id '{req.model_id}'. "
                f"Valid options: {sorted(valid_ids)}"
            ),
        )

    job = Job(
        id=str(uuid.uuid4()),
        model_id=req.model_id,
        prompt=req.prompt.strip(),
        num_inference_steps=req.num_inference_steps,
        num_frames=req.num_frames,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        num_gpus=req.num_gpus,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    logger.info(
        "Created job %s (model=%s, prompt=%s…)",
        job.id,
        job.model_id,
        job.prompt[:60],
    )
    return job.to_dict()


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str) -> dict[str, Any]:
    """Start (or restart) a pending / stopped / failed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=409, detail="Job is already running"
        )
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="Job already completed. Delete and re-create to run again.",
        )

    # Reset state for re-run
    job.status = JobStatus.PENDING
    job.error = None
    job.output_path = None
    job.started_at = None
    job.finished_at = None
    job._stop_event.clear()

    thread = threading.Thread(
        target=_run_job, args=(job,), daemon=True
    )
    job._thread = thread
    thread.start()
    logger.info("Started job %s", job.id)
    return job.to_dict()


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    """Request a running job to stop.

    Because video generation is a single atomic call to the FastVideo
    library, the stop is *cooperative*: the flag is checked between major
    phases (model loading ↔ generation ↔ saving).  If the model is
    already mid-forward-pass, it will complete before the stop takes
    effect.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not running (status={job.status.value})",
        )

    job._stop_event.set()
    logger.info("Stop requested for job %s", job.id)
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job.  Running jobs are stopped first."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Best-effort stop
    job._stop_event.set()
    logger.info("Deleted job %s", job.id)
    return {"detail": f"Job {job_id} deleted"}


# ---- Serve generated videos -----------------------------------------------


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """Stream the generated video/image for a completed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED or not job.output_path:
        raise HTTPException(
            status_code=404, detail="No output available for this job"
        )
    if not os.path.isfile(job.output_path):
        raise HTTPException(
            status_code=404, detail="Output file not found on disk"
        )

    media_type = (
        "video/mp4"
        if job.output_path.endswith(".mp4")
        else "image/png"
    )
    return FileResponse(job.output_path, media_type=media_type)


# ---- Static frontend -------------------------------------------------------
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount(
        "/",
        StaticFiles(directory=_static_dir, html=True),
        name="static",
    )

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastVideo Job Runner web UI"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8188,
        help="Port number (default: 8188)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where generated videos are saved "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    args = parser.parse_args()

    global _output_dir  # noqa: PLW0603
    _output_dir = os.path.abspath(args.output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    logger.info("Output directory: %s", _output_dir)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
