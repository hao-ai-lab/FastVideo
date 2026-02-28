# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Job Runner — API server only.

Provides REST endpoints for creating, starting, stopping, and deleting
video-generation jobs powered by the FastVideo library.

This is the API-only server. Use web_server.py to serve the frontend.

Usage:
    python -m ui.api_server                       # from repo root
    python ui/api_server.py --output-dir ./videos # explicit output dir
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import uuid
import uvicorn
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastvideo.registry import get_registered_model_paths
from ui.job_runner import JobRunner, JobStatus


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fastvideo.ui.api")

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "ui_jobs"
)

def _get_model_label(model_path: str) -> str:
    """Derive a readable label from a HF model path."""
    return model_path.split("/")[-1].replace("-", " ").replace("_", " ")

_available_models: list[dict[str, str]] = [
    {"id": path, "label": _get_model_label(path)} for path in get_registered_model_paths()
]

job_runner: JobRunner


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
    dit_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    use_fsdp_inference: bool = False


app = FastAPI(
    title="FastVideo Job Runner API",
    version="0.1.0",
    description="REST API for FastVideo job management",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models() -> list[dict[str, str]]:
    """Return the catalogue of available video-generation models."""
    return _available_models


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """Return every job (newest first)."""
    return [j.to_dict() for j in job_runner.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Get details for a single job."""
    job = job_runner.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/jobs", status_code=201)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    """Create a new job (does **not** start it automatically)."""
    valid_ids = {m["id"] for m in _available_models}
    if req.model_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_id '{req.model_id}'. "
                f"Valid options: {sorted(valid_ids)}"
            ),
        )

    job = job_runner.create_job(
        job_id=str(uuid.uuid4()),
        model_id=req.model_id,
        prompt=req.prompt,
        num_inference_steps=req.num_inference_steps,
        num_frames=req.num_frames,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        num_gpus=req.num_gpus,
        dit_cpu_offload=req.dit_cpu_offload,
        text_encoder_cpu_offload=req.text_encoder_cpu_offload,
        use_fsdp_inference=req.use_fsdp_inference,
    )
    return job.to_dict()


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str) -> dict[str, Any]:
    """Start (or restart) a pending / stopped / failed job."""
    try:
        job = job_runner.start_job(job_id)
        return job.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 409, detail=str(e))


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    """Request a running job to stop.

    Because video generation is a single atomic call to the FastVideo
    library, the stop is *cooperative*: the flag is checked between major
    phases (model loading ↔ generation ↔ saving).  If the model is
    already mid-forward-pass, it will complete before the stop takes
    effect.
    """
    try:
        job = job_runner.stop_job(job_id)
        return job.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e) else 409, detail=str(e))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job.  Running jobs are stopped first."""
    if not job_runner.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"detail": f"Job {job_id} deleted"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
    """Return log lines for a job.

    Query params:
        after: return only lines after this index (for incremental polling).
    """
    try:
        return job_runner.get_job_logs(job_id, after=after)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """Stream the generated video/image for a completed job."""
    job = job_runner.get_job(job_id)
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


@app.get("/api/jobs/{job_id}/download_log")
def get_job_log_file(job_id: str) -> FileResponse:
    """Download the log file for a job."""
    job = job_runner.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.log_file_path:
        raise HTTPException(
            status_code=404, detail="Log file not available for this job"
        )
    if not os.path.isfile(job.log_file_path):
        raise HTTPException(
            status_code=404, detail="Log file not found on disk"
        )

    return FileResponse(
        job.log_file_path,
        media_type="text/plain",
        filename=f"job_{job_id}.log"
    )


def _setup_signal_handlers():
    def handle_sigquit(signum, frame):
        logger.warning(
            "Received SIGQUIT (likely from a crashed worker process). "
            "Ignoring to keep server running."
        )
    
    def handle_sigterm(signum, frame):
        logger.info("Received SIGTERM. Shutting down gracefully...")
        raise SystemExit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    if hasattr(signal, "SIGQUIT"): # SIGQUIT might not be available on all platforms (e.g., Windows)
        signal.signal(signal.SIGQUIT, handle_sigquit)


def create_local_env(host: str, port: int) -> None:
    """Check if .env.local exists in the ui directory, and create it if not."""
    ui_dir = os.path.dirname(__file__)
    env_local_path = os.path.join(ui_dir, ".env.local")
    
    # Use localhost for the API URL since browsers can't connect to 0.0.0.0
    api_host = "localhost" if host == "0.0.0.0" else host
    api_url = f"http://{api_host}:{port}/api"
    
    if not os.path.exists(env_local_path):
        logger.info(f"Creating .env.local with API URL: {api_url}")
        with open(env_local_path, "w", encoding="utf-8") as f:
            f.write(f"NEXT_PUBLIC_API_BASE_URL={api_url}\n")
    else:
        logger.debug(f".env.local already exists at {env_local_path}")


def main():
    global job_runner  # noqa: PLW0603

    # Set up signal handlers to prevent worker crashes from killing the server
    _setup_signal_handlers()

    default_log_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "ui_logs"
    )

    parser = argparse.ArgumentParser(
        description="FastVideo Job Runner API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8189,
        help="Port number (default: 8189)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where generated videos are saved "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=default_log_dir,
        help=(
            "Directory where job log files are saved "
            f"(default: {default_log_dir})"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks in error messages (default: False)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    log_dir = os.path.abspath(args.log_dir)
    
    create_local_env(args.host, args.port)
    
    # Initialize job runner
    job_runner = JobRunner(
        output_dir=output_dir,
        log_dir=log_dir,
        verbose=args.verbose
    )
    
    logger.info("Output directory: %s", output_dir)
    logger.info("Log directory: %s", log_dir)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
