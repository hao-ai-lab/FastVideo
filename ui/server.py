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
import collections
import enum
import logging
import os
import re
import signal
import threading
import time
import uuid
import uvicorn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastvideo.registry import get_registered_model_paths

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fastvideo.ui.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "ui_jobs"
)

def _get_model_label(model_path: str) -> str:
    """Derive a readable label from a HF model path."""
    return model_path.split("/")[-1].replace("-", " ").replace("_", " ")

_available_models: list[dict[str, str]] = [
    {"id": path, "label": _get_model_label(path)} for path in get_registered_model_paths()
]

# ---------------------------------------------------------------------------
# Per-job log buffer & progress tracker
# ---------------------------------------------------------------------------

# Regex patterns for parsing tqdm-style progress output.
# Matches e.g. " 40%|████      | 20/50 " or " 20/50 "
_TQDM_PCT_RE = re.compile(r"(\d+)%\|")
_TQDM_FRAC_RE = re.compile(r"\b(\d+)/(\d+)\b")

_MAX_LOG_LINES = 2000  # ring-buffer cap per job


class JobLogBuffer:

    def __init__(self, maxlen: int = _MAX_LOG_LINES):
        self._lines: collections.deque[str] = collections.deque(
            maxlen=maxlen
        )
        self._lock = threading.Lock()
        self.progress: float = 0.0        # 0 – 100
        self.progress_msg: str = ""       # e.g. "20/50 steps"
        self.phase: str = "initializing"  # human-readable phase

    def write(self, text: str):
        """Append *text* (may contain embedded newlines)."""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            with self._lock:
                self._lines.append(stripped)
            self._parse_progress(stripped)

    def get_lines(self, after: int = 0) -> tuple[list[str], int]:
        """Return ``(lines[after:], total_len)``."""
        with self._lock:
            all_lines = list(self._lines)
        return all_lines[after:], len(all_lines)

    # -- internal helpers ---------------------------------------------------

    def _parse_progress(self, line: str):
        """Try to extract a percentage / fraction from a tqdm line."""
        m = _TQDM_PCT_RE.search(line)
        if m:
            self.progress = min(float(m.group(1)), 100.0)
        m2 = _TQDM_FRAC_RE.search(line)
        if m2:
            cur, total = int(m2.group(1)), int(m2.group(2))
            if total > 0:
                self.progress = min(cur / total * 100.0, 100.0)
                self.progress_msg = f"{cur}/{total} steps"
        # Detect high-level phases from FastVideo log messages
        low = line.lower()
        if "loading model" in low or "loading" in low and "checkpoint" in low:
            self.phase = "loading model"
        elif "denoising" in low or "timestep" in low:
            self.phase = "denoising"
        elif "saving" in low or "saved" in low:
            self.phase = "saving"
        elif "encoding" in low or "vae" in low:
            self.phase = "VAE encoding"


class LogBufferHandler(logging.Handler):
    """Logging handler that writes to a JobLogBuffer."""
    
    def __init__(self, buffer: JobLogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.write(msg)
        except Exception:
            self.handleError(record)

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
    log_file_path: str | None = None  # Path to the job's log file
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
    # Internal
    _thread: threading.Thread | None = field(
        default=None, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _log_buf: JobLogBuffer = field(
        default_factory=JobLogBuffer, repr=False
    )
    _file_handler: logging.FileHandler | None = field(
        default=None, repr=False
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
            "log_file_path": self.log_file_path,
            "num_inference_steps": self.num_inference_steps,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_gpus": self.num_gpus,
            "dit_cpu_offload": self.dit_cpu_offload,
            "text_encoder_cpu_offload": self.text_encoder_cpu_offload,
            "use_fsdp_inference": self.use_fsdp_inference,
            "progress": self._log_buf.progress,
            "progress_msg": self._log_buf.progress_msg,
            "phase": self._log_buf.phase,
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
    dit_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    use_fsdp_inference: bool = False


# ---------------------------------------------------------------------------
# Job store (in-memory, guarded by a lock)
# ---------------------------------------------------------------------------
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_output_dir: str = DEFAULT_OUTPUT_DIR
_log_dir: str = os.path.join(
    os.path.dirname(__file__), "logs"
)
_verbose: bool = False

# Cache of loaded generators keyed by (model_id, num_gpus, dit_cpu_offload, 
# text_encoder_cpu_offload, use_fsdp_inference) so that we only pay the
# model-loading cost once per model configuration.
_generators: dict[tuple[str, int, bool | None, bool | None, bool | None], Any] = {}
_generators_lock = threading.Lock()


def _get_or_create_generator(
    model_id: str,
    num_gpus: int,
    dit_cpu_offload: bool = False,
    text_encoder_cpu_offload: bool = False,
    use_fsdp_inference: bool = False
) -> Any:
    """Return a cached VideoGenerator, creating one on first use.
    
    Generators are cached by model_id and configuration parameters to ensure
    different configurations get separate generator instances.
    """
    cache_key = (model_id, num_gpus, dit_cpu_offload, text_encoder_cpu_offload, use_fsdp_inference)
    
    with _generators_lock:
        if cache_key in _generators:
            return _generators[cache_key]

    # Import lazily so starting the server is fast even without a GPU.
    from fastvideo import VideoGenerator

    logger.info(
        "Loading model %s (num_gpus=%d, dit_cpu_offload=%s, "
        "text_encoder_cpu_offload=%s, use_fsdp_inference=%s) …",
        model_id, num_gpus, dit_cpu_offload, text_encoder_cpu_offload, use_fsdp_inference
    )
    
    gen = VideoGenerator.from_pretrained(
        model_id,
        dit_cpu_offload=dit_cpu_offload,
        text_encoder_cpu_offload=text_encoder_cpu_offload,
        use_fsdp_inference=use_fsdp_inference
    )
    
    with _generators_lock:
        if cache_key not in _generators:
            _generators[cache_key] = gen
        else: # Another thread may have created it while we were loading.
            gen.shutdown()
            gen = _generators[cache_key]
    return gen


# ---------------------------------------------------------------------------
# Worker — runs a single job in a background thread
# ---------------------------------------------------------------------------


def _run_job(job: Job):
    # Create log buffer
    buf = job._log_buf
    os.makedirs(_log_dir, exist_ok=True)
    job.log_file_path = os.path.join(_log_dir, f"{job.id}.log")

    # Add file handler to persist logs
    file_handler = logging.FileHandler(job.log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    job._file_handler = file_handler

    # Hook logger output into job log buffer    
    root_logger = logging.getLogger()
    buffer_handler = LogBufferHandler(buf)
    root_logger.addHandler(buffer_handler)

    # Set output directory, create if it doesn't exist
    job_output_dir = os.path.join(_output_dir, job.id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        buf.phase = "starting"
        buf.write(f"Job {job.id} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at))}")
        buf.write(f"Model: {job.model_id}")
        buf.write(f"Prompt: {job.prompt}")

        if job._stop_event.is_set():
            job.status = JobStatus.STOPPED
            job.finished_at = time.time()
            buf.write("Job stopped before execution started")
            return

        buf.phase = "loading model"
        buf.write("Loading model...")
        
        generator = _get_or_create_generator(
            job.model_id,
            job.num_gpus,
            dit_cpu_offload=job.dit_cpu_offload,
            text_encoder_cpu_offload=job.text_encoder_cpu_offload,
            use_fsdp_inference=job.use_fsdp_inference,
        )
        buf.phase = "generating"
        buf.write(
            f"Starting generation for job {job.id} (model={job.model_id})"
        )
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
    
        buf.phase = "saving"
        buf.write("Generation completed, searching for output file...")

        # Find the generated video file
        video_files = sorted(Path(job_output_dir).glob("*.mp4"))
        if video_files:
            job.output_path = str(video_files[0])
            buf.write(f"Found video output: {job.output_path}")
        else:
            # Could be an image workload
            image_files = sorted(Path(job_output_dir).glob("*.png"))
            if image_files:
                job.output_path = str(image_files[0])
                buf.write(f"Found image output: {job.output_path}")
            else:
                buf.write("Warning: No output file found in job directory")

        if job._stop_event.is_set():
            job.status = JobStatus.STOPPED
            buf.write("Job was stopped during execution")
        else:
            job.status = JobStatus.COMPLETED
            buf.progress = 100.0
            buf.write("Job completed successfully")
        job.finished_at = time.time()
        buf.phase = "done"
        logger.info("Job %s completed successfully", job.id)

    except Exception as exception:
        error_msg = str(exception) if _verbose else str(exception).split('\n')[0]
        buf.write(f"Critical error in job thread: {error_msg}")
        job.status = JobStatus.FAILED
        job.error = f"Critical error ({type(exception).__name__}): {error_msg}"
        job.finished_at = time.time()
        buf.phase = "failed"

    finally:
        # Save log file
        file_handler.flush()
        file_handler.close()
        root_logger.removeHandler(buffer_handler)
        job._file_handler = None


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
    valid_ids = {m["id"] for m in _available_models}
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
        dit_cpu_offload=req.dit_cpu_offload,
        text_encoder_cpu_offload=req.text_encoder_cpu_offload,
        use_fsdp_inference=req.use_fsdp_inference,
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
    job.log_file_path = None  # Will be set when job starts
    job.started_at = None
    job.finished_at = None
    job._stop_event.clear()
    job._log_buf = JobLogBuffer()  # fresh log buffer
    job._file_handler = None  # Reset file handler

    # Wrap _run_job in an additional safety layer to catch any exceptions
    # that might escape (though they shouldn't with our comprehensive handling)
    def safe_run_job(job: Job):
        """Wrapper to ensure _run_job never raises an unhandled exception."""
        try:
            _run_job(job)
        except BaseException as exc:
            # This should never happen, but if it does, we catch it here
            logger.critical(
                "Unhandled exception escaped from _run_job for job %s: %s",
                job.id, exc, exc_info=True
            )
            try:
                job.status = JobStatus.FAILED
                job.error = f"Unhandled exception: {type(exc).__name__}: {str(exc)}"
                job.finished_at = time.time()
            except Exception:
                pass  # If even setting status fails, at least we logged it
    
    thread = threading.Thread(
        target=safe_run_job, args=(job,), daemon=True
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


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
    """Return log lines for a job.

    Query params:
        after: return only lines after this index (for incremental polling).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    lines, total = job._log_buf.get_lines(after=after)
    return {
        "lines": lines,
        "total": total,
        "progress": job._log_buf.progress,
        "progress_msg": job._log_buf.progress_msg,
        "phase": job._log_buf.phase,
    }


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


@app.get("/api/jobs/{job_id}/download_log")
def get_job_log_file(job_id: str) -> FileResponse:
    """Download the log file for a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
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
    
    signal.signal(signal.SIGQUIT, handle_sigquit)
    if hasattr(signal, "SIGQUIT"): # SIGQUIT might not be available on all platforms (e.g., Windows)
        signal.signal(signal.SIGTERM, handle_sigterm)


def main():
    global _output_dir, _log_dir, _verbose  # noqa: PLW0603

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

    _verbose = args.verbose

    _output_dir = os.path.abspath(args.output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    logger.info("Output directory: %s", _output_dir)

    _log_dir = os.path.abspath(args.log_dir)
    os.makedirs(_log_dir, exist_ok=True)
    logger.info("Log directory: %s", _log_dir)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
