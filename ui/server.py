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
import collections
import enum
import io
import logging
import os
import re
import signal
import sys
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

# ---------------------------------------------------------------------------
# Dynamic model catalogue — built from fastvideo.registry
# ---------------------------------------------------------------------------

_WORKLOAD_PATTERNS: list[tuple[str, str]] = [
    ("v2v", "v2v"),
    ("v2w", "v2w"),
    ("i2w", "i2w"),
    ("t2w", "t2w"),
    ("t2i", "t2i"),
    ("i2v", "i2v"),
    ("ti2v", "ti2v"),
    ("t2v", "t2v"),
]


def _infer_workload_type(model_path: str) -> str:
    """Best-effort workload type from the HF model path."""
    low = model_path.lower()
    for token, wtype in _WORKLOAD_PATTERNS:
        if token in low:
            return wtype
    return "t2v"  # default


def _make_label(model_path: str) -> str:
    """Derive a readable label from a HF model path."""
    name = model_path.split("/")[-1] if "/" in model_path else model_path
    # Strip common suffixes that add noise
    for suffix in ("-Diffusers", "_Diffusers", "-diffusers"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Replace separators with spaces
    return name.replace("-", " ").replace("_", " ")


def _get_available_models() -> list[dict[str, str]]:
    """Build the model catalogue from the fastvideo registry."""
    from fastvideo.registry import get_registered_model_paths

    models: list[dict[str, str]] = []
    for path in get_registered_model_paths():
        models.append({
            "id": path,
            "label": _make_label(path),
            "type": _infer_workload_type(path),
        })
    return models


# Lazily cached so the import only happens on first access.
_available_models_cache: list[dict[str, str]] | None = None


def _available_models() -> list[dict[str, str]]:
    global _available_models_cache  # noqa: PLW0603
    if _available_models_cache is None:
        _available_models_cache = _get_available_models()
    return _available_models_cache

# ---------------------------------------------------------------------------
# Per-job log buffer & progress tracker
# ---------------------------------------------------------------------------

# Regex patterns for parsing tqdm-style progress output.
# Matches e.g. " 40%|████      | 20/50 " or " 20/50 "
_TQDM_PCT_RE = re.compile(r"(\d+)%\|")
_TQDM_FRAC_RE = re.compile(r"\b(\d+)/(\d+)\b")

_MAX_LOG_LINES = 2000  # ring-buffer cap per job


class JobLogBuffer:
    """Thread-safe ring-buffer that stores log lines for a single job.

    It also continuously parses tqdm progress output so the UI can
    display a determinate progress bar.
    """

    def __init__(self, maxlen: int = _MAX_LOG_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(
            maxlen=maxlen
        )
        self._lock = threading.Lock()
        self.progress: float = 0.0        # 0 – 100
        self.progress_msg: str = ""       # e.g. "20/50 steps"
        self.phase: str = "initializing"  # human-readable phase

    def write(self, text: str) -> None:
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

    def _parse_progress(self, line: str) -> None:
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


class _JobLogHandler(logging.Handler):
    """A logging.Handler that forwards records to a JobLogBuffer."""

    def __init__(self, buf: JobLogBuffer) -> None:
        super().__init__()
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.write(self.format(record))
        except Exception:
            pass


class _TqdmCapture(io.StringIO):
    """Intercepts stderr writes (where tqdm prints) for a job thread."""

    def __init__(self, buf: JobLogBuffer, original: io.TextIOBase) -> None:
        super().__init__()
        self.buf = buf
        self.original = original

    def write(self, s: str) -> int:  # type: ignore[override]
        self.buf.write(s)
        return self.original.write(s)

    def flush(self) -> None:
        self.original.flush()


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
    _log_buf: JobLogBuffer = field(
        default_factory=JobLogBuffer, repr=False
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
    buf = job._log_buf

    # ── Attach a logging handler so all fastvideo.* logs go to the
    #    job's buffer while this thread is running. ──
    log_handler = _JobLogHandler(buf)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()  # capture everything
    root_logger.addHandler(log_handler)

    # ── Redirect stderr so tqdm output is captured too. ──
    original_stderr = sys.stderr
    sys.stderr = _TqdmCapture(buf, original_stderr)  # type: ignore[assignment]

    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        buf.phase = "starting"

        if job._stop_event.is_set():
            job.status = JobStatus.STOPPED
            job.finished_at = time.time()
            return

        buf.phase = "loading model"
        generator = _get_or_create_generator(
            job.model_id, job.num_gpus
        )

        job_output_dir = os.path.join(_output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)

        buf.phase = "generating"
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
            buf.progress = 100.0
        job.finished_at = time.time()
        buf.phase = "done"
        logger.info("Job %s completed successfully", job.id)

    except Exception as exc:
        logger.exception("Job %s failed", job.id)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.finished_at = time.time()
        buf.phase = "failed"

    finally:
        # ── Restore stderr and remove the per-job log handler. ──
        sys.stderr = original_stderr  # type: ignore[assignment]
        root_logger.removeHandler(log_handler)


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
    return _available_models()


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
    valid_ids = {m["id"] for m in _available_models()}
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
    job._log_buf = JobLogBuffer()  # fresh log buffer

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


# ---- Job logs --------------------------------------------------------------


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
