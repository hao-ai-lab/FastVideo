# SPDX-License-Identifier: Apache-2.0
"""
Job Runner for FastVideo video generation jobs.

Manages job lifecycle, execution, logging, and generator caching.
"""

from __future__ import annotations

import collections
import enum
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("fastvideo.ui.job_runner")

# Regex patterns for parsing tqdm-style progress output.
# Matches e.g. " 40%|████      | 20/50 " or " 20/50 "
_TQDM_PCT_RE = re.compile(r"(\d+)%\|")
_TQDM_FRAC_RE = re.compile(r"\b(\d+)/(\d+)\b")

_MAX_LOG_LINES = 2000  # ring-buffer cap per job


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class JobLogBuffer:
    """Ring buffer for storing job log lines with progress tracking."""

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
    """Logging handler that writes to a JobLogBuffer.
    
    Only processes log records from the thread that created this handler.
    """
    
    def __init__(self, buffer: JobLogBuffer):
        super().__init__()
        self.buffer = buffer
        self.thread_id = threading.get_ident()
        self.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s"))
    
    def filter(self, record):
        """Only process logs from the thread that created this handler."""
        return threading.get_ident() == self.thread_id
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.write(msg)
        except Exception:
            self.handleError(record)


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
    log_file_handler: logging.FileHandler | None = field(
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


class JobRunner:
    """Manages video generation jobs, their execution, and generator caching."""
    
    def __init__(
        self,
        output_dir: str,
        log_dir: str,
        verbose: bool = False
    ):
        """Initialize the job runner.
        
        Args:
            output_dir: Directory where generated videos are saved
            log_dir: Directory where job log files are saved
            verbose: Whether to print full tracebacks in error messages
        """
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.verbose = verbose
        
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        
        # Cache of loaded generators keyed by (model_id, num_gpus, dit_cpu_offload, 
        # text_encoder_cpu_offload, use_fsdp_inference) so that we only pay the
        # model-loading cost once per model configuration.
        self._generators: dict[tuple[str, int, bool | None, bool | None, bool | None], Any] = {}
        self._generators_lock = threading.Lock()
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_job(
        self,
        job_id: str,
        model_id: str,
        prompt: str,
        num_inference_steps: int = 50,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        guidance_scale: float = 5.0,
        seed: int = 1024,
        num_gpus: int = 1,
        dit_cpu_offload: bool = False,
        text_encoder_cpu_offload: bool = False,
        use_fsdp_inference: bool = False,
    ) -> Job:
        """Create a new job (does not start it automatically)."""
        job = Job(
            id=job_id,
            model_id=model_id,
            prompt=prompt.strip(),
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=seed,
            num_gpus=num_gpus,
            dit_cpu_offload=dit_cpu_offload,
            text_encoder_cpu_offload=text_encoder_cpu_offload,
            use_fsdp_inference=use_fsdp_inference,
        )
        with self._jobs_lock:
            self._jobs[job.id] = job
        logger.info(
            "Created job %s (model=%s, prompt=%s…)",
            job.id,
            job.model_id,
            job.prompt[:60],
        )
        return job
    
    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        with self._jobs_lock:
            return self._jobs.get(job_id)
    
    def list_jobs(self) -> list[Job]:
        """Return all jobs, sorted by creation time (newest first)."""
        with self._jobs_lock:
            return sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job. Running jobs are stopped first.
        
        Returns:
            True if job was found and deleted, False otherwise
        """
        with self._jobs_lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        
        # Best-effort stop
        job._stop_event.set()
        logger.info("Deleted job %s", job.id)
        return True
    
    def start_job(self, job_id: str) -> Job:
        """Start (or restart) a pending / stopped / failed job.
        
        Raises:
            ValueError: If job not found or cannot be started
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        if job.status == JobStatus.RUNNING:
            raise ValueError("Job is already running")
        if job.status == JobStatus.COMPLETED:
            raise ValueError(
                "Job already completed. Delete and re-create to run again."
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
        job.log_file_handler = None  # Reset file handler

        # Wrap _run_job in an additional safety layer to catch any exceptions
        # that might escape (though they shouldn't with our comprehensive handling)
        def safe_run_job(job: Job):
            """Wrapper to ensure _run_job never raises an unhandled exception."""
            try:
                self._run_job(job)
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
        return job
    
    def stop_job(self, job_id: str) -> Job:
        """Request a running job to stop.
        
        Because video generation is a single atomic call to the FastVideo
        library, the stop is *cooperative*: the flag is checked between major
        phases (model loading ↔ generation ↔ saving).  If the model is
        already mid-forward-pass, it will complete before the stop takes
        effect.
        
        Raises:
            ValueError: If job not found or not running
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        if job.status != JobStatus.RUNNING:
            raise ValueError(f"Job is not running (status={job.status.value})")

        job._stop_event.set()
        logger.info("Stop requested for job %s", job.id)
        return job
    
    def get_job_logs(self, job_id: str, after: int = 0) -> dict[str, Any]:
        """Return log lines for a job.
        
        Args:
            job_id: The job ID
            after: Return only lines after this index (for incremental polling)
            
        Returns:
            Dictionary with 'lines', 'total', 'progress', 'progress_msg', 'phase'
            
        Raises:
            ValueError: If job not found
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        lines, total = job._log_buf.get_lines(after=after)
        return {
            "lines": lines,
            "total": total,
            "progress": job._log_buf.progress,
            "progress_msg": job._log_buf.progress_msg,
            "phase": job._log_buf.phase,
        }
    
    def _get_or_create_generator(
        self,
        model_id: str,
        num_gpus: int,
        dit_cpu_offload: bool = False,
        text_encoder_cpu_offload: bool = False,
        use_fsdp_inference: bool = False
    ) -> Any:
        cache_key = (model_id, num_gpus, dit_cpu_offload, text_encoder_cpu_offload, use_fsdp_inference)
        
        # Generators are cached by model_id and configuration parameters
        with self._generators_lock:
            if cache_key in self._generators:
                return self._generators[cache_key]

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
        
        with self._generators_lock:
            if cache_key not in self._generators:
                self._generators[cache_key] = gen
            else: # Another thread may have created it while we were loading.
                gen.shutdown()
                gen = self._generators[cache_key]
        return gen

    def _run_job(self, job: Job):
        buf = job._log_buf
        os.makedirs(self.log_dir, exist_ok=True)
        job.log_file_path = os.path.join(self.log_dir, f"{job.id}.log")

        # Add file handler to persist logs
        file_handler = logging.FileHandler(job.log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        job.log_file_handler = file_handler

        # Hook logger output into job log buffer
        fastvideo_logger = logging.getLogger("fastvideo")
        buffer_handler = LogBufferHandler(buf)
        fastvideo_logger.addHandler(buffer_handler)
        fastvideo_logger.addHandler(file_handler)

        # Set output directory, create if it doesn't exist
        job_output_dir = os.path.join(self.output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)
        
        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            buf.phase = "starting"
            logger.info(f"Job {job.id} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at))}")
            logger.info(f"Model: {job.model_id}")
            logger.info(f"Prompt: {job.prompt}")

            if job._stop_event.is_set():
                job.status = JobStatus.STOPPED
                job.finished_at = time.time()
                logger.warning("Job stopped before execution started")
                return

            buf.phase = "loading model"
            logger.info("Loading model...")
            
            generator = self._get_or_create_generator(
                job.model_id,
                job.num_gpus,
                dit_cpu_offload=job.dit_cpu_offload,
                text_encoder_cpu_offload=job.text_encoder_cpu_offload,
                use_fsdp_inference=job.use_fsdp_inference,
            )
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
            logger.info("Generation completed, searching for output file...")

            # Find the generated video file
            video_files = sorted(Path(job_output_dir).glob("*.mp4"))
            if video_files:
                job.output_path = str(video_files[0])
                logger.info(f"Found video output: {job.output_path}")
            else:
                # Could be an image workload
                image_files = sorted(Path(job_output_dir).glob("*.png"))
                if image_files:
                    job.output_path = str(image_files[0])
                    logger.info(f"Found image output: {job.output_path}")
                else:
                    logger.warning("No output file found in job directory")

            if job._stop_event.is_set():
                job.status = JobStatus.STOPPED
                logger.warning("Job was stopped during execution")
            else:
                job.status = JobStatus.COMPLETED
                buf.progress = 100.0
                logger.info("Job completed successfully")
            job.finished_at = time.time()
            buf.phase = "done"

        except Exception as exception:
            error_msg = str(exception)
            logger.error(f"Critical error in job thread: {error_msg}")
            job.status = JobStatus.FAILED
            job.error = f"Critical error ({type(exception).__name__}): {error_msg}"
            job.finished_at = time.time()
            buf.phase = "failed"

        finally:
            # Remove handlers and close file
            fastvideo_logger.removeHandler(buffer_handler)
            fastvideo_logger.removeHandler(file_handler)
            file_handler.flush()
            file_handler.close()
            job.log_file_handler = None
