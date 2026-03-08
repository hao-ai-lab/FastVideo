# SPDX-License-Identifier: Apache-2.0
"""
Job Runner for FastVideo video generation jobs.

Manages job lifecycle, execution, logging, and generator caching.
"""

from __future__ import annotations

import atexit
import collections
import enum
import logging
import logging.handlers
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from fastvideo.utils import get_mp_context
from ui.database import Database
from ui.training_config import (
    build_training_args,
    get_training_env,
    get_training_module_info,
)

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
        elif "saving" in low or "saved" in low or "checkpoint" in low:
            self.phase = "saving"
        elif "encoding" in low or "vae" in low:
            self.phase = "VAE encoding"
        elif "training" in low or "step" in low or "loss" in low:
            self.phase = "training"


class LogBufferHandler(logging.Handler):
    """Logging handler that writes to a JobLogBuffer."""

    def __init__(self, buffer: JobLogBuffer):
        super().__init__()
        self.buffer = buffer
        self.setFormatter(
            logging.Formatter(
                "%(levelname)s %(asctime)s [%(name)s] %(message)s"
            )
        )

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
    workload_type: str = "t2v"
    job_type: str = "inference"
    image_path: str = ""
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
    guidance_rescale: float = 0.0
    fps: int = 24
    seed: int = 1024
    negative_prompt: str = ""
    num_gpus: int = 1
    dit_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    image_encoder_cpu_offload: bool = False
    use_fsdp_inference: bool = False
    enable_torch_compile: bool = False
    vsa_sparsity: float = 0.0
    tp_size: int = -1
    sp_size: int = -1
    # Training-specific (for finetuning, distillation, LoRA)
    data_path: str = ""
    max_train_steps: int = 1000
    train_batch_size: int = 1
    learning_rate: float = 5e-5
    num_latent_t: int = 20
    validation_dataset_file: str = ""
    lora_rank: int = 32
    ltx2_first_frame_conditioning_p: float | None = None
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
    _process: subprocess.Popen | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "workload_type": self.workload_type,
            "job_type": self.job_type,
            "image_path": self.image_path,
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
            "guidance_rescale": self.guidance_rescale,
            "fps": self.fps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "num_gpus": self.num_gpus,
            "dit_cpu_offload": self.dit_cpu_offload,
            "text_encoder_cpu_offload": self.text_encoder_cpu_offload,
            "vae_cpu_offload": self.vae_cpu_offload,
            "image_encoder_cpu_offload": self.image_encoder_cpu_offload,
            "use_fsdp_inference": self.use_fsdp_inference,
            "enable_torch_compile": self.enable_torch_compile,
            "vsa_sparsity": self.vsa_sparsity,
            "tp_size": self.tp_size,
            "sp_size": self.sp_size,
            "data_path": self.data_path,
            "max_train_steps": self.max_train_steps,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "num_latent_t": self.num_latent_t,
            "num_height": self.height,
            "num_width": self.width,
            "num_frames": self.num_frames,
            "validation_dataset_file": self.validation_dataset_file,
            "lora_rank": self.lora_rank,
            "ltx2_first_frame_conditioning_p": self.ltx2_first_frame_conditioning_p,
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
        database: Database,
        verbose: bool = False,
    ):
        """Initialize the job runner.

        Args:
            output_dir: Directory where generated videos are saved
            log_dir: Directory where job log files are saved
            verbose: Whether to print full tracebacks in error messages
            database: Optional SQLite database for job persistence
        """
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.verbose = verbose
        self._db = database

        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._load_jobs()
        
        # Cache of loaded generators keyed by model config so that we only pay
        # the model-loading cost once per model configuration.
        self._generators: dict[tuple, Any] = {}
        self._generators_lock = threading.Lock()

        # Shared Manager for log queues (avoids spawning a new process per job)
        self._mp_manager = get_mp_context().Manager()
        atexit.register(self._shutdown)

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_logs(self, job: Job) -> None:
        """Populate job's log buffer from its log file if it exists."""
        path = job.log_file_path
        if not path:
            path = os.path.join(self.log_dir, f"{job.id}.log")
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.rstrip("\n\r")
                    if stripped:
                        job._log_buf.write(stripped)
            if not job.log_file_path:
                job.log_file_path = path
        except Exception as exc:
            logger.warning(
                "Failed to load logs from %s for job %s: %s",
                path, job.id, exc,
            )

    def _load_jobs(self) -> None:
        """Load jobs from database."""
        try:
            for row in self._db.get_all_jobs():
                status = row["status"]
                if status == "running":
                    status = JobStatus.FAILED
                    row["error"] = "Server restarted (job was running)"
                    row["finished_at"] = time.time()
                    self._db.update_job(row["id"], {
                        "status": "failed",
                        "error": row["error"],
                        "finished_at": row["finished_at"],
                    })
                elif status == "pending":
                    status = JobStatus.PENDING
                job = Job(
                    id=row["id"],
                    model_id=row["model_id"],
                    prompt=row["prompt"],
                    workload_type=row.get("workload_type", "t2v"),
                    job_type=row.get("job_type", "inference"),
                    image_path=row.get("image_path", "") or "",
                    data_path=row.get("data_path", "") or "",
                    max_train_steps=row.get("max_train_steps", 1000),
                    train_batch_size=row.get("train_batch_size", 1),
                    learning_rate=float(row.get("learning_rate", 5e-5)),
                    num_latent_t=row.get("num_latent_t", 20),
                    validation_dataset_file=row.get("validation_dataset_file", "") or "",
                    lora_rank=row.get("lora_rank", 32),
                    ltx2_first_frame_conditioning_p=row.get("ltx2_first_frame_conditioning_p"),
                    status=JobStatus(status),
                    created_at=row["created_at"],
                    started_at=row.get("started_at"),
                    finished_at=row.get("finished_at"),
                    error=row.get("error"),
                    output_path=row.get("output_path"),
                    log_file_path=row.get("log_file_path"),
                    num_inference_steps=row.get("num_inference_steps", 50),
                    num_frames=row.get("num_frames", 81),
                    height=row.get("height", 480),
                    width=row.get("width", 832),
                    guidance_scale=row.get("guidance_scale", 5.0),
                    guidance_rescale=row.get("guidance_rescale", 0.0),
                    fps=row.get("fps", 24),
                    seed=row.get("seed", 1024),
                    negative_prompt=row.get("negative_prompt", "") or "",
                    num_gpus=row.get("num_gpus", 1),
                    dit_cpu_offload=row.get("dit_cpu_offload", False),
                    text_encoder_cpu_offload=row.get("text_encoder_cpu_offload", False),
                    vae_cpu_offload=row.get("vae_cpu_offload", False),
                    image_encoder_cpu_offload=row.get("image_encoder_cpu_offload", False),
                    use_fsdp_inference=row.get("use_fsdp_inference", False),
                    enable_torch_compile=row.get("enable_torch_compile", False),
                    vsa_sparsity=row.get("vsa_sparsity", 0.0),
                    tp_size=row.get("tp_size", -1),
                    sp_size=row.get("sp_size", -1),
                )
                self._load_logs(job)
                with self._jobs_lock:
                    self._jobs[job.id] = job
            if self._jobs:
                logger.info("Loaded %d jobs from database", len(self._jobs))
        except Exception as exc:
            logger.warning("Failed to load jobs from database: %s", exc)

    def _save_job(self, job: Job) -> None:
        """Persist job to database."""
        try:
            self._db.update_job(job.id, {
                "status": job.status.value,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
                "output_path": job.output_path,
                "log_file_path": job.log_file_path,
            })
        except Exception as exc:
            logger.warning("Failed to persist job %s: %s", job.id, exc)

    def _shutdown(self) -> None:
        """Shutdown the shared multiprocessing manager on exit."""
        try:
            self._mp_manager.shutdown()
        except Exception as exc:
            logger.warning("Failed to shutdown mp manager: %s", exc)

    def create_job(
        self,
        job_id: str,
        model_id: str,
        prompt: str,
        workload_type: str = "t2v",
        job_type: str = "inference",
        image_path: str = "",
        data_path: str = "",
        max_train_steps: int = 1000,
        train_batch_size: int = 1,
        learning_rate: float = 5e-5,
        num_latent_t: int = 20,
        validation_dataset_file: str = "",
        lora_rank: int = 32,
        ltx2_first_frame_conditioning_p: float | None = None,
        num_inference_steps: int = 50,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        fps: int = 24,
        seed: int = 1024,
        num_gpus: int = 1,
        negative_prompt: str = "",
        dit_cpu_offload: bool = False,
        text_encoder_cpu_offload: bool = False,
        vae_cpu_offload: bool = False,
        image_encoder_cpu_offload: bool = False,
        use_fsdp_inference: bool = False,
        enable_torch_compile: bool = False,
        vsa_sparsity: float = 0.0,
        tp_size: int = -1,
        sp_size: int = -1,
    ) -> Job:
        """Create a new job (does not start it automatically)."""
        job = Job(
            id=job_id,
            model_id=model_id,
            prompt=prompt.strip(),
            workload_type=workload_type or "t2v",
            job_type=job_type or "inference",
            image_path=image_path or "",
            data_path=data_path or "",
            max_train_steps=max_train_steps,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            num_latent_t=num_latent_t,
            validation_dataset_file=validation_dataset_file or "",
            lora_rank=lora_rank,
            ltx2_first_frame_conditioning_p=ltx2_first_frame_conditioning_p,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            fps=fps,
            seed=seed,
            negative_prompt=negative_prompt or "",
            num_gpus=num_gpus,
            dit_cpu_offload=dit_cpu_offload,
            text_encoder_cpu_offload=text_encoder_cpu_offload,
            vae_cpu_offload=vae_cpu_offload,
            image_encoder_cpu_offload=image_encoder_cpu_offload,
            use_fsdp_inference=use_fsdp_inference,
            enable_torch_compile=enable_torch_compile,
            vsa_sparsity=vsa_sparsity,
            tp_size=tp_size,
            sp_size=sp_size,
        )
        with self._jobs_lock:
            self._jobs[job.id] = job
        try:
            self._db.insert_job(job.to_dict())
        except Exception as exc:
            logger.warning("Failed to persist new job %s: %s", job.id, exc)
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
    
    def list_jobs(self, job_type: str | None = None) -> list[Job]:
        """Return jobs, sorted by creation time (newest first).
        If job_type is set, filter to that type only."""
        with self._jobs_lock:
            jobs = list(self._jobs.values())
            if job_type:
                jobs = [j for j in jobs if j.job_type == job_type]
            return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
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
        try:
            self._db.delete_job(job_id)
        except Exception as exc:
            logger.warning("Failed to delete job %s from database: %s", job_id, exc)
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
        job._process = None  # Reset subprocess handle

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
        
        For inference: cooperative stop between phases.
        For training: terminates the subprocess.
        
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
        if job._process is not None:
            try:
                job._process.terminate()
            except Exception:
                pass
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
        workload_type: str,
        num_gpus: int,
        dit_cpu_offload: bool = False,
        text_encoder_cpu_offload: bool = False,
        vae_cpu_offload: bool = False,
        image_encoder_cpu_offload: bool = False,
        use_fsdp_inference: bool = False,
        enable_torch_compile: bool = False,
        vsa_sparsity: float = 0.0,
        tp_size: int = -1,
        sp_size: int = -1,
        log_queue: mp.Queue | None = None,
    ) -> Any:
        cache_key = (
            model_id, workload_type, num_gpus, dit_cpu_offload,
            text_encoder_cpu_offload, vae_cpu_offload, image_encoder_cpu_offload,
            use_fsdp_inference, enable_torch_compile, vsa_sparsity, tp_size,
            sp_size,
        )

        # Generators are cached by model_id and configuration parameters
        with self._generators_lock:
            if cache_key in self._generators:
                return self._generators[cache_key]

        # Import lazily so starting the server is fast even without a GPU.
        from fastvideo import VideoGenerator

        logger.info(
            "Loading model %s (workload=%s, num_gpus=%d, offloads: dit=%s te=%s "
            "vae=%s ie=%s, fsdp=%s, torch_compile=%s, vsa_sparsity=%.2f, tp=%d "
            "sp=%d) …",
            model_id, workload_type, num_gpus, dit_cpu_offload,
            text_encoder_cpu_offload,
            vae_cpu_offload, image_encoder_cpu_offload, use_fsdp_inference,
            enable_torch_compile, vsa_sparsity, tp_size, sp_size,
        )

        gen = VideoGenerator.from_pretrained(
            model_id,
            workload_type=workload_type,
            dit_cpu_offload=dit_cpu_offload,
            text_encoder_cpu_offload=text_encoder_cpu_offload,
            vae_cpu_offload=vae_cpu_offload,
            image_encoder_cpu_offload=image_encoder_cpu_offload,
            use_fsdp_inference=use_fsdp_inference,
            enable_torch_compile=enable_torch_compile,
            VSA_sparsity=vsa_sparsity,
            tp_size=tp_size,
            sp_size=sp_size,
            log_queue=log_queue,
        )
        
        with self._generators_lock:
            if cache_key not in self._generators:
                self._generators[cache_key] = gen
            else: # Another thread may have created it while we were loading.
                gen.shutdown()
                gen = self._generators[cache_key]
        return gen

    def _run_job(self, job: Job):
        if job.job_type == "inference":
            self._run_inference_job(job)
        else:
            self._run_training_job(job)

    def _run_training_job(self, job: Job):
        """Run a finetuning, distillation, or LoRA job via subprocess."""
        buf = job._log_buf
        os.makedirs(self.log_dir, exist_ok=True)
        job.log_file_path = os.path.join(self.log_dir, f"{job.id}.log")
        job_output_dir = os.path.join(self.output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)

        if not job.data_path or not os.path.isdir(job.data_path):
            job.status = JobStatus.FAILED
            job.error = (
                f"Data path '{job.data_path}' is required and must be an "
                "existing directory. Preprocess your dataset first."
            )
            job.finished_at = time.time()
            self._save_job(job)
            return

        module_info = get_training_module_info(job.workload_type, job.model_id)
        if not module_info:
            job.status = JobStatus.FAILED
            job.error = f"Unknown workload type: {job.workload_type}"
            job.finished_at = time.time()
            self._save_job(job)
            return

        module_path, _pipeline_workload, use_vsa, _is_lora = module_info
        env = os.environ.copy()
        env.update(get_training_env(use_vsa))

        job_dict = {
            "model_id": job.model_id,
            "data_path": job.data_path,
            "workload_type": job.workload_type,
            "num_gpus": job.num_gpus,
            "max_train_steps": job.max_train_steps,
            "train_batch_size": job.train_batch_size,
            "learning_rate": job.learning_rate,
            "num_latent_t": job.num_latent_t,
            "num_height": job.height,
            "num_width": job.width,
            "num_frames": job.num_frames,
            "validation_dataset_file": job.validation_dataset_file,
            "lora_rank": job.lora_rank,
            "ltx2_first_frame_conditioning_p": job.ltx2_first_frame_conditioning_p,
        }
        train_args = build_training_args(job_dict, job_output_dir)

        repo_root = Path(__file__).resolve().parent.parent
        torchrun_cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(job.num_gpus),
            "--nnodes", "1",
            str(repo_root / module_path),
        ] + train_args
        buf.write(f"Starting training: {' '.join(torchrun_cmd[:12])}...")
        buf.phase = "starting"

        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._save_job(job)

            with open(job.log_file_path, "w", encoding="utf-8") as log_file:
                job._process = subprocess.Popen(
                    torchrun_cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert job._process.stdout is not None
                for line in iter(job._process.stdout.readline, ""):
                    if job._stop_event.is_set():
                        job._process.terminate()
                        try:
                            job._process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            job._process.kill()
                        job.status = JobStatus.STOPPED
                        job.finished_at = time.time()
                        self._save_job(job)
                        buf.phase = "stopped"
                        return
                    line = line.rstrip()
                    if line:
                        buf.write(line)
                        log_file.write(line + "\n")
                        log_file.flush()

                job._process.wait()
                exit_code = job._process.returncode or 0

            if exit_code == 0:
                job.status = JobStatus.COMPLETED
                buf.progress = 100.0
                buf.phase = "done"
                # Training outputs checkpoints, not video
                ckpt_dirs = sorted(Path(job_output_dir).glob("checkpoint-*"))
                if ckpt_dirs:
                    job.output_path = str(ckpt_dirs[-1])
            else:
                job.status = JobStatus.FAILED
                job.error = f"Training exited with code {exit_code}"
                buf.phase = "failed"
            job.finished_at = time.time()
            self._save_job(job)

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            self._save_job(job)
            buf.phase = "failed"
            logger.exception("Training job %s failed", job.id)
        finally:
            job._process = None

    def _run_inference_job(self, job: Job):
        buf = job._log_buf
        os.makedirs(self.log_dir, exist_ok=True)
        job.log_file_path = os.path.join(self.log_dir, f"{job.id}.log")

        # Add file handler to persist logs
        file_handler = logging.FileHandler(
            job.log_file_path, mode='w', encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        job.log_file_handler = file_handler

        # Hook logger output into job log buffer (main process logs)
        fastvideo_logger = logging.getLogger("fastvideo")
        buffer_handler = LogBufferHandler(buf)
        fastvideo_logger.addHandler(buffer_handler)
        fastvideo_logger.addHandler(file_handler)

        # Queue for worker process logs (fsdp_load, cuda, etc.)
        # Use Manager().Queue() so it can be shared with spawned workers (spawn
        # does not inherit memory; mp.Queue only works through inheritance).
        log_queue = self._mp_manager.Queue()
        queue_listener = logging.handlers.QueueListener(
            log_queue, buffer_handler, file_handler, respect_handler_level=True
        )
        queue_listener.start()

        # Set output directory, create if it doesn't exist
        job_output_dir = os.path.join(self.output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)

        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._save_job(job)
            buf.phase = "starting"
            logger.info(f"Job {job.id} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at))}")
            logger.info(f"Model: {job.model_id}")
            logger.info(f"Prompt: {job.prompt}")

            if job._stop_event.is_set():
                job.status = JobStatus.STOPPED
                job.finished_at = time.time()
                self._save_job(job)
                logger.warning("Job stopped before execution started")
                return

            buf.phase = "loading model"
            logger.info("Loading model...")

            # Run generator creation in a background thread so we
            # can poll _stop_event while the (potentially slow)
            # model download / load is in progress.
            _gen_result: list[Any] = []
            _gen_error: list[BaseException] = []

            def _load_generator() -> None:
                try:
                    gen = self._get_or_create_generator(
                        job.model_id,
                        job.workload_type,
                        job.num_gpus,
                        dit_cpu_offload=job.dit_cpu_offload,
                        text_encoder_cpu_offload=(
                            job.text_encoder_cpu_offload
                        ),
                        vae_cpu_offload=job.vae_cpu_offload,
                        image_encoder_cpu_offload=(
                            job.image_encoder_cpu_offload
                        ),
                        use_fsdp_inference=job.use_fsdp_inference,
                        enable_torch_compile=(
                            job.enable_torch_compile
                        ),
                        vsa_sparsity=job.vsa_sparsity,
                        tp_size=job.tp_size,
                        sp_size=job.sp_size,
                        log_queue=log_queue,
                    )
                    _gen_result.append(gen)
                except BaseException as exc:
                    _gen_error.append(exc)

            loader = threading.Thread(
                target=_load_generator, daemon=True,
            )
            loader.start()

            while loader.is_alive():
                if job._stop_event.is_set():
                    job.status = JobStatus.STOPPED
                    job.finished_at = time.time()
                    self._save_job(job)
                    logger.warning(
                        "Job %s stopped during model loading",
                        job.id,
                    )
                    buf.phase = "stopped"
                    return
                loader.join(timeout=0.5)

            if _gen_error:
                raise _gen_error[0]

            generator = _gen_result[0]
            buf.phase = "generating"
            logger.info(
                "Starting generation for job %s (model=%s)", job.id, job.model_id
            )

            gen_kwargs: dict[str, Any] = {
                "prompt": job.prompt,
                "output_path": job_output_dir,
                "save_video": True,
                "num_inference_steps": job.num_inference_steps,
                "num_frames": job.num_frames,
                "height": job.height,
                "width": job.width,
                "guidance_scale": job.guidance_scale,
                "guidance_rescale": job.guidance_rescale,
                "fps": job.fps,
                "seed": job.seed,
                "negative_prompt": job.negative_prompt or "",
                "log_queue": log_queue,
            }
            if job.image_path:
                gen_kwargs["image_path"] = job.image_path
            generator.generate_video(**gen_kwargs)
        
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
            self._save_job(job)
            buf.phase = "done"

        except Exception as exception:
            error_msg = str(exception)
            logger.error(f"Critical error in job thread: {error_msg}")
            job.status = JobStatus.FAILED
            job.error = f"Critical error ({type(exception).__name__}): {error_msg}"
            job.finished_at = time.time()
            self._save_job(job)
            buf.phase = "failed"

        finally:
            queue_listener.stop()
            # Remove handlers and close file
            fastvideo_logger.removeHandler(buffer_handler)
            fastvideo_logger.removeHandler(file_handler)
            file_handler.flush()
            file_handler.close()
            job.log_file_handler = None
