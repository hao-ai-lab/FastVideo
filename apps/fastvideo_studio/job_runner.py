# SPDX-License-Identifier: Apache-2.0
"""
Job Runner for FastVideo video generation jobs.

Manages job lifecycle, execution, logging, and generator caching.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
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
import yaml

from fastvideo.utils import get_mp_context
from fastvideo_studio.database import Database
from fastvideo_studio.training_config import (
    build_training_config,
    get_training_env,
)

logger = logging.getLogger("fastvideo.studio.job_runner")

# Regex patterns for parsing tqdm-style progress output.
# Matches e.g. " 40%|████      | 20/50 " or " 20/50 "
_TQDM_PCT_RE = re.compile(r"(\d+)%\|")
_TQDM_FRAC_RE = re.compile(r"\b(\d+)/(\d+)\b")

_MAX_LOG_LINES = 2000  # ring-buffer cap per job

# ray's log relay prefixes worker lines like "(RayWorkerWrapper pid=123, ip=…)"
_RAY_RELAY_RE = re.compile(r"^\(\w+ pid=")


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class JobLogBuffer:
    """Ring buffer for storing job log lines with progress tracking."""

    def __init__(self, maxlen: int = _MAX_LOG_LINES):
        self._lines: collections.deque[str] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.progress: float = 0.0  # 0 – 100
        self.progress_msg: str = ""  # e.g. "20/50 steps"
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
        self.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s"))

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
    # DMD options
    dmd_use_vsa: bool = False
    dmd_vsa_sparsity: float = 0.8
    dmd_denoising_steps: str = "1000,757,522"
    real_score_guidance_scale: float = 3.5
    generator_update_interval: int = 5
    real_score_model_path: str = ""
    fake_score_model_path: str = ""
    # Internal
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _log_buf: JobLogBuffer = field(default_factory=JobLogBuffer, repr=False)
    log_file_handler: logging.FileHandler | None = field(default=None, repr=False)
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
            "validation_dataset_file": self.validation_dataset_file,
            "lora_rank": self.lora_rank,
            "dmd_use_vsa": self.dmd_use_vsa,
            "dmd_vsa_sparsity": self.dmd_vsa_sparsity,
            "dmd_denoising_steps": self.dmd_denoising_steps,
            "real_score_guidance_scale": self.real_score_guidance_scale,
            "generator_update_interval": self.generator_update_interval,
            "real_score_model_path": self.real_score_model_path or "",
            "fake_score_model_path": self.fake_score_model_path or "",
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

        # Exactly one generator lives in memory at a time. Loading a new
        # config always releases the old instance first (shutdown + placement
        # group teardown); unload deletes it outright.
        self._generator: Any | None = None
        self._generator_config: dict[str, Any] | None = None
        self._generator_state: str = "empty"  # empty | loading | ready | failed
        self._generator_error: str | None = None
        self._generator_lock = threading.Lock()  # guards the fields above
        # Serializes every slot transition (preload, job-triggered replace,
        # unload). Held for the full duration of a load.
        self._load_lock = threading.Lock()
        # The inference job currently generating, fed by the engine log tee
        # (ray relays worker output to the driver; tqdm lines land there).
        self._active_inference_job: Job | None = None

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
                path,
                job.id,
                exc,
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
                    dmd_use_vsa=row.get("dmd_use_vsa", False),
                    dmd_vsa_sparsity=float(row.get("dmd_vsa_sparsity", 0.8)),
                    dmd_denoising_steps=row.get("dmd_denoising_steps", "1000,757,522") or "1000,757,522",
                    real_score_guidance_scale=float(row.get("real_score_guidance_scale", 3.5)),
                    generator_update_interval=int(row.get("generator_update_interval", 5)),
                    real_score_model_path=row.get("real_score_model_path", "") or "",
                    fake_score_model_path=row.get("fake_score_model_path", "") or "",
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
            self._db.update_job(
                job.id, {
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
        dmd_use_vsa: bool = False,
        dmd_vsa_sparsity: float = 0.8,
        dmd_denoising_steps: str = "1000,757,522",
        real_score_guidance_scale: float = 3.5,
        generator_update_interval: int = 5,
        real_score_model_path: str = "",
        fake_score_model_path: str = "",
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
            dmd_use_vsa=dmd_use_vsa,
            dmd_vsa_sparsity=dmd_vsa_sparsity,
            dmd_denoising_steps=dmd_denoising_steps,
            real_score_guidance_scale=real_score_guidance_scale,
            generator_update_interval=generator_update_interval,
            real_score_model_path=real_score_model_path or "",
            fake_score_model_path=fake_score_model_path or "",
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
            raise ValueError("Job already completed. Delete and re-create to run again.")
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
                logger.critical("Unhandled exception escaped from _run_job for job %s: %s", job.id, exc, exc_info=True)
                with contextlib.suppress(Exception):
                    job.status = JobStatus.FAILED
                    job.error = (f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
                    job.finished_at = time.time()

        thread = threading.Thread(target=safe_run_job, args=(job, ), daemon=True)
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
            with contextlib.suppress(Exception):
                job._process.terminate()
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

    @staticmethod
    def _generator_config_dict(
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
    ) -> dict[str, Any]:
        """Canonical engine-config dict; equality here == same generator."""
        return {
            "model_id": model_id,
            "workload_type": workload_type,
            "num_gpus": num_gpus,
            "dit_cpu_offload": dit_cpu_offload,
            "text_encoder_cpu_offload": text_encoder_cpu_offload,
            "vae_cpu_offload": vae_cpu_offload,
            "image_encoder_cpu_offload": image_encoder_cpu_offload,
            "use_fsdp_inference": use_fsdp_inference,
            "enable_torch_compile": enable_torch_compile,
            "vsa_sparsity": vsa_sparsity,
            "tp_size": tp_size,
            "sp_size": sp_size,
        }

    def _slot_entry(self) -> dict[str, Any]:
        return {
            "state": self._generator_state,
            "error": self._generator_error,
            **(self._generator_config or {}),
        }

    def _running_inference_jobs(self) -> list[str]:
        with self._jobs_lock:
            return [j.id for j in self._jobs.values()
                    if j.status == JobStatus.RUNNING and j.job_type == "inference"]

    def preload_generator(self, **params: Any) -> dict[str, Any]:
        """Load a model into memory ahead of time. One load at a time; loading
        a different config always releases the current instance first."""
        config = self._generator_config_dict(**params)
        with self._generator_lock:
            if self._generator_state == "ready" and self._generator_config == config:
                return self._slot_entry()
        if not self._load_lock.acquire(blocking=False):
            raise RuntimeError("a model load is already in progress")
        try:
            if self._running_inference_jobs():
                raise RuntimeError("cannot swap models while inference jobs are running")
            with self._generator_lock:
                self._generator_state = "loading"
                self._generator_config = config
                self._generator_error = None
                entry = self._slot_entry()
        except BaseException:
            self._load_lock.release()
            raise

        def _load() -> None:
            try:  # the preload owns _load_lock until the load resolves
                self._load_into_slot_locked(config)
            except Exception:  # noqa: BLE001 -- state already set to failed
                pass
            finally:
                self._load_lock.release()

        threading.Thread(target=_load, daemon=True, name="generator-preload").start()
        return entry

    def _load_into_slot_locked(self, config: dict[str, Any]) -> Any:
        """Release whatever is resident and load ``config``. Caller MUST hold
        ``_load_lock``. State is 'loading' on entry or set here."""
        with self._generator_lock:
            gen = self._generator
            self._generator = None
            self._generator_state = "loading"
            self._generator_config = config
            self._generator_error = None
        if gen is not None:
            logger.info("Releasing resident generator before loading a new one")
            gen.shutdown()
            del gen

        # Import lazily so starting the server is fast even without a GPU.
        from fastvideo import VideoGenerator

        # Deployment-level knobs (set where the API server is launched):
        # FASTVIDEO_STUDIO_MODEL_PATHS="id=/local/dir,..." serves a registered
        # model id from local weights instead of the HF hub;
        # FASTVIDEO_STUDIO_EXECUTOR_BACKEND=ray runs workers on an existing
        # Ray cluster (the multi-node path — "mp" spawns local processes only).
        model_path = config["model_id"]
        for pair in os.environ.get("FASTVIDEO_STUDIO_MODEL_PATHS", "").split(","):
            mid, sep, path = pair.partition("=")
            if sep and mid.strip() == config["model_id"]:
                model_path = path.strip()
        executor_kwargs: dict[str, Any] = {}
        backend = os.environ.get("FASTVIDEO_STUDIO_EXECUTOR_BACKEND")
        if backend:
            executor_kwargs["distributed_executor_backend"] = backend

        logger.info("Loading model %s (%s)", config["model_id"],
                    ", ".join(f"{k}={v}" for k, v in config.items() if k != "model_id"))
        try:
            new_gen = VideoGenerator.from_pretrained(
                model_path,
                workload_type=config["workload_type"],
                num_gpus=config["num_gpus"],
                dit_cpu_offload=config["dit_cpu_offload"],
                text_encoder_cpu_offload=config["text_encoder_cpu_offload"],
                vae_cpu_offload=config["vae_cpu_offload"],
                image_encoder_cpu_offload=config["image_encoder_cpu_offload"],
                use_fsdp_inference=config["use_fsdp_inference"],
                enable_torch_compile=config["enable_torch_compile"],
                VSA_sparsity=config["vsa_sparsity"],
                tp_size=config["tp_size"],
                sp_size=config["sp_size"],
                **executor_kwargs,
            )
        except Exception as exc:
            logger.exception("Model load failed for %s", config["model_id"])
            with self._generator_lock:
                self._generator_state = "failed"
                self._generator_error = str(exc)
            raise
        with self._generator_lock:
            self._generator = new_gen
            self._generator_config = config
            self._generator_state = "ready"
            self._generator_error = None
        return new_gen

    def list_generators(self) -> list[dict[str, Any]]:
        """The resident slot, or empty when nothing is loaded."""
        with self._generator_lock:
            if self._generator_state == "empty":
                return []
            return [self._slot_entry()]

    def unload_generator(self, **_ignored: Any) -> bool:
        """Shut down and delete the resident generator, freeing GPU memory."""
        if not self._load_lock.acquire(blocking=False):
            raise RuntimeError("cannot unload while a model load is in progress")
        try:
            running = self._running_inference_jobs()
            if running:
                raise RuntimeError(f"cannot unload while inference jobs are running: {running}")
            with self._generator_lock:
                gen = self._generator
                empty = self._generator_state == "empty"
                self._generator = None
                self._generator_state = "empty"
                self._generator_config = None
                self._generator_error = None
            if empty:
                return False
            if gen is not None:
                logger.info("Releasing resident generator")
                gen.shutdown()
                del gen
            return True
        finally:
            self._load_lock.release()

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
        """Return the resident generator if the config matches; otherwise
        replace the slot. Blocks behind any in-flight load — every slot
        transition happens under ``_load_lock``, so a job can never observe a
        half-replaced slot."""
        del log_queue  # single-slot generators log via the engine tee
        config = self._generator_config_dict(
            model_id=model_id,
            workload_type=workload_type,
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
        with self._load_lock:  # waits out preloads / other jobs' replaces
            with self._generator_lock:
                if (self._generator_state == "ready"
                        and self._generator_config == config
                        and self._generator is not None):
                    return self._generator
            return self._load_into_slot_locked(config)

    def feed_engine_line(self, line: str) -> None:
        """Bridge ray-relayed worker output into the running job's log buffer.

        On the ray backend worker logs cannot cross nodes via the mp queue,
        but ray already relays them to the driver's stdout — which the engine
        tee captures. Lines with ray's actor prefix are attributed to the one
        running inference job, whose buffer parses tqdm into UI progress.
        Driver-side logging is excluded (it reaches the buffer via the
        logging handlers already).
        """
        job = self._active_inference_job
        if job is None or not _RAY_RELAY_RE.match(line):
            return
        try:
            job._log_buf.write(line)
        except Exception:  # noqa: BLE001 -- never break the tee
            pass

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
            job.error = (f"Data path '{job.data_path}' is required and must be an "
                         "existing directory. Preprocess your dataset first.")
            job.finished_at = time.time()
            self._save_job(job)
            return

        env = os.environ.copy()
        env.update(get_training_env())

        try:
            # Job.to_dict() carries every key the config builder reads
            # (extra keys are ignored by its .get() lookups).
            train_config = build_training_config(job.to_dict(), job_output_dir)
        except ValueError as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = time.time()
            self._save_job(job)
            return

        config_path = os.path.join(job_output_dir, "train_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(train_config, f, sort_keys=False)

        repo_root = Path(__file__).resolve().parent.parent
        torchrun_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(job.num_gpus),
            "--nnodes",
            "1",
            "-m",
            "fastvideo.train.entrypoint.train",
            "--config",
            config_path,
        ]
        buf.write(f"Starting training: {' '.join(torchrun_cmd)}")
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

            if job._stop_event.is_set():
                # Terminated by stop_job() without the reader loop observing the
                # flag (e.g. the process died between log lines).
                job.status = JobStatus.STOPPED
                buf.phase = "stopped"
            elif exit_code == 0:
                job.status = JobStatus.COMPLETED
                buf.progress = 100.0
                buf.phase = "done"
                # Training outputs checkpoints, not video. Sort by step number,
                # not lexically ("checkpoint-1000" < "checkpoint-500" as strings).
                ckpt_dirs = sorted(
                    Path(job_output_dir).glob("checkpoint-*"),
                    key=lambda p: int(m.group(1)) if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) else -1,
                )
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
        file_handler = logging.FileHandler(job.log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
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
        queue_listener = logging.handlers.QueueListener(log_queue,
                                                        buffer_handler,
                                                        file_handler,
                                                        respect_handler_level=True)
        queue_listener.start()

        # Set output directory, create if it doesn't exist
        job_output_dir = os.path.join(self.output_dir, job.id)
        os.makedirs(job_output_dir, exist_ok=True)

        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._save_job(job)
            buf.phase = "starting"
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.started_at))
            logger.info("Job %s started at %s", job.id, started)
            logger.info("Model: %s", job.model_id)
            logger.info("Prompt: %s", job.prompt)

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
                        text_encoder_cpu_offload=(job.text_encoder_cpu_offload),
                        vae_cpu_offload=job.vae_cpu_offload,
                        image_encoder_cpu_offload=(job.image_encoder_cpu_offload),
                        use_fsdp_inference=job.use_fsdp_inference,
                        enable_torch_compile=(job.enable_torch_compile),
                        vsa_sparsity=job.vsa_sparsity,
                        tp_size=job.tp_size,
                        sp_size=job.sp_size,
                        log_queue=log_queue,
                    )
                    _gen_result.append(gen)
                except BaseException as exc:
                    _gen_error.append(exc)

            loader = threading.Thread(
                target=_load_generator,
                daemon=True,
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
            self._active_inference_job = job  # engine tee feeds tqdm from here
            logger.info("Starting generation for job %s (model=%s)", job.id, job.model_id)

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
                logger.info("Found video output: %s", job.output_path)
            else:
                # Could be an image workload
                image_files = sorted(Path(job_output_dir).glob("*.png"))
                if image_files:
                    job.output_path = str(image_files[0])
                    logger.info("Found image output: %s", job.output_path)
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
            logger.error("Critical error in job thread: %s", error_msg)
            job.status = JobStatus.FAILED
            job.error = f"Critical error ({type(exception).__name__}): {error_msg}"
            job.finished_at = time.time()
            self._save_job(job)
            buf.phase = "failed"

        finally:
            if self._active_inference_job is job:
                self._active_inference_job = None
            queue_listener.stop()
            # Remove handlers and close file
            fastvideo_logger.removeHandler(buffer_handler)
            fastvideo_logger.removeHandler(file_handler)
            file_handler.flush()
            file_handler.close()
            job.log_file_handler = None
