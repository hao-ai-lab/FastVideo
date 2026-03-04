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
import threading
import time
import uuid
from pathlib import Path
import uvicorn
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastvideo.registry import (get_registered_model_paths,
                                get_registered_models_with_workloads)
from ui.database import Database, _get_db_path
from ui.job_runner import JobRunner, JobStatus
from ui.preprocess_runner import run_preprocess


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
database: Database | None = None
upload_dir: str = ""
datasets_upload_dir: str = ""
datasets_output_dir: str = ""
_preprocess_tasks: dict[str, tuple[threading.Thread, threading.Event]] = {}


class CreateJobRequest(BaseModel):
    model_id: str
    prompt: str
    workload_type: str = "t2v"
    job_type: str = "inference"
    image_path: str = ""
    data_path: str = ""
    max_train_steps: int = 1000
    train_batch_size: int = 1
    learning_rate: float = 5e-5
    num_latent_t: int = 20
    validation_dataset_file: str = ""
    lora_rank: int = 32
    negative_prompt: str = ""
    num_inference_steps: int = 50
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 5.0
    guidance_rescale: float = 0.0
    fps: int = 24
    seed: int = 1024
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


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    """Return persisted default options (for new job creation)."""
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized (persistence disabled)",
        )
    return database.get_settings()


class SettingsUpdate(BaseModel):
    defaultModelId: str | None = None
    defaultModelIdT2v: str | None = None
    defaultModelIdI2v: str | None = None
    defaultModelIdT2i: str | None = None
    numInferenceSteps: int | None = None
    numFrames: int | None = None
    height: int | None = None
    width: int | None = None
    guidanceScale: float | None = None
    guidanceRescale: float | None = None
    fps: int | None = None
    seed: int | None = None
    numGpus: int | None = None
    ditCpuOffload: bool | None = None
    textEncoderCpuOffload: bool | None = None
    vaeCpuOffload: bool | None = None
    imageEncoderCpuOffload: bool | None = None
    useFsdpInference: bool | None = None
    enableTorchCompile: bool | None = None
    vsaSparsity: float | None = None
    tpSize: int | None = None
    spSize: int | None = None
    autoStartJob: bool | None = None


@app.put("/api/settings")
def update_settings(settings: SettingsUpdate) -> dict[str, Any]:
    """Update persisted default options. Only provided fields are updated."""
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized (persistence disabled)",
        )
    updates = settings.model_dump(exclude_unset=True)
    if not updates:
        return database.get_settings()
    database.save_settings(updates)
    return database.get_settings()


@app.get("/api/models")
def list_models(workload_type: str | None = None) -> list[dict[str, Any]]:
    """Return the catalogue of available video-generation models.

    Query params:
        workload_type: If set (t2v, i2v, t2i), only return models that
            support this workload. Otherwise return all models.
    """
    if workload_type:
        models = get_registered_models_with_workloads(workload_type=workload_type)
        return [{"id": m["id"], "label": m["label"]} for m in models]
    return _available_models


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload an image file for I2V jobs. Returns the absolute path."""
    global upload_dir  # noqa: PLW0603
    if not upload_dir:
        raise HTTPException(
            status_code=503,
            detail="Upload directory not configured",
        )
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type. Allowed: "
                f"{', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            ),
        )
    os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(upload_dir, unique_name)
    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save upload: {e}",
        ) from e
    return {"path": os.path.abspath(dest_path)}


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov"}
ALLOWED_PARQUET_EXTENSIONS = {".parquet"}


@app.post("/api/upload-dataset-captions")
async def upload_dataset_captions(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Upload captions JSON for raw dataset. Returns folder path."""
    global datasets_upload_dir  # noqa: PLW0603
    if not datasets_upload_dir:
        raise HTTPException(
            status_code=503,
            detail="Upload directory not configured",
        )
    ext = Path(file.filename or "").suffix.lower()
    if ext != ".json":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: .json",
        )
    upload_id = uuid.uuid4().hex
    folder = os.path.join(datasets_upload_dir, upload_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "videos"), exist_ok=True)
    dest_path = os.path.join(folder, "videos2caption.json")
    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save upload: {e}",
        ) from e
    return {"path": os.path.abspath(folder), "upload_id": upload_id}


@app.post("/api/upload-dataset-videos")
async def upload_dataset_videos(
    upload_id: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, str]:
    """Upload video files to existing raw dataset folder."""
    global datasets_upload_dir  # noqa: PLW0603
    if not datasets_upload_dir:
        raise HTTPException(
            status_code=503,
            detail="Upload directory not configured",
        )
    folder = os.path.join(datasets_upload_dir, upload_id, "videos")
    if not os.path.isdir(os.path.dirname(folder)):
        raise HTTPException(
            status_code=400,
            detail="Upload captions first to create dataset folder",
        )
    os.makedirs(folder, exist_ok=True)
    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            continue
        dest = os.path.join(folder, uf.filename or f"{uuid.uuid4().hex}{ext}")
        try:
            contents = await uf.read()
            with open(dest, "wb") as f:
                f.write(contents)
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save {uf.filename}: {e}",
            ) from e
    return {"path": os.path.abspath(os.path.dirname(folder))}


@app.post("/api/upload-dataset-parquet")
async def upload_dataset_parquet(
    files: list[UploadFile] = File(...),
) -> dict[str, str]:
    """Upload parquet files for preprocessed dataset. Returns folder path."""
    global datasets_upload_dir  # noqa: PLW0603
    if not datasets_upload_dir:
        raise HTTPException(
            status_code=503,
            detail="Upload directory not configured",
        )
    parquet_files = [
        f for f in files
        if Path(f.filename or "").suffix.lower() in ALLOWED_PARQUET_EXTENSIONS
    ]
    if not parquet_files:
        raise HTTPException(
            status_code=400,
            detail="No .parquet files provided",
        )
    upload_id = uuid.uuid4().hex
    folder = os.path.join(
        datasets_upload_dir, upload_id, "combined_parquet_dataset"
    )
    os.makedirs(folder, exist_ok=True)
    for uf in parquet_files:
        dest = os.path.join(folder, uf.filename or f"{uuid.uuid4().hex}.parquet")
        try:
            contents = await uf.read()
            with open(dest, "wb") as f:
                f.write(contents)
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save {uf.filename}: {e}",
            ) from e
    return {"path": os.path.abspath(os.path.dirname(folder))}


@app.get("/api/jobs")
def list_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
    """Return jobs (newest first). Optionally filter by job_type."""
    return [j.to_dict() for j in job_runner.list_jobs(job_type=job_type)]


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
    job_type = req.job_type or "inference"
    if job_type == "inference":
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
        workload_type=req.workload_type or "t2v",
        job_type=job_type,
        image_path=req.image_path or "",
        data_path=req.data_path or "",
        max_train_steps=req.max_train_steps,
        train_batch_size=req.train_batch_size,
        learning_rate=req.learning_rate,
        num_latent_t=req.num_latent_t,
        validation_dataset_file=req.validation_dataset_file or "",
        lora_rank=req.lora_rank,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.num_inference_steps,
        num_frames=req.num_frames,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        guidance_rescale=req.guidance_rescale,
        fps=req.fps,
        seed=req.seed,
        num_gpus=req.num_gpus,
        dit_cpu_offload=req.dit_cpu_offload,
        text_encoder_cpu_offload=req.text_encoder_cpu_offload,
        vae_cpu_offload=req.vae_cpu_offload,
        image_encoder_cpu_offload=req.image_encoder_cpu_offload,
        use_fsdp_inference=req.use_fsdp_inference,
        enable_torch_compile=req.enable_torch_compile,
        vsa_sparsity=req.vsa_sparsity,
        tp_size=req.tp_size,
        sp_size=req.sp_size,
    )

    if database is not None:
        settings = database.get_settings()
        if settings.get("autoStartJob"):
            try:
                job = job_runner.start_job(job.id)
            except ValueError as exc:
                logger.warning(
                    "Auto-start failed for job %s: %s",
                    job.id, exc,
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


# --- Datasets ---


class CreateDatasetRequest(BaseModel):
    name: str
    raw_path: str = ""
    output_path: str | None = None  # For parquet: preprocessed path, status=ready
    workload_type: str = "t2v"
    model_path: str
    dataset_type: str = "merged"
    num_gpus: int = 1


@app.get("/api/datasets")
def list_datasets(status: str | None = None) -> list[dict[str, Any]]:
    """Return datasets. Optionally filter by status (e.g. 'ready')."""
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized",
        )
    return database.get_all_datasets(status=status)


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Get a single dataset by ID."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@app.post("/api/datasets", status_code=201)
def create_dataset(req: CreateDatasetRequest) -> dict[str, Any]:
    """Create a new dataset. Parquet: status=ready. Raw/HF: status=pending."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    dataset_id = str(uuid.uuid4())
    created_at = time.time()
    if req.output_path:
        output_path = req.output_path
        status = "ready"
    else:
        output_path = os.path.join(
            datasets_output_dir, dataset_id, "output"
        )
        status = "pending"
    dataset = {
        "id": dataset_id,
        "name": req.name,
        "raw_path": req.raw_path,
        "output_path": output_path,
        "workload_type": req.workload_type,
        "model_path": req.model_path,
        "dataset_type": req.dataset_type,
        "status": status,
        "error": None,
        "created_at": created_at,
        "num_gpus": req.num_gpus,
        "log_file_path": "",
    }
    database.insert_dataset(dataset)
    return dataset


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    """Delete a dataset. Stops preprocessing if running."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    if dataset_id in _preprocess_tasks:
        _, stop_ev = _preprocess_tasks[dataset_id]
        stop_ev.set()
        del _preprocess_tasks[dataset_id]
    if not database.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"detail": f"Dataset {dataset_id} deleted"}


@app.post("/api/datasets/{dataset_id}/preprocess")
def start_preprocess(dataset_id: str) -> dict[str, Any]:
    """Start preprocessing for a dataset."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not ds.get("raw_path"):
        raise HTTPException(
            status_code=400,
            detail="Dataset has no raw path (e.g. Parquet datasets are already preprocessed)",
        )
    if dataset_id in _preprocess_tasks:
        raise HTTPException(
            status_code=409,
            detail="Preprocessing already running for this dataset",
        )
    if ds["status"] == "ready":
        raise HTTPException(
            status_code=409,
            detail="Dataset is already preprocessed",
        )

    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "ui_logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"preprocess_{dataset_id}.log")

    output_dir = ds["output_path"]
    os.makedirs(output_dir, exist_ok=True)

    stop_event = threading.Event()

    def on_status_change(status: str, error: str | None) -> None:
        if database is None:
            return
        updates: dict[str, Any] = {"status": status}
        if error:
            updates["error"] = error
        if status == "ready":
            updates["output_path"] = output_dir
        database.update_dataset(dataset_id, updates)
        if status in ("ready", "failed", "stopped"):
            _preprocess_tasks.pop(dataset_id, None)

    def run() -> None:
        run_preprocess(
            dataset_id=dataset_id,
            raw_path=ds["raw_path"],
            output_dir=output_dir,
            workload_type=ds["workload_type"],
            model_path=ds["model_path"],
            dataset_type=ds["dataset_type"],
            num_gpus=ds["num_gpus"],
            log_file_path=log_file_path,
            on_status_change=on_status_change,
            stop_event=stop_event,
        )

    database.update_dataset(
        dataset_id,
        {"status": "preprocessing", "error": None, "log_file_path": log_file_path},
    )
    thread = threading.Thread(target=run, daemon=True)
    _preprocess_tasks[dataset_id] = (thread, stop_event)
    thread.start()

    return database.get_dataset(dataset_id) or ds


@app.post("/api/datasets/{dataset_id}/stop-preprocess")
def stop_preprocess(dataset_id: str) -> dict[str, Any]:
    """Stop preprocessing for a dataset."""
    if dataset_id not in _preprocess_tasks:
        raise HTTPException(
            status_code=404,
            detail="No preprocessing running for this dataset",
        )
    _, stop_event = _preprocess_tasks[dataset_id]
    stop_event.set()
    if database is not None:
        ds = database.get_dataset(dataset_id)
        if ds:
            return ds
    return {"detail": "Stop requested"}


@app.get("/api/datasets/{dataset_id}/logs")
def get_dataset_logs(dataset_id: str, after: int = 0) -> dict[str, Any]:
    """Return preprocessing log lines for a dataset."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    log_path = ds.get("log_file_path") or ""
    if not log_path or not os.path.isfile(log_path):
        return {"lines": [], "total": 0}
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    lines = [ln.rstrip() for ln in lines[after:] if ln]
    return {"lines": lines, "total": total}


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
    global job_runner, database, upload_dir, datasets_upload_dir, datasets_output_dir  # noqa: PLW0603

    # Set up signal handlers to prevent worker crashes from killing the server
    _setup_signal_handlers()

    default_log_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "ui_logs"
    )
    default_data_dir = Path(
        os.path.dirname(__file__), "..", "outputs", "ui_data"
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
        "--data-dir",
        default=str(default_data_dir),
        help=(
            "Directory for SQLite database (jobs + settings persistence) "
            f"(default: {default_data_dir})"
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
    data_dir = Path(args.data_dir).resolve()
    upload_dir = str(data_dir / "uploads")
    datasets_upload_dir = str(data_dir / "uploads" / "datasets")
    datasets_output_dir = str(data_dir / "datasets")

    create_local_env(args.host, args.port)

    db_path = _get_db_path(data_dir)
    database = Database(db_path)
    logger.info("Database: %s", db_path)

    job_runner = JobRunner(
        output_dir=output_dir,
        log_dir=log_dir,
        verbose=args.verbose,
        database=database,
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
