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
import shutil
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

from fastvideo.registry import (get_registered_model_paths,
                                get_registered_models_with_workloads)
from ui.database import Database, _get_db_path
from ui.job_runner import JobRunner, JobStatus
from ui.models import (CreateDatasetRequest, CreateJobRequest,
                       SettingsUpdate, UpdateCaptionRequest)


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
verbose = 0

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


def _filter_video_files(files: list[UploadFile]) -> list[UploadFile]:
    """Filter uploaded files to only include videos."""
    return [
        f for f in files
        if Path(f.filename or "").suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
    ]


@app.post("/api/upload-raw-dataset")
async def upload_raw_dataset(
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """
    Upload raw video dataset. Returns path and file list.
    Filters files to only include videos. For folder upload, only adds videos.
    """
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized",
        )
    settings = database.get_settings()
    base_path = (
        settings.get("datasetUploadPath")
        or settings.get("dataset_upload_path")
        or ""
    )
    if base_path and isinstance(base_path, str):
        base_path = base_path.strip()
    else:
        base_path = ""
    if not base_path:
        base_path = datasets_upload_dir
    else:
        base_path = os.path.abspath(base_path)
    if not base_path:
        raise HTTPException(
            status_code=503,
            detail="Dataset upload directory not configured. Set it in Settings.",
        )
    filtered = _filter_video_files(files)
    if not filtered:
        raise HTTPException(
            status_code=400,
            detail=(
                "No video files found. "
                f"Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            ),
        )
    upload_id = uuid.uuid4().hex
    media_folder = os.path.join(base_path, upload_id)
    os.makedirs(media_folder, exist_ok=True)
    file_names = []
    for uf in filtered:
        name = uf.filename or f"{uuid.uuid4().hex}"
        if "/" in name or "\\" in name:
            name = os.path.basename(name)
        if not Path(name).suffix:
            name += ".mp4"
        dest = os.path.join(media_folder, name)
        try:
            contents = await uf.read()
            # Detect Git LFS pointer files (they look like tiny text files, not real videos)
            if contents.startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File {uf.filename} appears to be a Git LFS pointer, not an actual video. "
                        "Please run `git lfs pull` (or otherwise download the real video files) "
                        "and upload the resolved videos instead."
                    ),
                )
            with open(dest, "wb") as f:
                f.write(contents)
            file_names.append(name)
        except HTTPException:
            # Re-raise HTTPExceptions (e.g. LFS pointer detection) as-is
            raise
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save {uf.filename}: {e}",
            ) from e
    return {
        "path": os.path.abspath(media_folder),
        "upload_id": upload_id,
        "file_names": file_names,
    }


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
        ltx2_first_frame_conditioning_p=req.ltx2_first_frame_conditioning_p,
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
        dmd_use_vsa=req.dmd_use_vsa,
        dmd_vsa_sparsity=req.dmd_vsa_sparsity,
        dmd_denoising_steps=req.dmd_denoising_steps or "1000,757,522",
        min_timestep_ratio=req.min_timestep_ratio,
        max_timestep_ratio=req.max_timestep_ratio,
        real_score_guidance_scale=req.real_score_guidance_scale,
        generator_update_interval=req.generator_update_interval,
        real_score_model_path=req.real_score_model_path or "",
        fake_score_model_path=req.fake_score_model_path or "",
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


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    """Return all datasets, newest first, with file_count and size_bytes."""
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized",
        )
    out = []
    for ds in database.get_all_datasets():
        count, size = _dataset_media_stats(ds["id"])
        out.append({**ds, "file_count": count, "size_bytes": size})
    return out


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Get a single dataset by ID, with file_count and size_bytes."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    count, size = _dataset_media_stats(dataset_id)
    return {**ds, "file_count": count, "size_bytes": size}


def _dataset_media_dir(dataset_id: str) -> str:
    """Return the path to a dataset's media directory."""
    return os.path.join(datasets_upload_dir, dataset_id)


def _dataset_media_stats(dataset_id: str) -> tuple[int, int]:
    """Return (file_count, total_size_bytes) for a dataset's media directory."""
    media_dir = _dataset_media_dir(dataset_id)
    video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    count = 0
    total = 0
    if os.path.isdir(media_dir):
        for name in os.listdir(media_dir):
            ext = os.path.splitext(name)[1].lower()
            path = os.path.join(media_dir, name)
            if os.path.isfile(path) and ext in video_exts:
                count += 1
                try:
                    total += os.path.getsize(path)
                except OSError:
                    pass
    return (count, total)


@app.post("/api/datasets", status_code=201)
def create_dataset(req: CreateDatasetRequest) -> dict[str, Any]:
    """Create a new dataset. Moves upload into datasets_upload_dir/{id}, persists dataset and captions."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    if not req.upload_path:
        raise HTTPException(
            status_code=400,
            detail="upload_path is required. Upload media files first.",
        )
    if not req.file_names:
        raise HTTPException(
            status_code=400,
            detail="No media files found. Ensure at least one image or video.",
        )
    dataset_id = str(uuid.uuid4())
    created_at = time.time()
    dest_dir = os.path.join(datasets_upload_dir, dataset_id)
    if os.path.exists(dest_dir):
        raise HTTPException(
            status_code=409,
            detail="Dataset directory already exists",
        )
    try:
        shutil.copytree(req.upload_path, dest_dir)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to move upload: {e}",
        ) from e
    try:
        shutil.rmtree(req.upload_path)
    except OSError:
        pass  # Best-effort cleanup of upload folder
    dataset = {
        "id": dataset_id,
        "name": req.name,
        "created_at": created_at,
    }
    database.insert_dataset(dataset)
    for fn in req.file_names:
        caption = (req.captions.get(fn, "") if req.captions else "") or ""
        database.upsert_dataset_caption(dataset_id, fn, caption)
    return database.get_dataset(dataset_id) or dataset


@app.get("/api/datasets/{dataset_id}/files")
def get_dataset_files(dataset_id: str) -> dict[str, Any]:
    """List media files in dataset with their captions."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    captions = database.get_dataset_captions(dataset_id)
    media_dir = _dataset_media_dir(dataset_id)
    video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    file_names = []
    if os.path.isdir(media_dir):
        for name in sorted(os.listdir(media_dir)):
            ext = os.path.splitext(name)[1].lower()
            path = os.path.join(media_dir, name)
            if os.path.isfile(path) and ext in video_exts:
                file_names.append(name)
    return {
        "file_names": file_names,
        "captions": captions,
    }


@app.put("/api/datasets/{dataset_id}/captions")
def update_dataset_caption(
    dataset_id: str, req: UpdateCaptionRequest
) -> dict[str, str]:
    """Update a single caption for a file. Persists to database."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    database.upsert_dataset_caption(dataset_id, req.file_name, req.caption)
    return {"detail": "Caption updated"}


@app.get("/api/datasets/{dataset_id}/media/{file_name:path}")
def serve_dataset_media(dataset_id: str, file_name: str) -> FileResponse:
    """Serve a media file from a dataset."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    ds = database.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    media_path = os.path.join(_dataset_media_dir(dataset_id), file_name)
    if not os.path.isfile(media_path):
        raise HTTPException(status_code=404, detail="File not found")
    import mimetypes
    mime, _ = mimetypes.guess_type(media_path)
    return FileResponse(media_path, media_type=mime or "video/mp4")


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    """Delete a dataset and its media directory."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    if not database.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    dest_dir = os.path.join(datasets_upload_dir, dataset_id)
    if os.path.isdir(dest_dir):
        try:
            shutil.rmtree(dest_dir)
        except OSError:
            pass  # Best-effort cleanup
    return {"detail": f"Dataset {dataset_id} deleted"}


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
    global job_runner, database, upload_dir, verbose, datasets_upload_dir  # noqa: PLW0603

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

    create_local_env(args.host, args.port)

    db_path = _get_db_path(data_dir)
    database = Database(db_path)
    verbose = args.verbose
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
