# SPDX-License-Identifier: Apache-2.0
"""
In-memory mock of the FastVideo Studio API (``server.py``).

This mock implements the same ``/api`` routes and response shapes as the real
FastAPI server but keeps everything in memory and never touches the FastVideo
library, a GPU, or a database. It is meant to back the Playwright e2e suite so
the Next.js frontend can be exercised end-to-end without a real backend.

Job lifecycle is simulated by recording a start timestamp and *computing* the
status on read: a started job reports ``running`` for a few seconds and then
flips to ``completed`` with an ``output_path``, so polling the job list / logs
shows progression. Generated media is a tiny 1s ``testsrc`` MP4 built lazily
with ffmpeg and cached.

Usage (from the ``apps/`` directory)::

    PYTHONPATH=.. python -m fastvideo_studio.mock_server --port 8189
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from fastvideo_studio.models import (CreateDatasetRequest, CreateJobRequest, SettingsUpdate, UpdateCaptionRequest)

# --- Config -----------------------------------------------------------------

# How long a started job stays "running" before it flips to "completed".
COMPLETE_AFTER_SECONDS = 3.0
FFMPEG_BIN = shutil.which(os.getenv("FASTVIDEO_FFMPEG_BIN", "ffmpeg"))

# A small catalogue of fake models keyed by workload type. Mirrors the real
# server's {id, label} shape (label derived from the HF-style path).
_MODELS_BY_WORKLOAD: dict[str, list[str]] = {
    "t2v": [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "FastVideo/FastHunyuan-diffusers",
    ],
    "i2v": [
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    ],
    "t2i": [
        "black-forest-labs/FLUX.1-schnell",
    ],
}


def _model_label(model_path: str) -> str:
    """Derive a readable label from an HF-style model path."""
    return model_path.split("/")[-1].replace("-", " ").replace("_", " ")


def _models_for(workload_type: str | None) -> list[dict[str, str]]:
    if workload_type:
        paths = _MODELS_BY_WORKLOAD.get(workload_type, [])
    else:
        seen: dict[str, None] = {}
        for paths_for_workload in _MODELS_BY_WORKLOAD.values():
            for path in paths_for_workload:
                seen.setdefault(path, None)
        paths = list(seen)
    return [{"id": path, "label": _model_label(path)} for path in paths]


# camelCase settings dict, mirroring database.get_settings() / the frontend
# Settings interface in src/lib/api.ts.
_DEFAULT_SETTINGS: dict[str, Any] = {
    "defaultModelId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "defaultModelIdT2v": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "defaultModelIdI2v": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "defaultModelIdT2i": "black-forest-labs/FLUX.1-schnell",
    "numInferenceSteps": 50,
    "numFrames": 81,
    "height": 480,
    "width": 832,
    "guidanceScale": 5.0,
    "guidanceRescale": 0.0,
    "fps": 24,
    "seed": 1024,
    "numGpus": 1,
    "ditCpuOffload": False,
    "textEncoderCpuOffload": False,
    "vaeCpuOffload": False,
    "imageEncoderCpuOffload": False,
    "useFsdpInference": False,
    "enableTorchCompile": False,
    "vsaSparsity": 0.0,
    "tpSize": -1,
    "spSize": -1,
    "autoStartJob": False,
    "datasetUploadPath": "",
}

# --- In-memory state --------------------------------------------------------

_state_lock = threading.Lock()
_settings: dict[str, Any] = dict(_DEFAULT_SETTINGS)
_jobs: dict[str, dict[str, Any]] = {}
_datasets: dict[str, dict[str, Any]] = {}
# dataset_id -> {"file_names": [...], "captions": {file_name: caption}}
_dataset_files: dict[str, dict[str, Any]] = {}

# Lazily-built, cached media clips (shared by every job/dataset media response),
# keyed by extension: "mp4" (1s testsrc video) or "png" (single testsrc frame).
_mock_media_cache: dict[str, str] = {}
_mock_media_lock = threading.Lock()


def _build_mock_media(kind: str) -> str:
    """Build (once) a tiny testsrc clip of the given ``kind`` with ffmpeg; cache the path."""
    with _mock_media_lock:
        cached = _mock_media_cache.get(kind)
        if cached and os.path.isfile(cached):
            return cached
        if not FFMPEG_BIN:
            raise HTTPException(status_code=500, detail="ffmpeg is required to build mock media")
        fd, path = tempfile.mkstemp(prefix="fvstudio_mock_", suffix=f".{kind}")
        os.close(fd)
        if kind == "png":
            command = [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=1",
                "-frames:v",
                "1",
                "-f",
                "image2",
                path,
            ]
        else:
            command = [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=24",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                path,
            ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            with contextlib.suppress(OSError):
                os.remove(path)
            raise HTTPException(status_code=500, detail=f"ffmpeg failed to build mock {kind}: {exc}") from exc
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg produced no mock {kind} bytes")
        _mock_media_cache[kind] = path
        return path


# --- Job helpers ------------------------------------------------------------


def _new_job_dict(req: CreateJobRequest) -> dict[str, Any]:
    """Build a job dict (mirrors job_runner.Job.to_dict()) in the pending state."""
    job_id = str(uuid.uuid4())
    return {
        "id": job_id,
        "model_id": req.model_id,
        "prompt": req.prompt,
        "workload_type": req.workload_type or "t2v",
        "job_type": req.job_type or "inference",
        "image_path": req.image_path or "",
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "output_path": None,
        "log_file_path": None,
        "num_inference_steps": req.num_inference_steps,
        "num_frames": req.num_frames,
        "height": req.height,
        "width": req.width,
        "guidance_scale": req.guidance_scale,
        "guidance_rescale": req.guidance_rescale,
        "fps": req.fps,
        "seed": req.seed,
        "negative_prompt": req.negative_prompt or "",
        "num_gpus": req.num_gpus,
        "data_path": req.data_path or "",
        "progress": 0.0,
        "progress_msg": "",
        "phase": "pending",
    }


def _advance_job(job: dict[str, Any]) -> None:
    """Flip a running job to completed once enough wall-clock time has passed.

    Status is *computed on read* from the recorded start timestamp, so polling
    the job list / logs naturally shows pending -> running -> completed.
    """
    if job["status"] != "running" or not job.get("started_at"):
        return
    elapsed = time.time() - job["started_at"]
    if elapsed >= COMPLETE_AFTER_SECONDS:
        job["status"] = "completed"
        job["finished_at"] = job["started_at"] + COMPLETE_AFTER_SECONDS
        job["progress"] = 100.0
        job["progress_msg"] = "50/50 steps"
        job["phase"] = "done"
        ext = "png" if job.get("workload_type") == "t2i" else "mp4"
        job["output_path"] = f"/mock/outputs/{job['id']}/output.{ext}"


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Advance + return a copy safe to serialize."""
    _advance_job(job)
    return dict(job)


_LOG_TAIL = [
    "Loading model...",
    "Model loaded.",
    "Starting generation...",
    "Denoising step 10/50",
    "Denoising step 20/50",
    "Denoising step 30/50",
    "Denoising step 40/50",
    "Denoising step 50/50",
    "Generation complete. Saving output...",
    "Saved output file.",
    "Job completed successfully.",
]


def _log_sequence(job: dict[str, Any]) -> list[str]:
    return [
        f"Job {job['id']} started",
        f"Model: {job['model_id']}",
        f"Prompt: {job['prompt']}",
        *_LOG_TAIL,
    ]


def _compute_logs(job: dict[str, Any]) -> dict[str, Any]:
    """Return JobLogs-shaped data, growing the visible lines as time passes."""
    seq = _log_sequence(job)
    status = job["status"]
    if status == "pending":
        return {"lines": [], "progress": 0.0, "progress_msg": "", "phase": "pending"}
    if status == "completed":
        return {"lines": seq, "progress": 100.0, "progress_msg": "50/50 steps", "phase": "done"}
    if status in ("failed", "stopped"):
        # Keep the total line count monotonic across running -> terminal (the
        # `after` cursor relies on it): show every non-final line plus a terminal
        # notice, which is always >= any prefix a running job revealed.
        lines = seq[:-1] + [f"Job {status}."]
        return {"lines": lines, "progress": job.get("progress", 0.0), "progress_msg": "", "phase": status}
    # running: reveal a prefix proportional to elapsed time, but never the final
    # "completed" line — that appears only once the job is actually completed.
    elapsed = time.time() - (job.get("started_at") or time.time())
    frac = max(0.0, min(elapsed / COMPLETE_AFTER_SECONDS, 0.99))
    reveal = min(max(4, round(frac * len(seq))), len(seq) - 1)
    lines = seq[:reveal]
    if frac < 0.2:
        phase = "loading model"
    elif frac < 0.85:
        phase = "denoising"
    else:
        phase = "saving"
    return {
        "lines": lines,
        "progress": round(frac * 100.0, 1),
        "progress_msg": f"{int(frac * 50)}/50 steps",
        "phase": phase,
    }


# --- Dataset helpers --------------------------------------------------------


def _dataset_stats(dataset_id: str) -> tuple[int, int]:
    files = _dataset_files.get(dataset_id, {}).get("file_names", [])
    count = len(files)
    # Fake but stable per-file size so the UI shows a non-zero footprint.
    return count, count * 1_048_576


def _public_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    count, size = _dataset_stats(dataset["id"])
    return {**dataset, "file_count": count, "size_bytes": size}


def _seed() -> None:
    """Seed a couple of datasets and one completed inference job."""
    now = time.time()
    seeds = [
        ("Sunset Clips", ["sunset_01.mp4", "sunset_02.mp4"], {
            "sunset_01.mp4": "A sunset over the ocean",
            "sunset_02.mp4": "A sunset over the mountains"
        }),
        ("City Timelapse", ["city_01.mp4"], {
            "city_01.mp4": "A busy city intersection at night"
        }),
    ]
    for idx, (name, file_names, captions) in enumerate(seeds):
        dataset_id = str(uuid.uuid4())
        _datasets[dataset_id] = {"id": dataset_id, "name": name, "created_at": now - 600 + idx}
        _dataset_files[dataset_id] = {"file_names": list(file_names), "captions": dict(captions)}

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "prompt": "A curious raccoon peers through a field of yellow sunflowers",
        "workload_type": "t2v",
        "job_type": "inference",
        "image_path": "",
        "status": "completed",
        "created_at": now - 300,
        "started_at": now - 280,
        "finished_at": now - 250,
        "error": None,
        "output_path": f"/mock/outputs/{job_id}/output.mp4",
        "log_file_path": f"/mock/logs/{job_id}.log",
        "num_inference_steps": 50,
        "num_frames": 81,
        "height": 480,
        "width": 832,
        "guidance_scale": 5.0,
        "guidance_rescale": 0.0,
        "fps": 24,
        "seed": 1024,
        "negative_prompt": "",
        "num_gpus": 1,
        "data_path": "",
        "progress": 100.0,
        "progress_msg": "50/50 steps",
        "phase": "done",
    }


# --- App --------------------------------------------------------------------

app = FastAPI(title="FastVideo Studio Mock API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_seed()

# --- Settings ---------------------------------------------------------------


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    with _state_lock:
        return dict(_settings)


@app.put("/api/settings")
def update_settings(settings: SettingsUpdate) -> dict[str, Any]:
    updates = settings.model_dump(exclude_unset=True)
    with _state_lock:
        _settings.update({k: v for k, v in updates.items() if v is not None})
        return dict(_settings)


# --- Models -----------------------------------------------------------------


@app.get("/api/models")
def list_models(workload_type: str | None = None) -> list[dict[str, Any]]:
    return _models_for(workload_type)


# --- Uploads ----------------------------------------------------------------


@app.post("/api/upload-image")
async def upload_image(file: Annotated[UploadFile, File()]) -> dict[str, str]:
    name = file.filename or "image.png"
    return {"path": f"/mock/uploads/{uuid.uuid4().hex}_{os.path.basename(name)}"}


@app.post("/api/upload-raw-dataset")
async def upload_raw_dataset(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
    video_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    file_names: list[str] = []
    for uf in files:
        name = os.path.basename(uf.filename or f"{uuid.uuid4().hex}.mp4")
        if os.path.splitext(name)[1].lower() in video_exts:
            file_names.append(name)
    if not file_names:
        raise HTTPException(status_code=400, detail="No video files found.")
    upload_id = uuid.uuid4().hex
    path = f"/mock/uploads/{upload_id}"
    return {"path": path, "upload_id": upload_id, "file_names": file_names}


# --- Jobs -------------------------------------------------------------------


@app.get("/api/jobs")
def list_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
    # _public_job mutates the shared job dict via _advance_job, so it must run
    # under the lock (matching every other job route) to avoid racing start/stop.
    with _state_lock:
        jobs = list(_jobs.values())
        if job_type:
            jobs = [j for j in jobs if j.get("job_type") == job_type]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return [_public_job(j) for j in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _public_job(job)


@app.post("/api/jobs", status_code=201)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    job = _new_job_dict(req)
    with _state_lock:
        _jobs[job["id"]] = job
        if _settings.get("autoStartJob"):
            _start(job)
        return _public_job(job)


def _start(job: dict[str, Any]) -> None:
    job["status"] = "running"
    job["started_at"] = time.time()
    job["finished_at"] = None
    job["error"] = None
    job["output_path"] = None
    # The real server assigns the log path once the job starts running; mirror
    # that so the Job Details "Download Log" button (gated on log_file_path) works.
    job["log_file_path"] = f"/mock/logs/{job['id']}.log"
    job["progress"] = 0.0
    job["progress_msg"] = ""
    job["phase"] = "starting"


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str) -> dict[str, Any]:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        _advance_job(job)
        if job["status"] == "running":
            raise HTTPException(status_code=409, detail="Job is already running")
        if job["status"] == "completed":
            raise HTTPException(status_code=409, detail="Job already completed. Delete and re-create to run again.")
        _start(job)
        return _public_job(job)


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        _advance_job(job)
        if job["status"] != "running":
            raise HTTPException(status_code=409, detail=f"Job is not running (status={job['status']})")
        job["status"] = "stopped"
        job["finished_at"] = time.time()
        job["phase"] = "stopped"
        return dict(job)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    with _state_lock:
        if _jobs.pop(job_id, None) is None:
            raise HTTPException(status_code=404, detail="Job not found")
    return {"detail": f"Job {job_id} deleted"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        _advance_job(job)
        result = _compute_logs(job)
    all_lines = result["lines"]
    return {
        "lines": all_lines[after:],
        "total": len(all_lines),
        "progress": result["progress"],
        "progress_msg": result["progress_msg"],
        "phase": result["phase"],
    }


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _advance_job(job)
        output_path = job.get("output_path")
        if job["status"] != "completed" or not output_path:
            raise HTTPException(status_code=404, detail="No output available for this job")
    # Mirror the real server: image workloads (t2i) output a .png served as an
    # image; everything else is a video.
    if output_path.endswith(".png"):
        return FileResponse(_build_mock_media("png"), media_type="image/png", filename=f"job_{job_id}.png")
    return FileResponse(_build_mock_media("mp4"), media_type="video/mp4", filename=f"job_{job_id}.mp4")


@app.get("/api/jobs/{job_id}/download_log")
def download_log(job_id: str) -> PlainTextResponse:
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _advance_job(job)
        lines = _compute_logs(job)["lines"]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


# --- Datasets ---------------------------------------------------------------


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    with _state_lock:
        datasets = sorted(_datasets.values(), key=lambda d: d["created_at"], reverse=True)
        return [_public_dataset(d) for d in datasets]


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    with _state_lock:
        dataset = _datasets.get(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return _public_dataset(dataset)


@app.post("/api/datasets", status_code=201)
def create_dataset(req: CreateDatasetRequest) -> dict[str, Any]:
    if not req.upload_path:
        raise HTTPException(status_code=400, detail="upload_path is required. Upload media files first.")
    if not req.file_names:
        raise HTTPException(status_code=400, detail="No media files found.")
    dataset_id = str(uuid.uuid4())
    dataset = {"id": dataset_id, "name": req.name, "created_at": time.time()}
    captions = {fn: (req.captions.get(fn, "") if req.captions else "") for fn in req.file_names}
    with _state_lock:
        _datasets[dataset_id] = dataset
        _dataset_files[dataset_id] = {"file_names": list(req.file_names), "captions": captions}
        return _public_dataset(dataset)


@app.get("/api/datasets/{dataset_id}/files")
def get_dataset_files(dataset_id: str) -> dict[str, Any]:
    with _state_lock:
        if dataset_id not in _datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        files = _dataset_files.get(dataset_id, {"file_names": [], "captions": {}})
        return {"file_names": list(files["file_names"]), "captions": dict(files["captions"])}


@app.put("/api/datasets/{dataset_id}/captions")
def update_dataset_caption(dataset_id: str, req: UpdateCaptionRequest) -> dict[str, str]:
    with _state_lock:
        if dataset_id not in _datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        files = _dataset_files.setdefault(dataset_id, {"file_names": [], "captions": {}})
        files["captions"][req.file_name] = req.caption
    return {"detail": "Caption updated"}


@app.get("/api/datasets/{dataset_id}/media/{file_name:path}")
def serve_dataset_media(dataset_id: str, file_name: str) -> FileResponse:
    with _state_lock:
        if dataset_id not in _datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if file_name not in _dataset_files.get(dataset_id, {}).get("file_names", []):
            raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(_build_mock_media("mp4"), media_type="video/mp4")


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    with _state_lock:
        if _datasets.pop(dataset_id, None) is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        _dataset_files.pop(dataset_id, None)
    return {"detail": f"Dataset {dataset_id} deleted"}


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="FastVideo Studio mock API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8189, help="Port number (default: 8189)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
