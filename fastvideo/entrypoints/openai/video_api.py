# SPDX-License-Identifier: Apache-2.0
"""OpenAI/vLLM-Omni-compatible video generation routes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile

from fastvideo.api.compat import explicit_request_updates, request_to_sampling_param
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.protocol import (
    VideoDeleteResponse,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
    generate_request_id,
)
from fastvideo.entrypoints.openai.request_adapter import (
    build_generation_request,
    prepare_reference_media,
    validate_model_and_lora,
)
from fastvideo.entrypoints.openai.state import (
    get_default_request,
    get_output_dir,
    get_served_model_name,
    get_server_args,
    get_serving_engine,
)
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.entrypoints.openai.utils import parse_size, save_image_to_path
from fastvideo.logger import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])

_VIDEO_JOB_TASKS: dict[str, asyncio.Task[None]] = {}
_DELETED_VIDEO_IDS: set[str] = set()
_JSON_FORM_FIELDS = {
    "image_reference",
    "video_reference",
    "audio_reference",
    "video_params",
    "lora",
    "extra_params",
}
_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def _build_generation_kwargs(
    request_id: str,
    req: VideoGenerationRequest,
    default_request: GenerationRequest | None = None,
) -> dict[str, Any]:
    """Backward-compatible flat projection used by helper-level callers.

    Runtime serving uses :func:`build_generation_request` and the typed
    ``VideoGenerator.generate`` API. Keeping this helper avoids breaking code
    that imported the original FastVideo adapter directly.
    """
    kwargs: dict[str, Any] = {}
    if default_request is not None:
        kwargs.update(explicit_request_updates(default_request))

    body_set = req.model_fields_set
    nested_set = req.video_params.model_fields_set if req.video_params is not None else set()
    kwargs["prompt"] = req.prompt
    if "size" in body_set and req.size:
        width, height = parse_size(req.size)
        if width is not None and height is not None:
            kwargs["width"], kwargs["height"] = width, height
    else:
        if "width" in body_set and req.width is not None:
            kwargs["width"] = req.width
        elif "video_params" in body_set and "width" in nested_set and req.video_params.width is not None:
            kwargs["width"] = req.video_params.width
        if "height" in body_set and req.height is not None:
            kwargs["height"] = req.height
        elif "video_params" in body_set and "height" in nested_set and req.video_params.height is not None:
            kwargs["height"] = req.video_params.height

    if "fps" in body_set and req.fps is not None:
        kwargs["fps"] = req.fps
    elif "video_params" in body_set and "fps" in nested_set and req.video_params.fps is not None:
        kwargs["fps"] = req.video_params.fps
    kwargs.setdefault("fps", 24)

    if "num_frames" in body_set and req.num_frames is not None:
        kwargs["num_frames"] = req.num_frames
    elif "video_params" in body_set and "num_frames" in nested_set and req.video_params.num_frames is not None:
        kwargs["num_frames"] = req.video_params.num_frames
    elif "seconds" in body_set and req.seconds is not None:
        kwargs["num_frames"] = int(req.seconds) * int(kwargs["fps"])

    for name in (
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "negative_prompt",
            "enable_teacache",
            "max_sequence_length",
            "boundary_ratio",
    ):
        if name in body_set and getattr(req, name) is not None:
            kwargs[name] = getattr(req, name)
    if "n" in body_set or "num_outputs_per_prompt" in body_set:
        kwargs["num_videos_per_prompt"] = req.resolved_num_outputs
    if "input_reference" in body_set and req.input_reference is not None:
        kwargs["image_path"] = req.input_reference

    kwargs.pop("output_path", None)
    output_dir = os.path.join(os.path.abspath(get_output_dir()), "videos")
    os.makedirs(output_dir, exist_ok=True)
    kwargs["output_path"] = os.path.join(output_dir, f"{request_id}.mp4")
    kwargs["save_video"] = True
    return kwargs


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _remove_artifact(file_path: str | None) -> None:
    if not file_path or not os.path.isfile(file_path):
        return
    try:
        os.unlink(file_path)
    except OSError:
        logger.warning("Failed to delete video artifact %s", file_path, exc_info=True)


def _stage_durations(result: Any) -> dict[str, float]:
    logging_info = _result_value(result, "logging_info")
    stages = getattr(logging_info, "stages", None)
    if not isinstance(stages, dict):
        return {}
    durations: dict[str, float] = {}
    for stage_name, metrics in stages.items():
        if isinstance(metrics, dict) and metrics.get("execution_time") is not None:
            durations[str(stage_name)] = float(metrics["execution_time"])
    return durations


def _make_video_job(
    request_id: str,
    req: VideoGenerationRequest,
    generation_request: GenerationRequest,
) -> dict[str, Any]:
    sampling = request_to_sampling_param(generation_request, model_path=get_server_args().model_path)
    size = f"{sampling.width}x{sampling.height}" if sampling.width and sampling.height else None
    seconds = int(round(sampling.num_frames / sampling.fps)) if sampling.fps else int(req.seconds or 4)
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or get_served_model_name(),
        "prompt": req.prompt,
        "status": VideoGenerationStatus.QUEUED,
        "progress": 0,
        "created_at": int(time.time()),
        "size": size,
        "seconds": str(max(1, seconds)),
        "quality": req.quality or "default",
        # ``file_path`` is a FastVideo compatibility extension. vLLM-Omni's
        # public job shape uses ``file_name`` after completion.
        "file_path": None,
        "_sequence": time.monotonic_ns(),
    }


async def _run_generation(
    request_id: str,
    generation_request: GenerationRequest,
) -> None:
    started = 0.0
    video_path = generation_request.output.output_path

    async def mark_started() -> None:
        nonlocal started
        started = time.perf_counter()
        await VIDEO_STORE.update_fields(
            request_id,
            {
                "status": VideoGenerationStatus.IN_PROGRESS,
                "progress": 0
            },
        )

    try:
        result = await get_serving_engine().generate(generation_request, on_start=mark_started)
        if isinstance(result, list):
            if not result:
                raise RuntimeError("FastVideo returned no generation results")
            result = result[0]
        elapsed = time.perf_counter() - started
        video_path = _result_value(result, "video_path") or generation_request.output.output_path
        generation_time = _result_value(result, "generation_time", elapsed)
        await VIDEO_STORE.update_fields(
            request_id,
            {
                "status": VideoGenerationStatus.COMPLETED,
                "progress": 100,
                "completed_at": int(time.time()),
                "file_path": video_path,
                "file_name": os.path.basename(video_path) if video_path else None,
                "inference_time_s": float(generation_time or elapsed),
                "peak_memory_mb": _result_value(result, "peak_memory_mb"),
                "stage_durations": _stage_durations(result),
            },
        )
        logger.info("Video %s completed in %.2fs", request_id, elapsed)
    except asyncio.CancelledError:
        logger.info("Video %s was cancelled", request_id)
        raise
    except Exception as error:
        logger.exception("Video generation failed for %s", request_id)
        await VIDEO_STORE.update_fields(
            request_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "error": {
                    "code": 500,
                    "message": str(error)
                },
                "inference_time_s": time.perf_counter() - started if started else 0.0,
            },
        )
    finally:
        if request_id in _DELETED_VIDEO_IDS:
            _DELETED_VIDEO_IDS.discard(request_id)
            if video_path and os.path.isfile(video_path):
                try:
                    os.unlink(video_path)
                except OSError:
                    logger.warning("Failed to clean up deleted video artifact %s", video_path, exc_info=True)


def _track_video_job(request_id: str, task: asyncio.Task[None]) -> None:
    _VIDEO_JOB_TASKS[request_id] = task

    def discard(completed: asyncio.Task[None]) -> None:
        if _VIDEO_JOB_TASKS.get(request_id) is completed:
            _VIDEO_JOB_TASKS.pop(request_id, None)

    task.add_done_callback(discard)


async def shutdown_video_jobs() -> None:
    """Cancel all transport tasks before the serving engine shuts down."""
    tasks = list(_VIDEO_JOB_TASKS.values())
    _VIDEO_JOB_TASKS.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


def _parse_json_form_value(name: str, value: Any) -> Any:
    if value is None or not isinstance(value, str) or name not in _JSON_FORM_FIELDS:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"{name} is not valid JSON") from error


async def _parse_video_request(raw_request: Request) -> VideoGenerationRequest:
    content_type = raw_request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await raw_request.form()
        payload: dict[str, Any] = {}
        for name, value in form.multi_items():
            if name == "input_reference" and isinstance(value, UploadFile):
                uploads_dir = os.path.join(get_output_dir(), "uploads")
                filename = os.path.basename(value.filename or "reference")
                target = os.path.join(uploads_dir, f"{generate_request_id()}_{filename}")
                saved_path = await save_image_to_path(value, target)
                upload_ext = os.path.splitext(filename)[1].lower()
                if (value.content_type or "").lower().startswith("video/") or upload_ext in _VIDEO_EXTENSIONS:
                    payload["video_reference"] = {"video_url": saved_path}
                else:
                    payload["input_reference"] = saved_path
                continue
            parsed = _parse_json_form_value(name, value)
            if name in payload:
                current = payload[name]
                payload[name] = current + [parsed] if isinstance(current, list) else [current, parsed]
            else:
                payload[name] = parsed
    else:
        try:
            body = await raw_request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        payload = dict(body)

    for name in ("extra_body", "extra_json"):
        extra = payload.pop(name, None)
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError as error:
                raise HTTPException(status_code=400, detail=f"{name} is not valid JSON") from error
        if extra is not None and not isinstance(extra, dict):
            raise HTTPException(status_code=400, detail=f"{name} must be a JSON object")
        if extra:
            payload.update(extra)

    try:
        return VideoGenerationRequest(**payload)
    except ValidationError as error:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {error}") from error


async def _adapt_request(request_id: str, request: VideoGenerationRequest) -> GenerationRequest:
    try:
        validate_model_and_lora(request, get_server_args(), get_served_model_name())
        await prepare_reference_media(request_id, request, get_output_dir())
        return await asyncio.to_thread(
            build_generation_request,
            request_id,
            request,
            get_server_args(),
            served_model_name=get_served_model_name(),
            output_dir=get_output_dir(),
            default_request=get_default_request(),
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("", response_model=VideoResponse)
@router.post("/generations", response_model=VideoResponse, include_in_schema=False)
async def create_video(raw_request: Request) -> VideoResponse:
    """Create an asynchronous video generation job."""
    request = await _parse_video_request(raw_request)
    request_id = f"video_gen_{generate_request_id()}"
    generation_request = await _adapt_request(request_id, request)
    job = _make_video_job(request_id, request, generation_request)
    await VIDEO_STORE.upsert(request_id, job)
    task = asyncio.create_task(_run_generation(request_id, generation_request), name=f"video-job-{request_id}")
    _track_video_job(request_id, task)
    return VideoResponse(**job)


@router.post("/sync")
async def create_video_sync(raw_request: Request) -> FileResponse:
    """Generate synchronously and return raw MP4 bytes with vLLM headers."""
    request = await _parse_video_request(raw_request)
    request_id = f"video_sync-{generate_request_id()}"
    generation_request = await _adapt_request(request_id, request)
    started = time.perf_counter()
    try:
        result = await get_serving_engine().generate(generation_request)
    except Exception as error:
        logger.exception("Sync video generation failed for %s", request_id)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {error}") from error
    if isinstance(result, list):
        if not result:
            raise HTTPException(status_code=500, detail="FastVideo returned no generation results")
        result = result[0]
    elapsed = time.perf_counter() - started
    video_path = _result_value(result, "video_path") or generation_request.output.output_path
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=500, detail="FastVideo did not produce an MP4 file")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
        headers={
            "X-Request-Id": request_id,
            "X-Model": get_served_model_name(),
            "X-Inference-Time-S": f"{elapsed:.3f}",
            "X-Stage-Durations": json.dumps(_stage_durations(result), separators=(",", ":")),
            "X-Peak-Memory-MB": f"{float(_result_value(result, 'peak_memory_mb', 0.0) or 0.0):.3f}",
        },
        background=BackgroundTask(_remove_artifact, video_path),
    )


@router.get("", response_model=VideoListResponse)
async def list_videos(
        after: str | None = Query(None),
        limit: int | None = Query(None, ge=1, le=100),
        order: str = Query("desc"),
) -> VideoListResponse:
    order = order.lower()
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda job: (job.get("created_at", 0), job.get("_sequence", 0)), reverse=order == "desc")
    if after is not None:
        index = next((i for i, job in enumerate(jobs) if job.get("id") == after), None)
        jobs = [] if index is None else jobs[index + 1:]
    has_more = limit is not None and len(jobs) > limit
    if limit is not None:
        jobs = jobs[:limit]
    responses = [VideoResponse(**job) for job in jobs]
    return VideoListResponse(
        data=responses,
        first_id=responses[0].id if responses else None,
        last_id=responses[-1].id if responses else None,
        has_more=has_more,
    )


@router.get("/{video_id}", response_model=None)
async def retrieve_video(video_id: str = Path(...)) -> VideoResponse | JSONResponse:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    response = VideoResponse(**job)
    if response.status is VideoGenerationStatus.FAILED:
        return JSONResponse(status_code=500, content=response.model_dump(mode="json"))
    return response


@router.delete("/{video_id}", response_model=VideoDeleteResponse)
async def delete_video(video_id: str = Path(...)) -> VideoDeleteResponse:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    task = _VIDEO_JOB_TASKS.get(video_id)
    status = VideoGenerationStatus(job.get("status", VideoGenerationStatus.QUEUED))
    if task is not None and not task.done():
        # The current synchronous generator cannot abort a CUDA call once it
        # starts. Remove the API resource immediately and let the tracked task
        # clean up its artifact on exit while the serving lock stays held.
        if status is VideoGenerationStatus.QUEUED:
            task.cancel()
        else:
            _DELETED_VIDEO_IDS.add(video_id)
            task.cancel()
    popped = await VIDEO_STORE.pop(video_id)
    file_path = None if popped is None else popped.get("file_path")
    if status in {VideoGenerationStatus.COMPLETED, VideoGenerationStatus.FAILED}:
        _remove_artifact(file_path)
    return VideoDeleteResponse(id=video_id, deleted=True)


@router.get("/{video_id}/content")
async def download_video_content(video_id: str = Path(...), variant: str | None = Query(None)) -> FileResponse:
    del variant
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    status = VideoGenerationStatus(job.get("status", VideoGenerationStatus.QUEUED))
    if status is VideoGenerationStatus.FAILED:
        raise HTTPException(status_code=422, detail="Video generation failed. Check job status for error details.")
    file_path = job.get("file_path")
    if status is not VideoGenerationStatus.COMPLETED or not file_path:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generated video file not found on disk")
    return FileResponse(path=file_path, media_type="video/mp4", filename=os.path.basename(file_path))


__all__ = [
    "_build_generation_kwargs",
    "create_video",
    "create_video_sync",
    "delete_video",
    "download_video_content",
    "list_videos",
    "retrieve_video",
    "router",
    "shutdown_video_jobs",
]
