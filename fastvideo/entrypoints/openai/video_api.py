# Adapted from SGLang (https://github.com/sgl-project/sglang)

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from fastvideo.entrypoints.openai.api_server import get_generator, get_server_args
from fastvideo.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
    generate_request_id,
)
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.entrypoints.openai.utils import (
    merge_image_input_list,
    parse_size,
    save_image_to_path,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])

OUTPUT_DIR = "outputs/videos"


def _build_generation_kwargs(
    request_id: str, req: VideoGenerationsRequest
) -> Dict[str, Any]:

    kwargs: Dict[str, Any] = {}
    kwargs["prompt"] = req.prompt

    # Resolution
    if req.size:
        w, h = parse_size(req.size)
        if w is not None and h is not None:
            kwargs["width"] = w
            kwargs["height"] = h

    # Frame count / duration
    fps = req.fps if req.fps is not None else 24
    kwargs["fps"] = fps

    if req.num_frames is not None:
        kwargs["num_frames"] = req.num_frames
    elif req.seconds is not None:
        kwargs["num_frames"] = fps * req.seconds

    # Sampling parameters
    if req.seed is not None:
        kwargs["seed"] = req.seed
    if req.num_inference_steps is not None:
        kwargs["num_inference_steps"] = req.num_inference_steps
    if req.guidance_scale is not None:
        kwargs["guidance_scale"] = req.guidance_scale
    if req.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = req.guidance_scale_2
    if req.negative_prompt is not None:
        kwargs["negative_prompt"] = req.negative_prompt
    if req.enable_teacache:
        kwargs["enable_teacache"] = True
    if req.true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = req.true_cfg_scale

    # Image-to-video input
    if req.input_reference is not None:
        kwargs["image_path"] = req.input_reference

    # Output path
    output_dir = req.output_path or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    kwargs["output_path"] = os.path.join(output_dir, f"{request_id}.mp4")
    kwargs["save_video"] = True

    return kwargs


def _make_video_job(
    request_id: str,
    req: VideoGenerationsRequest,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the initial job dict stored in VIDEO_STORE."""
    w = kwargs.get("width", 0)
    h = kwargs.get("height", 0)
    size_str = f"{w}x{h}" if w and h else ""
    num_frames = kwargs.get("num_frames", 0)
    fps = kwargs.get("fps", 24)
    seconds = int(round(num_frames / fps)) if fps else 0
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or get_server_args().model_path,
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": kwargs.get("output_path"),
    }


async def _run_generation(request_id: str, kwargs: Dict[str, Any]) -> None:
    """
    Run video generation in a background thread (VideoGenerator.generate_video
    is synchronous) and update the store on completion or failure.
    """
    generator = get_generator()
    loop = asyncio.get_running_loop()

    try:
        start = time.perf_counter()

        result = await loop.run_in_executor(
            None,
            lambda: generator.generate_video(**kwargs),
        )

        elapsed = time.perf_counter() - start
        update: Dict[str, Any] = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "inference_time_s": elapsed,
        }

        if isinstance(result, dict):
            gen_time = result.get("generation_time")
            if gen_time is not None:
                update["inference_time_s"] = gen_time
            peak_mem = result.get("peak_memory_mb")
            if peak_mem is not None:
                update["peak_memory_mb"] = peak_mem

        await VIDEO_STORE.update_fields(request_id, update)
        logger.info(
            "Video %s completed in %.2fs", request_id, elapsed
        )

    except Exception as e:
        logger.error("Video generation failed for %s: %s", request_id, e)
        await VIDEO_STORE.update_fields(
            request_id,
            {"status": "failed", "error": {"message": str(e)}},
        )


# Endpoints

@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(1024),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        input_path = None
        image_list = merge_image_input_list(input_reference, reference_url)
        if image_list:
            image = image_list[0]
            uploads_dir = os.path.join("outputs", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            filename = getattr(image, "filename", "url_image")
            input_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
            try:
                input_path = await save_image_to_path(image, input_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image: {e}"
                )

        extra: Dict[str, Any] = {}
        if extra_body:
            try:
                extra = json.loads(extra_body)
            except Exception:
                extra = {}

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            model=model,
            seconds=seconds if seconds is not None else 4,
            size=size,
            fps=fps if fps is not None else extra.get("fps"),
            num_frames=num_frames if num_frames is not None else extra.get("num_frames"),
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
            **({"guidance_scale": guidance_scale} if guidance_scale is not None else {}),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

        payload: Dict[str, Any] = dict(body or {})
        for key in ("extra_body", "extra_json"):
            extra = payload.pop(key, None)
            if isinstance(extra, dict):
                payload.update(extra)

        if payload.get("reference_url"):
            image_list = merge_image_input_list(payload.get("reference_url"))
            if image_list:
                image = image_list[0]
                uploads_dir = os.path.join("outputs", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                input_path = os.path.join(uploads_dir, f"{request_id}_url_image")
                try:
                    input_path = await save_image_to_path(image, input_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=400, detail=f"Failed to process image: {e}"
                    )
                payload["input_reference"] = input_path

        try:
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.info("Video generation request %s: prompt=%s", request_id, req.prompt[:100])

    gen_kwargs = _build_generation_kwargs(request_id, req)
    job = _make_video_job(request_id, req, gen_kwargs)
    await VIDEO_STORE.upsert(request_id, job)

    asyncio.create_task(_run_generation(request_id, gen_kwargs))

    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=(order != "asc"))

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1:]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    return VideoListResponse(data=[VideoResponse(**j) for j in jobs])


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(**job)


@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.pop(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        if job.get("status") == "failed":
            raise HTTPException(status_code=500, detail="Video generation failed")
        raise HTTPException(status_code=404, detail="Video still being generated")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
    )
