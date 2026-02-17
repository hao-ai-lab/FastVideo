# Adapted from SGLang (https://github.com/sgl-project/sglang)

import asyncio
import base64
import os
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse

from fastvideo.entrypoints.openai.api_server import get_generator, get_server_args
from fastvideo.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
    generate_request_id,
)
from fastvideo.entrypoints.openai.stores import IMAGE_STORE
from fastvideo.entrypoints.openai.utils import (
    choose_image_ext,
    merge_image_input_list,
    parse_size,
    save_image_to_path,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/images", tags=["images"])

OUTPUT_DIR = "outputs/images"


def _build_generation_kwargs(
    request_id: str,
    prompt: str,
    n: int = 1,
    size: Optional[str] = None,
    output_format: Optional[str] = None,
    background: Optional[str] = None,
    image_path: Optional[list[str]] = None,
    seed: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    true_cfg_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    enable_teacache: Optional[bool] = None,
) -> dict:
    """Convert API request params to VideoGenerator.generate_video kwargs"""
    kwargs: dict = {"prompt": prompt}

    if size:
        w, h = parse_size(size)
        if w is not None and h is not None:
            kwargs["width"] = w
            kwargs["height"] = h

    ext = choose_image_ext(output_format, background)
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    kwargs["output_path"] = os.path.join(output_dir, f"{request_id}.{ext}")

    # Image generation
    kwargs["num_frames"] = 1
    kwargs["save_video"] = True
    kwargs["num_videos_per_prompt"] = max(1, min(n, 10))

    if seed is not None:
        kwargs["seed"] = seed
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = true_cfg_scale
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if enable_teacache:
        kwargs["enable_teacache"] = True
    if image_path:
        kwargs["image_path"] = image_path[0] if len(image_path) == 1 else image_path

    return kwargs


@router.post("", response_model=ImageResponse)
async def generations(request: ImageGenerationsRequest):
    request_id = generate_request_id()
    generator = get_generator()
    loop = asyncio.get_running_loop()

    gen_kwargs = _build_generation_kwargs(
        request_id=request_id,
        prompt=request.prompt,
        n=request.n or 1,
        size=request.size,
        output_format=request.output_format,
        background=request.background,
        seed=request.seed,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
    )

    start = time.perf_counter()
    try:
        result = await loop.run_in_executor(
            None, lambda: generator.generate_video(**gen_kwargs)
        )
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.perf_counter() - start

    save_file_path = gen_kwargs["output_path"]

    resp_format = (request.response_format or "b64_json").lower()
    if resp_format == "b64_json":
        if not os.path.exists(save_file_path):
            raise HTTPException(status_code=500, detail="Image was not saved to disk")
        with open(save_file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        data = [ImageResponseData(b64_json=b64_data, revised_prompt=request.prompt)]
    elif resp_format == "url":
        data = [
            ImageResponseData(
                url=f"/v1/images/{request_id}/content",
                revised_prompt=request.prompt,
                file_path=os.path.abspath(save_file_path),
            )
        ]
    else:
        raise HTTPException(
            status_code=400, detail=f"response_format={resp_format} is not supported"
        )

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        },
    )

    return ImageResponse(
        id=request_id,
        data=data,
        inference_time_s=elapsed,
    )


@router.post("/edits", response_model=ImageResponse)
async def edits(
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    url: Optional[List[str]] = Form(None),
    url_array: Optional[List[str]] = Form(None, alias="url[]"),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    seed: Optional[int] = Form(1024),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
):
    request_id = generate_request_id()
    generator = get_generator()
    loop = asyncio.get_running_loop()

    images = image or image_array
    urls = url or url_array
    if (not images or len(images) == 0) and (not urls or len(urls) == 0):
        raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")

    # Save input images
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    image_list = merge_image_input_list(images, urls)

    input_paths: list[str] = []
    try:
        for idx, img in enumerate(image_list):
            filename = getattr(img, "filename", f"image_{idx}")
            input_path = await save_image_to_path(
                img, os.path.join(uploads_dir, f"{request_id}_{idx}_{filename}")
            )
            input_paths.append(input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    gen_kwargs = _build_generation_kwargs(
        request_id=request_id,
        prompt=prompt,
        n=n or 1,
        size=size,
        output_format=output_format,
        background=background,
        image_path=input_paths,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        enable_teacache=enable_teacache,
    )

    start = time.perf_counter()
    try:
        result = await loop.run_in_executor(
            None, lambda: generator.generate_video(**gen_kwargs)
        )
    except Exception as e:
        logger.error("Image edit failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = time.perf_counter() - start

    save_file_path = gen_kwargs["output_path"]

    resp_format = (response_format or "b64_json").lower()
    if resp_format == "b64_json":
        with open(save_file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        data = [
            ImageResponseData(
                b64_json=b64_data,
                revised_prompt=prompt,
                file_path=os.path.abspath(save_file_path),
            )
        ]
    else:
        data = [
            ImageResponseData(
                url=f"/v1/images/{request_id}/content",
                revised_prompt=prompt,
                file_path=os.path.abspath(save_file_path),
            )
        ]

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        },
    )

    return ImageResponse(id=request_id, data=data, inference_time_s=elapsed)


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = item.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image is still being generated")

    ext = os.path.splitext(file_path)[1].lower()
    media_type = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")

    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
