# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py)

import asyncio
import base64
import os
import time

import aiofiles
import imageio.v2 as imageio

from fastapi import (APIRouter, File, Form, HTTPException, Path, Query, Request, UploadFile)
from fastapi.responses import FileResponse

from fastvideo.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
    generate_request_id,
)
from fastvideo.entrypoints.openai.request_adapter import (
    RequestAdaptationError,
    validate_served_model_name,
)
from fastvideo.entrypoints.openai.state import (
    get_output_dir,
    get_served_model_name,
    get_server_args,
    get_serving_engine,
)
from fastvideo.entrypoints.openai.stores import IMAGE_STORE
from fastvideo.entrypoints.openai.utils import (
    choose_image_ext,
    merge_image_input_list,
    parse_size,
    save_image_to_path,
)
from fastvideo.utils import pixels_to_uint8
from fastvideo.logger import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/images", tags=["images"])

_SUPPORTED_RESPONSE_FORMATS = frozenset({"b64_json", "url"})


def _normalize_response_format(value: str | None) -> str:
    """Lowercase and validate an OpenAI image response_format value."""
    fmt = (value or "b64_json").lower()
    if fmt not in _SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"response_format={fmt} is not supported")
    return fmt


def _validate_request_model(model: str | None) -> None:
    """Reject a model id that is not the one this server loaded."""
    if model is None:
        return
    try:
        validate_served_model_name(model, get_server_args(), get_served_model_name())
    except RequestAdaptationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _build_generation_kwargs(
    request_id: str,
    prompt: str,
    n: int = 1,
    size: str | None = None,
    output_format: str | None = None,
    background: str | None = None,
    image_path: list[str] | None = None,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    true_cfg_scale: float | None = None,
    negative_prompt: str | None = None,
    enable_teacache: bool | None = None,
) -> dict:
    """Convert API request params to VideoGenerator.generate_video kwargs"""
    kwargs: dict = {"prompt": prompt}
    if not 1 <= n <= 10:
        raise HTTPException(status_code=400, detail="n must be between 1 and 10")
    if output_format is not None and output_format.lower() not in {"png", "jpeg", "jpg", "webp"}:
        raise HTTPException(status_code=400, detail="output_format must be png, jpeg, jpg, or webp")

    if size is not None:
        w, h = parse_size(size)
        if w is None or h is None or w <= 0 or h <= 0:
            raise HTTPException(status_code=400, detail="size must contain positive WIDTHxHEIGHT dimensions")
        kwargs["width"] = w
        kwargs["height"] = h

    ext = choose_image_ext(output_format, background)
    output_dir = os.path.join(get_output_dir(), "images")
    os.makedirs(output_dir, exist_ok=True)
    kwargs["output_path"] = os.path.join(output_dir, f"{request_id}.{ext}")

    # Image generation
    kwargs["num_frames"] = 1
    kwargs["save_video"] = True
    kwargs["num_videos_per_prompt"] = n

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


async def _generate_image_files(engine, gen_kwargs: dict) -> list[str]:
    output_path = gen_kwargs["output_path"]
    count = gen_kwargs["num_videos_per_prompt"]
    if count == 1:
        await engine.run_serialized(engine.generator.generate_video, **gen_kwargs)
        return [output_path]

    # Keep one native model batch, but save separate images instead of its preview grid.
    batch_kwargs = dict(gen_kwargs,
                        save_video=False,
                        return_frames=False,
                        return_samples=True,
                        output_path=os.path.dirname(output_path))
    result = await engine.run_serialized(engine.generator.generate_video, **batch_kwargs)

    def save_images() -> list[str]:
        samples = result["samples"]
        if samples is None or samples.ndim != 5 or samples.shape[0] != count or samples.shape[2] != 1:
            raise RuntimeError("Image generation did not return the requested batch of single-frame samples")
        images = pixels_to_uint8(samples)[:, :, 0].permute(0, 2, 3, 1).cpu().numpy()
        base, ext = os.path.splitext(output_path)
        paths = []
        for index, image in enumerate(images):
            path = f"{base}_{index}{ext}"
            imageio.imwrite(path, image)
            paths.append(path)
        return paths

    return await asyncio.to_thread(save_images)


async def _build_image_response(request_id: str,
                                prompt: str,
                                resp_format: str,
                                paths: list[str],
                                elapsed: float,
                                *,
                                include_file_path: bool = False) -> ImageResponse:
    data = []
    for index, path in enumerate(paths):
        image_id = request_id if index == 0 else f"{request_id}_{index}"
        item = ImageResponseData(revised_prompt=prompt)
        if resp_format == "b64_json":
            if not os.path.exists(path):
                raise HTTPException(status_code=500, detail="Image was not saved to disk")
            async with aiofiles.open(path, "rb") as f:
                item.b64_json = base64.b64encode(await f.read()).decode("utf-8")
        else:
            item.url = f"/v1/images/{image_id}/content"
        if include_file_path or resp_format == "url":
            item.file_path = os.path.abspath(path)
        data.append(item)
        await IMAGE_STORE.upsert(image_id, {"id": image_id, "created_at": int(time.time()), "file_path": path})
    return ImageResponse(id=request_id, data=data, inference_time_s=elapsed)


@router.post("/generations", response_model=ImageResponse)
@router.post("", response_model=ImageResponse)
async def generations(request: ImageGenerationsRequest):
    resp_format = _normalize_response_format(request.response_format)
    _validate_request_model(request.model)

    request_id = generate_request_id()
    engine = get_serving_engine()

    gen_kwargs = _build_generation_kwargs(
        request_id=request_id,
        prompt=request.prompt,
        n=1 if request.n is None else request.n,
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
        paths = await _generate_image_files(engine, gen_kwargs)
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from None
    elapsed = time.perf_counter() - start

    return await _build_image_response(request_id, request.prompt, resp_format, paths, elapsed)


@router.post("/edits", response_model=ImageResponse)
async def edits(
        raw_request: Request,
        image: list[UploadFile] | None = File(None),  # noqa: B008
        image_array: list[UploadFile] | None = File(  # noqa: B008
            None, alias="image[]"),
        url: list[str] | None = Form(None),  # noqa: B008
        url_array: list[str] | None = Form(None, alias="url[]"),  # noqa: B008
        prompt: str = Form(...),
        model: str | None = Form(None),
        n: int | None = Form(1),
        response_format: str | None = Form(None),
        size: str | None = Form(None),
        output_format: str | None = Form(None),
        background: str | None = Form("auto"),
        seed: int | None = Form(1024),
        negative_prompt: str | None = Form(None),
        guidance_scale: float | None = Form(None),
        true_cfg_scale: float | None = Form(None),
        num_inference_steps: int | None = Form(None),
        enable_teacache: bool | None = Form(False),
):
    resp_format = _normalize_response_format(response_format)
    _validate_request_model(model)

    request_id = generate_request_id()
    engine = get_serving_engine()

    # Optional Form parameters normalize explicit empty strings to None.
    form = await raw_request.form()
    if form.get("size") == "":
        size = ""
    if form.get("output_format") == "":
        output_format = ""

    images = image or image_array
    urls = url or url_array
    if (not images or len(images) == 0) and (not urls or len(urls) == 0):
        raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")

    gen_kwargs = _build_generation_kwargs(
        request_id=request_id,
        prompt=prompt,
        n=1 if n is None else n,
        size=size,
        output_format=output_format,
        background=background,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        enable_teacache=enable_teacache,
    )

    # Save input images
    uploads_dir = os.path.join(get_output_dir(), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    image_list = merge_image_input_list(images, urls)

    input_paths: list[str] = []
    try:
        for idx, img in enumerate(image_list):
            filename = getattr(img, "filename", f"image_{idx}")
            input_path = await save_image_to_path(img, os.path.join(uploads_dir, f"{request_id}_{idx}_{filename}"))
            input_paths.append(input_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}") from None

    gen_kwargs["image_path"] = input_paths[0] if len(input_paths) == 1 else input_paths

    start = time.perf_counter()
    try:
        paths = await _generate_image_files(engine, gen_kwargs)
    except Exception as e:
        logger.error("Image edit failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from None
    elapsed = time.perf_counter() - start

    return await _build_image_response(request_id, prompt, resp_format, paths, elapsed, include_file_path=True)


@router.get("/{image_id}/content")
async def download_image_content(image_id: str = Path(...), variant: str | None = Query(None)):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = item.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image is still being generated")

    ext = os.path.splitext(file_path)[1].lower()
    media_type = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")

    return FileResponse(path=file_path, media_type=media_type, filename=os.path.basename(file_path))
