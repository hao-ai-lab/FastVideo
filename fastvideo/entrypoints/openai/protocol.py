# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ImageResponseData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    file_path: str | None = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageResponseData]
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int | None = 1
    quality: str | None = "auto"
    response_format: str | None = "url"  # url | b64_json
    size: str | None = "1024x1024"
    style: str | None = "vivid"
    background: str | None = "auto"  # transparent | opaque | auto
    output_format: str | None = None  # png | jpeg | webp
    user: str | None = None
    # FastVideo extensions (SGLang-compatible)
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    true_cfg_scale: float | None = None
    seed: int | None = 1024
    negative_prompt: str | None = None
    enable_teacache: bool | None = False


class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = ""
    status: str = "queued"
    progress: int = 100
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: str | None = None
    file_path: str | None = None
    completed_at: int | None = None
    error: dict[str, Any] | None = None
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: str | None = None
    reference_url: str | None = None
    model: str | None = None
    seconds: int | None = 4
    size: str | None = ""
    fps: int | None = None
    num_frames: int | None = None
    seed: int | None = 1024
    # FastVideo extensions (SGLang-compatible)
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str | None = None
    enable_teacache: bool | None = False
    output_path: str | None = None


class VideoListResponse(BaseModel):
    data: list[VideoResponse]
    object: str = "list"


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return uuid.uuid4().hex
