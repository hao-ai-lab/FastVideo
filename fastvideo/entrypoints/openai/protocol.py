# Adapted from SGLang (https://github.com/sgl-project/sglang)

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field



class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    file_path: Optional[str] = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "auto"
    response_format: Optional[str] = "url"  # url | b64_json
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    background: Optional[str] = "auto"  # transparent | opaque | auto
    output_format: Optional[str] = None  # png | jpeg | webp
    user: Optional[str] = None
    # FastVideo extensions (SGLang-compatible)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    seed: Optional[int] = 1024
    negative_prompt: Optional[str] = None
    enable_teacache: Optional[bool] = False



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
    url: Optional[str] = None
    file_path: Optional[str] = None
    completed_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: Optional[str] = None
    reference_url: Optional[str] = None
    model: Optional[str] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = 1024
    # FastVideo extensions (SGLang-compatible)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    enable_teacache: Optional[bool] = False
    output_path: Optional[str] = None


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return uuid.uuid4().hex
