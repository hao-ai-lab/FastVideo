# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)

import time
import uuid
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


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


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class VideoGenerationStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


SizeStr = Annotated[str, StringConstraints(pattern=r"^\d+x\d+$")]
SecondStr = Annotated[str, StringConstraints(pattern=r"^[1-9]\d*$")]
DEFAULT_FPS = 24


class VideoParams(BaseModel):
    """Optional vLLM-Omni-compatible video parameter block."""

    width: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    height: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    num_frames: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    fps: int | None = Field(default=None, ge=1, le=_INT64_MAX)

    @property
    def size(self) -> str | None:
        if self.width is not None and self.height is not None:
            return f"{self.width}x{self.height}"
        return None


class FileImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: str


class UrlImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_url: str


ImageReference = UrlImageReference | FileImageReference


class FileVideoReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: str


class UrlVideoReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_url: str


VideoReference = UrlVideoReference | FileVideoReference


class UrlAudioReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_url: str


AudioReference = UrlAudioReference


class VideoGenerationRequest(BaseModel):
    """OpenAI/vLLM-Omni-compatible video generation request.

    Model-specific parameters belong in ``extra_params``. FastVideo keeps the
    legacy ``input_reference`` and ``reference_url`` fields for clients that
    predate vLLM-Omni's typed reference objects.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str
    model: str | None = None
    seconds: int | SecondStr | None = None
    size: SizeStr | None = None
    image_reference: ImageReference | list[ImageReference] | None = None
    video_reference: VideoReference | list[VideoReference] | None = None
    audio_reference: AudioReference | list[AudioReference] | None = None
    input_reference: str | None = None
    reference_url: str | None = None
    # SGLang's legacy direct-video spellings.
    video_path: str | None = None
    video_url: str | None = None
    video_params: VideoParams | None = None
    user: str | None = None
    task: str | None = None

    width: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    height: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    fps: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    num_frames: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    aspect_ratio: str | None = None
    short_edge: int | None = Field(default=None, ge=1, le=_INT64_MAX)
    num_outputs_per_prompt: int = Field(default=1, ge=1, le=10)
    # SGLang spelling retained as an alias-like input field.
    n: int | None = Field(default=None, ge=1, le=10)
    start_time_seconds: float | None = Field(default=None, ge=0.0)
    quality: str | None = None
    negative_prompt: str | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    guidance_scale_2: float | None = Field(default=None, ge=0.0, le=20.0)
    boundary_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    flow_shift: float | None = None
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=_INT64_MIN, le=_INT64_MAX)
    generate_sound: bool = False
    sound_duration: float | None = Field(default=None, gt=0.0)
    enable_teacache: bool = False
    max_sequence_length: int | None = Field(default=None, ge=1)

    enable_frame_interpolation: bool = False
    frame_interpolation_exp: int = Field(default=1, ge=1, le=_INT64_MAX)
    frame_interpolation_scale: float = Field(default=1.0, gt=0.0)
    frame_interpolation_model_path: str | None = None

    lora: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None
    output_path: str | None = None

    def resolve_video_params(self) -> VideoParams:
        """Resolve top-level, nested, and ``size`` dimensions like vLLM-Omni."""
        params = VideoParams(
            width=self.width,
            height=self.height,
            fps=self.fps,
            num_frames=self.num_frames,
        )
        if self.video_params is not None:
            params.width = params.width or self.video_params.width
            params.height = params.height or self.video_params.height
            params.fps = params.fps or self.video_params.fps
            params.num_frames = params.num_frames or self.video_params.num_frames
        if self.size is not None:
            width, height = self.size.split("x", 1)
            params.width, params.height = int(width), int(height)
        if params.fps is None:
            params.fps = DEFAULT_FPS
        if params.num_frames is None and self.seconds is not None:
            params.num_frames = int(self.seconds) * params.fps
        return params

    @property
    def resolved_num_outputs(self) -> int:
        return self.n if self.n is not None else self.num_outputs_per_prompt


# Backward-compatible spelling used by the original FastVideo/SGLang surface.
VideoGenerationsRequest = VideoGenerationRequest


class VideoError(BaseModel):
    code: int | str = 500
    message: str


class VideoResponse(BaseModel):
    id: str
    object: Literal["video"] = "video"
    model: str = ""
    prompt: str = ""
    status: VideoGenerationStatus = VideoGenerationStatus.QUEUED
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: SizeStr | None = None
    seconds: SecondStr = "4"
    quality: str = "default"
    url: str | None = None
    remixed_from_video_id: str | None = None
    expires_at: int | None = None
    file_path: str | None = None
    file_name: str | None = None
    media_type: Literal["video/mp4"] = "video/mp4"
    completed_at: int | None = None
    error: VideoError | None = None
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None
    stage_durations: dict[str, float] = Field(default_factory=dict)


class VideoDeleteResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["video.deleted"] = "video.deleted"


class VideoListResponse(BaseModel):
    data: list[VideoResponse]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    object: Literal["list"] = "list"


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return uuid.uuid4().hex
