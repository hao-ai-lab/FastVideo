# SPDX-License-Identifier: Apache-2.0
"""Translate OpenAI/vLLM-Omni requests into FastVideo's typed request API."""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from fastvideo.api.compat import (
    REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS,
    explicit_request_updates,
    legacy_generate_call_to_request,
    request_to_sampling_param,
)
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.protocol import (
    FileImageReference,
    FileVideoReference,
    UrlImageReference,
    VideoGenerationRequest,
)
from fastvideo.entrypoints.openai.utils import save_image_to_path
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_image
from fastvideo.registry import get_preset_selection


class RequestAdaptationError(ValueError):
    """The transport request cannot be represented by the loaded pipeline."""


def _as_list(value: Any | list[Any] | None) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _image_sources(request: VideoGenerationRequest) -> list[str]:
    sources: list[str] = []
    for reference in _as_list(request.image_reference):
        if isinstance(reference, FileImageReference):
            raise RequestAdaptationError("file_id image references are not supported; provide image_url instead")
        sources.append(reference.image_url)
    legacy_sources = [source for source in (request.input_reference, request.reference_url) if source]
    if len(legacy_sources) > 1:
        raise RequestAdaptationError("Provide only one of input_reference or reference_url.")
    legacy = legacy_sources[0] if legacy_sources else None
    if legacy is not None:
        if sources:
            raise RequestAdaptationError("Provide only one of input_reference/reference_url or image_reference.")
        sources.append(legacy)
    return sources


async def prepare_reference_media(
    request_id: str,
    request: VideoGenerationRequest,
    output_dir: str,
) -> None:
    """Materialize and decode image references before a job reaches workers."""
    _image_sources(request)
    uploads_dir = os.path.join(os.path.abspath(output_dir), "uploads")

    async def materialize(source: str, index: int) -> str:
        source = source.strip()
        if not source:
            raise RequestAdaptationError("Image references must not be empty.")
        if source.lower().startswith(("http://", "https://", "data:image")):
            target = os.path.join(uploads_dir, f"{request_id}_{index}")
            try:
                local_path = await save_image_to_path(source, target)
            except Exception as error:
                raise RequestAdaptationError(f"Unable to fetch image reference: {error}") from error
        else:
            local_path = os.path.abspath(os.path.expanduser(source))
            if not os.path.isfile(local_path):
                raise RequestAdaptationError(f"Image reference does not exist: {source}")
        try:
            await asyncio.to_thread(load_image, local_path)
        except Exception as error:
            raise RequestAdaptationError(f"Unable to decode image reference: {error}") from error
        return local_path

    index = 0
    references = _as_list(request.image_reference)
    for reference in references:
        if isinstance(reference, UrlImageReference):
            reference.image_url = await materialize(reference.image_url, index)
            index += 1
    if request.input_reference:
        request.input_reference = await materialize(request.input_reference, index)
        index += 1
    elif request.input_reference == "":
        request.input_reference = None
    if request.reference_url:
        request.reference_url = await materialize(request.reference_url, index)
    elif request.reference_url == "":
        request.reference_url = None


def _video_sources(request: VideoGenerationRequest) -> list[str]:
    sources: list[str] = []
    for reference in _as_list(request.video_reference):
        if isinstance(reference, FileVideoReference):
            raise RequestAdaptationError("file_id video references are not supported; provide video_url instead")
        sources.append(reference.video_url)
    direct = request.video_path or request.video_url
    if direct is not None:
        if sources:
            raise RequestAdaptationError("Provide only one of video_reference or video_path/video_url.")
        sources.append(direct)
    return sources


def _audio_sources(request: VideoGenerationRequest) -> list[str]:
    return [reference.audio_url for reference in _as_list(request.audio_reference)]


def _parse_aspect_ratio(value: str) -> tuple[float, float]:
    try:
        width, height = value.split(":", 1)
        result = float(width), float(height)
    except (AttributeError, TypeError, ValueError) as error:
        raise RequestAdaptationError(f"Invalid aspect_ratio {value!r}; expected WIDTH:HEIGHT") from error
    if not all(math.isfinite(term) for term in result) or result[0] <= 0 or result[1] <= 0:
        raise RequestAdaptationError(f"Invalid aspect_ratio {value!r}; both terms must be positive")
    return result


def _apply_aspect_ratio(
    kwargs: dict[str, Any],
    request: VideoGenerationRequest,
    *,
    model_family: str | None,
) -> None:
    if request.aspect_ratio is None:
        return
    aspect_width, aspect_height = _parse_aspect_ratio(request.aspect_ratio)
    if model_family == "minimax_h3":
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_SHORT_EDGE, resolve_canvas_size

        if request.short_edge is not None and request.short_edge != MINIMAX_H3_SHORT_EDGE:
            raise RequestAdaptationError(
                f"MiniMax-H3 currently uses a fixed short_edge={MINIMAX_H3_SHORT_EDGE}, got {request.short_edge}.")
        try:
            height, width = resolve_canvas_size(aspect_width, aspect_height)
        except ValueError as error:
            raise RequestAdaptationError(str(error)) from error
    elif request.short_edge is not None:
        if aspect_width >= aspect_height:
            height = request.short_edge
            width = round(request.short_edge * aspect_width / aspect_height)
        else:
            width = request.short_edge
            height = round(request.short_edge * aspect_height / aspect_width)
    else:
        return
    kwargs["width"], kwargs["height"] = width, height


def validate_model_and_lora(
    request: VideoGenerationRequest,
    args: FastVideoArgs,
    served_model_name: str,
) -> None:
    """Validate vLLM-style model and LoRA selectors against startup state.

    FastVideo's published FastH3 adapters include dense replacement tensors in
    addition to low-rank factors. Those tensors are applied while the model is
    loaded and cannot be swapped safely between concurrent requests. The API
    accepts vLLM's selector shape, but it must identify the startup adapter.
    """
    allowed_models = {args.lora_nickname} if args.lora_path else {served_model_name}
    if request.model is not None and request.model not in allowed_models:
        choices = ", ".join(sorted(allowed_models))
        raise RequestAdaptationError(
            f"Model mismatch: request specifies {request.model!r}; this server provides {choices}.")

    if request.lora is None:
        return
    if not args.lora_path:
        raise RequestAdaptationError(
            "This server has no startup LoRA. Configure generator.pipeline.components.lora_path before using "
            "the request lora selector.")

    body = request.lora
    name = next((body[key] for key in ("name", "lora_name", "adapter") if body.get(key) is not None), None)
    path = next((body[key] for key in ("path", "lora_path", "local_path") if body.get(key) is not None), None)
    scale = next((body[key] for key in ("scale", "lora_scale") if body.get(key) is not None), None)
    if name is None and path is None:
        raise RequestAdaptationError("lora must provide a name or path")
    if name is not None and str(name) != args.lora_nickname:
        raise RequestAdaptationError(f"Requested LoRA {name!r} is not the startup adapter {args.lora_nickname!r}.")
    if path is not None and str(path) != args.lora_path:
        raise RequestAdaptationError(
            f"Requested LoRA path {path!r} does not match the startup adapter {args.lora_path!r}.")
    if scale is not None:
        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as error:
            raise RequestAdaptationError(f"Invalid LoRA scale {scale!r}") from error
        if not math.isclose(scale_value, args.lora_strength, rel_tol=0.0, abs_tol=1e-8):
            raise RequestAdaptationError(
                f"Requested LoRA scale {scale_value:g} does not match startup strength {args.lora_strength:g}.")


def _apply_reference_inputs(
    kwargs: dict[str, Any],
    request: VideoGenerationRequest,
    args: FastVideoArgs,
    *,
    model_family: str | None,
) -> None:
    images = _image_sources(request)
    videos = _video_sources(request)
    audios = _audio_sources(request)
    ref2va = model_family == "minimax_h3" and "ref2va" in (args.override_pipeline_cls_name or "").lower()

    if model_family == "minimax_h3" and request.task is not None:
        normalized_task = request.task.lower()
        if normalized_task not in {"t2va", "fl2va", "ref2va"}:
            raise RequestAdaptationError("MiniMax-H3 task must be one of t2va, fl2va, or ref2va.")
        if normalized_task == "ref2va" and not ref2va:
            raise RequestAdaptationError(
                "MiniMax-H3 task='ref2va' requires MiniMaxH3Ref2VAModularPipeline at server startup.")
        if normalized_task != "ref2va" and ref2va:
            raise RequestAdaptationError(
                f"This server is configured for MiniMax-H3 Ref2VA, not task={normalized_task!r}.")
        if normalized_task == "t2va" and (images or videos or audios):
            raise RequestAdaptationError("MiniMax-H3 task='t2va' does not accept reference media.")
        if normalized_task == "fl2va" and not images:
            raise RequestAdaptationError("MiniMax-H3 task='fl2va' requires one or two image references.")

    if ref2va:
        from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference
        from fastvideo.pipelines.basic.minimax_h3.reference import validate_references

        references = [MiniMaxH3Reference(source=source, media_type="image") for source in images]
        references.extend(MiniMaxH3Reference(source=source, media_type="video") for source in videos)
        references.extend(MiniMaxH3Reference(source=source, media_type="audio") for source in audios)
        if references:
            try:
                kwargs["references"] = validate_references(references)
            except (TypeError, ValueError) as error:
                raise RequestAdaptationError(str(error)) from error
        return

    if request.task is not None and model_family != "minimax_h3":
        raise RequestAdaptationError("The task selector is only defined for MiniMax-H3 servers.")

    if model_family == "minimax_h3" and (videos or audios):
        raise RequestAdaptationError("MiniMax-H3 video/audio references require a server configured with "
                                     "override_pipeline_cls_name=MiniMaxH3Ref2VAModularPipeline.")
    if len(images) > 2:
        raise RequestAdaptationError("The loaded pipeline accepts at most first and last image references.")
    if images:
        kwargs["image_path"] = images[0]
    if len(images) == 2:
        kwargs["last_image"] = load_image(images[1])
    if len(videos) > 1:
        raise RequestAdaptationError("The loaded pipeline accepts at most one video reference.")
    if videos:
        kwargs["video_path"] = videos[0]
    if audios:
        raise RequestAdaptationError("The loaded pipeline does not accept audio reference inputs.")


def build_generation_request(
    request_id: str,
    request: VideoGenerationRequest,
    args: FastVideoArgs,
    *,
    served_model_name: str,
    output_dir: str,
    default_request: GenerationRequest | None = None,
) -> GenerationRequest:
    """Build one tracked FastVideo request using explicit-field precedence."""
    validate_model_and_lora(request, args, served_model_name)
    kwargs: dict[str, Any] = {}
    if default_request is not None:
        kwargs.update(explicit_request_updates(default_request))

    body_set = request.model_fields_set
    nested_set = request.video_params.model_fields_set if request.video_params is not None else set()
    if "size" in body_set and request.size is not None:
        width, height = request.size.split("x", 1)
        kwargs["width"], kwargs["height"] = int(width), int(height)
    else:
        if "width" in body_set and request.width is not None:
            kwargs["width"] = request.width
        elif "video_params" in body_set and "width" in nested_set and request.video_params.width is not None:
            kwargs["width"] = request.video_params.width
        if "height" in body_set and request.height is not None:
            kwargs["height"] = request.height
        elif "video_params" in body_set and "height" in nested_set and request.video_params.height is not None:
            kwargs["height"] = request.video_params.height

    fps_explicit = ("fps" in body_set
                    and request.fps is not None) or ("video_params" in body_set and "fps" in nested_set
                                                     and request.video_params.fps is not None)
    if fps_explicit:
        fps = request.fps if "fps" in body_set else request.video_params.fps
        if fps is not None:
            kwargs["fps"] = fps
    kwargs.setdefault("fps", 24)

    frames_explicit = ("num_frames" in body_set
                       and request.num_frames is not None) or ("video_params" in body_set and "num_frames" in nested_set
                                                               and request.video_params.num_frames is not None)
    if frames_explicit:
        num_frames = request.num_frames if "num_frames" in body_set else request.video_params.num_frames
        if num_frames is not None:
            kwargs["num_frames"] = num_frames
    elif "seconds" in body_set and request.seconds is not None:
        kwargs["num_frames"] = int(request.seconds) * int(kwargs["fps"])

    direct_fields = (
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "negative_prompt",
        "enable_teacache",
        "max_sequence_length",
        "boundary_ratio",
    )
    for name in direct_fields:
        if name in body_set:
            value = getattr(request, name)
            if value is not None:
                kwargs[name] = value
    if "n" in body_set or "num_outputs_per_prompt" in body_set:
        kwargs["num_videos_per_prompt"] = request.resolved_num_outputs

    try:
        _, model_family = get_preset_selection(args.model_path)
    except (RuntimeError, ValueError):
        model_family = None
    if request.resolved_num_outputs != 1:
        raise RequestAdaptationError("FastVideo serving currently supports exactly one video output per request.")
    if "short_edge" in body_set and request.short_edge is not None and request.aspect_ratio is None:
        raise RequestAdaptationError("short_edge requires aspect_ratio.")
    _apply_aspect_ratio(kwargs, request, model_family=model_family)
    _apply_reference_inputs(kwargs, request, args, model_family=model_family)

    extension_fields = ("flow_shift", "sound_duration", "start_time_seconds")
    for name in extension_fields:
        if name in body_set and getattr(request, name) is not None:
            kwargs[name] = getattr(request, name)
    if "generate_sound" in body_set and request.generate_sound and model_family != "minimax_h3":
        kwargs["generate_sound"] = True
    if "enable_frame_interpolation" in body_set and request.enable_frame_interpolation:
        kwargs["enable_frame_interpolation"] = True
        for name in (
                "frame_interpolation_exp",
                "frame_interpolation_scale",
                "frame_interpolation_model_path",
        ):
            kwargs[name] = getattr(request, name)
    if request.extra_params:
        unknown_extra_params = sorted(set(request.extra_params) - set(REQUEST_BATCH_EXTRA_PASSTHROUGH_FIELDS))
        if unknown_extra_params:
            raise RequestAdaptationError("Unsupported extra_params fields: " + ", ".join(unknown_extra_params))
        kwargs.update(request.extra_params)

    width = kwargs.get("width")
    height = kwargs.get("height")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise RequestAdaptationError(f"width must be a positive integer, got {width!r}")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise RequestAdaptationError(f"height must be a positive integer, got {height!r}")

    if model_family == "minimax_h3":
        from fastvideo.pipelines.basic.minimax_h3.packing import (
            MINIMAX_H3_CANVAS_MULTIPLE,
            MINIMAX_H3_MAX_PIXELS,
        )
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
            resolve_target_num_frames, )

        if kwargs["fps"] != 24:
            raise RequestAdaptationError(f"MiniMax-H3 requires fps=24, got {kwargs['fps']}.")
        if width is None or height is None:
            raise RequestAdaptationError("MiniMax-H3 requires both width and height.")
        if width % MINIMAX_H3_CANVAS_MULTIPLE or height % MINIMAX_H3_CANVAS_MULTIPLE:
            raise RequestAdaptationError("MiniMax-H3 width and height must be positive multiples of "
                                         f"{MINIMAX_H3_CANVAS_MULTIPLE}, got {width}x{height}.")
        if width * height > MINIMAX_H3_MAX_PIXELS:
            raise RequestAdaptationError(
                f"MiniMax-H3 canvas exceeds the {MINIMAX_H3_MAX_PIXELS}-pixel limit: {width}x{height}.")
        try:
            requested_num_frames = kwargs.get("num_frames")
            aligned_num_frames = resolve_target_num_frames(requested_num_frames)
        except (TypeError, ValueError) as error:
            raise RequestAdaptationError(str(error)) from error
        if frames_explicit and aligned_num_frames != requested_num_frames:
            raise RequestAdaptationError("MiniMax-H3 num_frames must be on the causal-VAE grid (17 * n + 5); "
                                         f"got {requested_num_frames}, next valid value is {aligned_num_frames}.")
        kwargs["num_frames"] = aligned_num_frames

    output_path = os.path.join(os.path.abspath(output_dir), "videos", f"{request_id}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kwargs.update({
        "output_path": output_path,
        "save_video": True,
        "return_frames": False,
    })
    generation_request = legacy_generate_call_to_request(request.prompt, None, legacy_kwargs=kwargs)
    try:
        # Resolve once at admission time so unsupported model-specific fields
        # are a deterministic 400, rather than an asynchronous failed job.
        request_to_sampling_param(generation_request, model_path=args.model_path)
    except (TypeError, ValueError) as error:
        raise RequestAdaptationError(str(error)) from error
    return generation_request


__all__ = [
    "RequestAdaptationError",
    "build_generation_request",
    "prepare_reference_media",
    "validate_model_and_lora",
]
