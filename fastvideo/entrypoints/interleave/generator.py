# SPDX-License-Identifier: Apache-2.0
"""FastVideo generator adapter for InterleaveThinker-style image calls."""

from __future__ import annotations

import base64
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from fastvideo.api.compat import (
    explicit_request_updates,
    legacy_generate_call_to_request,
    normalize_generation_request,
)
from fastvideo.api.results import GenerationResult
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.interleave.schema import (
    GeneratedImage,
    InterleaveEditRequest,
)


class ImageGeneratorBackend(Protocol):
    """Minimal image-generation backend used by the Interleave app layer."""

    def generate(
        self,
        request: InterleaveEditRequest,
        *,
        request_id: str | None = None,
    ) -> GeneratedImage:
        ...


class FastVideoImageGeneratorBackend:
    """Translate InterleaveThinker image requests into ``VideoGenerator`` calls."""

    def __init__(
        self,
        generator: Any,
        *,
        output_dir: str,
        default_request: GenerationRequest | Mapping[str, Any] | None = None,
    ) -> None:
        self.generator = generator
        self.output_dir = output_dir
        self.default_request = normalize_generation_request(default_request) if default_request is not None else None

    def generate(
        self,
        request: InterleaveEditRequest,
        *,
        request_id: str | None = None,
    ) -> GeneratedImage:
        request_id = request_id or uuid.uuid4().hex
        request_output_dir = os.path.join(self.output_dir, "interleave")
        upload_dir = os.path.join(self.output_dir, "uploads")
        os.makedirs(request_output_dir, exist_ok=True)

        input_path = None
        if request.image:
            os.makedirs(upload_dir, exist_ok=True)
            input_path = decode_base64_image_to_path(
                request.image,
                os.path.join(upload_dir, f"{request_id}_input.png"),
            )

        output_path = os.path.join(request_output_dir, f"{request_id}.png")
        generation_request = self._build_generation_request(
            request,
            output_path=output_path,
            input_image_path=input_path,
        )

        start = time.perf_counter()
        result = self.generator.generate(generation_request)
        elapsed = time.perf_counter() - start
        result = _first_generation_result(result)
        file_path = result.video_path or output_path
        if not file_path or not os.path.exists(file_path):
            raise RuntimeError(f"FastVideo generation did not produce an image at {file_path!r}")

        return GeneratedImage(
            prompt=request.prompt,
            image_base64=encode_file_to_base64(file_path),
            file_path=os.path.abspath(file_path),
            inference_time_s=result.generation_time or elapsed,
            metadata={
                "request_id": request_id,
                "input_image_path": input_path,
                "peak_memory_mb": result.peak_memory_mb,
            },
        )

    def _build_generation_request(
        self,
        request: InterleaveEditRequest,
        *,
        output_path: str,
        input_image_path: str | None,
    ) -> GenerationRequest:
        kwargs = {}
        if self.default_request is not None:
            kwargs.update(_safe_explicit_request_updates(self.default_request))

        kwargs.update({
            "num_frames": 1,
            "fps": 1,
            "save_video": True,
            "return_frames": False,
            "output_path": output_path,
        })
        if input_image_path is not None:
            kwargs["image_path"] = input_image_path
        if request.width is not None:
            kwargs["width"] = int(request.width)
        if request.height is not None:
            kwargs["height"] = int(request.height)
        if request.seed is not None:
            kwargs["seed"] = int(request.seed)
        if request.resolved_num_inference_steps() is not None:
            kwargs["num_inference_steps"] = int(request.resolved_num_inference_steps())
        if request.guidance_scale is not None:
            kwargs["guidance_scale"] = float(request.guidance_scale)
        if request.true_cfg_scale is not None:
            kwargs["true_cfg_scale"] = float(request.true_cfg_scale)
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt

        return legacy_generate_call_to_request(
            request.prompt,
            None,
            legacy_kwargs=kwargs,
        )


def _safe_explicit_request_updates(request: GenerationRequest) -> dict[str, Any]:
    try:
        return explicit_request_updates(request)
    except AssertionError:
        return explicit_request_updates(normalize_generation_request(request))


def _first_generation_result(result: GenerationResult | list[GenerationResult]) -> GenerationResult:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("FastVideo generation returned an empty result list")
        return result[0]
    return result


def encode_file_to_base64(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def decode_base64_image_to_path(
    image_base64: str,
    output_path: str | os.PathLike[str],
) -> str:
    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    data = base64.b64decode(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)
