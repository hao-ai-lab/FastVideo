# SPDX-License-Identifier: Apache-2.0
"""FastVideo generator adapter for InterleaveThinker-style image calls."""

from __future__ import annotations

import base64
import io
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
from fastvideo.workflows.interleave_thinker.schema import (
    GeneratedImage,
    InterleaveEditRequest,
)

_NANO_BANANA_MODEL_ALIASES = {
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-pro": "gemini-3-pro-image",
    "nano-banana-2": "gemini-3.1-flash-image",
}


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


class NanoBananaImageGeneratorBackend:
    """Google Gemini API image backend for Nano Banana models.

    This wraps the closed-source Gemini native-image API behind the same
    ``ImageGeneratorBackend`` protocol used by Interleave orchestration. The SDK
    import and API-key validation are intentionally lazy so
    installing FastVideo does not require ``google-genai`` unless this backend is
    configured.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-image",
        api_key: str | None = None,
        base_url: str | None = None,
        output_dir: str = "outputs/nano_banana",
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        max_attempts: int = 3,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.model = _NANO_BANANA_MODEL_ALIASES.get(model, model)
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = output_dir
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_s = float(retry_delay_s)
        self._client: Any | None = None

    def generate(
        self,
        request: InterleaveEditRequest,
        *,
        request_id: str | None = None,
    ) -> GeneratedImage:
        request_id = request_id or uuid.uuid4().hex
        output_format = (request.output_format or "png").lower()
        if output_format == "jpg":
            output_format = "jpeg"
        output_path = Path(self.output_dir) / "interleave" / f"{request_id}.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        contents: list[Any] = [request.prompt]
        if request.image:
            contents.append(_decode_base64_to_pil(request.image))

        last_exc: Exception | None = None
        start = time.perf_counter()
        client = self._client_instance()
        generate_config = self._make_generate_config()
        for attempt in range(self.max_attempts):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generate_config,
                )
                image = _extract_first_response_image(response)
                image.save(output_path)
                return GeneratedImage(
                    prompt=request.prompt,
                    image_base64=encode_file_to_base64(output_path),
                    file_path=str(output_path.resolve()),
                    inference_time_s=time.perf_counter() - start,
                    metadata={
                        "request_id": request_id,
                        "model": self.model,
                        "attempt": attempt + 1,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - remote API errors vary by SDK version
                last_exc = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.retry_delay_s)
        raise RuntimeError(
            f"Nano Banana generation failed after {self.max_attempts} attempts: {last_exc}") from last_exc

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        genai, _ = _import_google_genai()
        kwargs: dict[str, Any] = {"api_key": _resolve_google_api_key(self.api_key)}
        if self.base_url:
            kwargs["http_options"] = {"base_url": self.base_url}
        self._client = genai.Client(**kwargs)
        return self._client

    def _make_generate_config(self) -> Any:
        _, types = _import_google_genai()
        kwargs: dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"]}
        if self.aspect_ratio or self.image_size:
            image_kwargs: dict[str, Any] = {}
            if self.aspect_ratio:
                image_kwargs["aspect_ratio"] = self.aspect_ratio
            if self.image_size:
                image_kwargs["image_size"] = self.image_size
            kwargs["image_config"] = types.ImageConfig(**image_kwargs)
        return types.GenerateContentConfig(**kwargs)


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


def _resolve_google_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    token_path = Path("~/.gemini_token").expanduser()
    if token_path.is_file():
        return token_path.read_text().strip()
    raise ValueError("Google Gemini API access requires GEMINI_API_KEY, GOOGLE_API_KEY, "
                     "an explicit api_key, or ~/.gemini_token.")


def _import_google_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Nano Banana API backend requires google-genai. "
                           "Install google-genai directly or with `uv pip install -e '.[eval-judge]'`.") from exc
    return genai, types


def _decode_base64_to_pil(image_base64: str) -> Any:
    from PIL import Image

    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _extract_first_response_image(response: Any) -> Any:
    from PIL import Image

    parts = getattr(response, "parts", None)
    if parts is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(getattr(candidates[0], "content", None), "parts", None)
    for part in parts or []:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            if isinstance(image, Image.Image):
                return image
        inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        data = getattr(inline_data, "data", None)
        if data:
            if isinstance(data, str):
                data = base64.b64decode(data)
            return Image.open(io.BytesIO(data)).convert("RGB")
    raise RuntimeError("Gemini image response did not include an image part")
