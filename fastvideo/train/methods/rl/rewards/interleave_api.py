# SPDX-License-Identifier: Apache-2.0
"""API-backed InterleaveThinker reward services.

These classes wrap the closed-source models used by InterleaveThinker without
making them mandatory FastVideo dependencies:

- Nano Banana / Gemini native-image models generate or edit images.
- Gemini VLM models score semantic alignment and perceptual quality.

The reusable reward parser in ``interleave_thinker.py`` remains pure; this file
contains the optional networked scorer that can be configured from YAML.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

from fastvideo.workflow.interleave_thinker.generator import (
    NanoBananaImageGeneratorBackend,
    encode_file_to_base64,
)
from fastvideo.workflow.interleave_thinker.schema import (
    InterleaveEditRequest as InterleaveAPIEditRequest, )
from fastvideo.train.methods.rl.rewards.interleave_thinker import (
    InterleaveThinkerEditRequest,
    InterleaveThinkerEditScore,
)

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_score": {
            "type": "number"
        },
        "quality_score": {
            "type": "number"
        },
        "semantic_analysis": {
            "type": "string"
        },
        "quality_analysis": {
            "type": "string"
        },
    },
    "required": ["semantic_score", "quality_score"],
}

_SCORE_SYSTEM_PROMPT = ("You are a strict image editing evaluator. Return only JSON. Scores must be "
                        "numbers from 0 to 10, where 10 is best.")

_SCORE_USER_PROMPT = """\
Evaluate the edited image for an iterative image generation/editing step.

Instruction:
{instruction}

Return:
- semantic_score: how well the edited image satisfies the instruction and preserves required content.
- quality_score: visual quality, realism/coherence, artifact absence, and logical consistency.

Use the full 0-10 range. Do not reward unrelated changes.
"""


class GeminiInterleaveImageScorer:
    """Gemini VLM scorer returning InterleaveThinker semantic/quality scores."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 6,
        retry_delay_s: float = 2.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_s = float(retry_delay_s)
        self.temperature = float(temperature)
        self._client: Any | None = None

    def score_images(
        self,
        *,
        edited_image_path: str,
        instruction: str,
        reference_image_path: str | None = None,
    ) -> InterleaveThinkerEditScore:
        prompt = _SCORE_USER_PROMPT.format(instruction=(instruction or "").strip())
        contents: list[Any] = [prompt]
        if reference_image_path:
            contents.extend(["Reference image before this step:", self._image_part(reference_image_path)])
        contents.extend(["Edited/generated image to score:", self._image_part(edited_image_path)])

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self._client_instance().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._make_score_config(),
                )
                payload = _load_jsonish(getattr(response, "text", "") or "")
                return InterleaveThinkerEditScore(
                    semantic_score=float(payload["semantic_score"]),
                    quality_score=float(payload["quality_score"]),
                )
            except Exception as exc:  # noqa: BLE001 - remote API errors vary by SDK version
                last_exc = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.retry_delay_s)
        raise RuntimeError(f"Gemini image scoring failed after {self.max_attempts} attempts: {last_exc}") from last_exc

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        genai, _ = _import_google_genai()
        kwargs: dict[str, Any] = {"api_key": _resolve_google_api_key(self.api_key)}
        if self.base_url:
            kwargs["http_options"] = {"base_url": self.base_url}
        self._client = genai.Client(**kwargs)
        return self._client

    def _make_score_config(self) -> Any:
        _, types = _import_google_genai()
        return types.GenerateContentConfig(
            system_instruction=_SCORE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SCORE_SCHEMA,
            temperature=self.temperature,
        )

    def _image_part(self, path: str) -> Any:
        _, types = _import_google_genai()
        image_path = Path(path)
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        return types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=mime_type,
        )


class GeminiNanoBananaEditScorer:
    """Generate an edit with Nano Banana and score it with Gemini.

    This callable matches ``EditScoreProvider`` from
    ``interleave_thinker.py`` and can be passed to
    ``InterleaveThinkerRewardScorer``.
    """

    def __init__(
        self,
        *,
        image_model: str = "gemini-3.1-flash-image",
        judge_model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        output_dir: str = "outputs/interleave_thinker_reward_api",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        aspect_ratio: str | None = "1:1",
        image_size: str | None = None,
        max_attempts: int = 3,
        retry_delay_s: float = 2.0,
        treat_white_canvas_as_text_to_image: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.treat_white_canvas_as_text_to_image = bool(treat_white_canvas_as_text_to_image)
        self.editor = NanoBananaImageGeneratorBackend(
            model=image_model,
            api_key=api_key,
            base_url=base_url,
            output_dir=output_dir,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
        )
        self.scorer = GeminiInterleaveImageScorer(
            model=judge_model,
            api_key=api_key,
            base_url=base_url,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
        )

    def __call__(
        self,
        request: InterleaveThinkerEditRequest,
    ) -> InterleaveThinkerEditScore | None:
        input_image_path = request.previous_image_path or request.origin_image_path
        image_base64 = None
        if input_image_path and not self._is_text_to_image_canvas(input_image_path):
            image_base64 = encode_file_to_base64(input_image_path)

        edit_request = InterleaveAPIEditRequest(
            prompt=request.refine_prompt,
            image=image_base64,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            output_format="png",
            enhance_prompt=False,
        )
        generated = self.editor.generate(
            edit_request,
            request_id=f"reward_{request.index}_{uuid.uuid4().hex}",
        )
        if not generated.file_path:
            return None
        return self.scorer.score_images(
            reference_image_path=input_image_path if input_image_path else None,
            edited_image_path=generated.file_path,
            instruction=request.origin_prompt or request.refine_prompt,
        )

    def _is_text_to_image_canvas(self, path: str) -> bool:
        if not self.treat_white_canvas_as_text_to_image:
            return False
        normalized = path.replace(os.sep, "/")
        return normalized.endswith("data/interleave/white.png") or Path(path).name == "white.png"


class ConstantInterleaveEditScorer:
    """Small deterministic scorer for smoke tests and offline debugging."""

    def __init__(
        self,
        *,
        semantic_reward: float = 0.5,
        quality_reward: float = 0.5,
    ) -> None:
        self.semantic_reward = float(semantic_reward)
        self.quality_reward = float(quality_reward)

    def __call__(
        self,
        request: InterleaveThinkerEditRequest,
    ) -> InterleaveThinkerEditScore:
        del request
        return InterleaveThinkerEditScore(
            semantic_reward=self.semantic_reward,
            quality_reward=self.quality_reward,
        )


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
    raise ValueError("Gemini API wrappers require GEMINI_API_KEY, GOOGLE_API_KEY, "
                     "an explicit api_key, or ~/.gemini_token.")


def _import_google_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini / Nano Banana API wrappers require google-genai. "
                           "Install google-genai directly or with `uv pip install -e '.[eval-judge]'`.") from exc
    return genai, types


def _load_jsonish(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini scorer returned non-object JSON: {type(payload).__name__}")
    return payload


def image_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def image_to_png_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image_bytes_to_base64(buffer.getvalue())


__all__ = [
    "ConstantInterleaveEditScorer",
    "GeminiInterleaveImageScorer",
    "GeminiNanoBananaEditScorer",
    "image_bytes_to_base64",
    "image_to_png_base64",
]
