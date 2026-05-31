from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:  # pragma: no cover - optional dependency
    Cerebras = None  # type: ignore[assignment]

LOCAL_DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = (
    LOCAL_DEMO_DIR / "prompts" / "prompt_extension_system_prompt.md"
)
DEFAULT_PROVIDER = "cerebras"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_TEMPERATURE = 1.0


def _enhance_print(level: str, message: str) -> None:
    print(f"[ENHANCE][{level}] {message}", flush=True)


@dataclass
class EnhanceResult:
    prompt: str
    fallback_used: bool
    error: str | None
    provider: str
    model: str
    latency_ms: float


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _preview_text(text: str, *, limit: int = 160) -> str:
    normalized = text.replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _expand_prompt_candidate_paths(path: str) -> list[Path]:
    prompt_path = Path(path)
    candidate_paths = [prompt_path]
    if prompt_path.suffix == ".txt":
        candidate_paths.append(prompt_path.with_suffix(".md"))
    elif prompt_path.suffix == ".md":
        candidate_paths.append(prompt_path.with_suffix(".txt"))
    return candidate_paths


def _load_prompt_required(path: str, prompt_name: str) -> str:
    candidate_paths = _expand_prompt_candidate_paths(path)

    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read {prompt_name} system prompt: {candidate}"
            ) from exc
        if not text:
            raise RuntimeError(
                f"{prompt_name} system prompt file is empty: {candidate}"
            )
        return text

    tried = ", ".join(str(candidate) for candidate in candidate_paths)
    raise RuntimeError(
        f"{prompt_name} system prompt file not found. Tried: {tried}"
    )


def _extract_assistant_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices in chat completion response.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            if isinstance(text, dict):
                value = text.get("value")
                if isinstance(value, str):
                    chunks.append(value)
                    continue
            alt_text = item.get("output_text")
            if isinstance(alt_text, str):
                chunks.append(alt_text)
        if chunks:
            return "".join(chunks)

    finish_reason = choices[0].get("finish_reason")
    refusal = message.get("refusal")
    raise ValueError(
        "Missing assistant content in chat completion response. "
        f"finish_reason={finish_reason!r}, refusal={refusal!r}"
    )


def _dump_response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif isinstance(response, dict):
        payload = response
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        raise TypeError(
            "Unsupported chat completion response type. "
            f"type={type(response)!r}"
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Chat completion response did not serialize to a JSON object."
        )
    return payload


def _parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        raise ValueError("Assistant response is empty.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_pattern = re.compile(
        r"```(?:json)?\s*([\s\S]*?)```",
        flags=re.IGNORECASE,
    )
    for match in fence_pattern.finditer(text):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No JSON object found in assistant response.")


def _normalize_prompt(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


class PromptEnhancer:
    def __init__(self) -> None:
        self.provider = DEFAULT_PROVIDER
        self.provider_label = DEFAULT_PROVIDER
        self.api_key = _env_value(
            "FASTVIDEO_PROMPT_API_KEY",
            "CEREBRAS_API_KEY",
        )
        self.model = _env_value("LTX2_PROMPT_MODEL") or DEFAULT_MODEL
        self.temperature = _env_float(
            "LTX2_PROMPT_TEMPERATURE",
            DEFAULT_TEMPERATURE,
        )
        self.system_prompt_path = _env_value(
            "LTX2_PROMPT_EXTENSION_SYSTEM_PROMPT_PATH"
        ) or str(DEFAULT_SYSTEM_PROMPT_PATH)
        self.client: Any | None = None
        self.system_prompt: str | None = None
        self.unavailable_reason: str | None = None

        try:
            self.client = self._build_client()
            self.system_prompt = _load_prompt_required(
                self.system_prompt_path,
                "prompt-extension",
            )
        except Exception as exc:
            self.unavailable_reason = str(exc)
            _enhance_print(
                "WARN",
                f"Prompt enhancement unavailable: {self.unavailable_reason}",
            )

    def _build_client(self) -> Any:
        if Cerebras is None:
            raise RuntimeError(
                "Cerebras SDK is not installed. Install "
                "'cerebras-cloud-sdk' to use prompt enhancement."
            )
        if not self.api_key:
            raise RuntimeError(
                "Missing FASTVIDEO_PROMPT_API_KEY or CEREBRAS_API_KEY."
            )
        return Cerebras(api_key=self.api_key)

    def _build_body(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        user_payload = {
            "request": (
                "Expand the user prompt into one detailed prompt for a "
                "single 5-second LTX-2.3 video. Respond with valid JSON "
                'only as {"prompt": "..."}.'  # noqa: E501
            ),
            "user_prompt": user_prompt,
        }
        return {
            "model": model or self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        }

    def _request_content(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        if self.client is None:
            raise RuntimeError(
                self.unavailable_reason
                or "Prompt enhancement client is not initialized."
            )

        body = self._build_body(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )
        response = self.client.chat.completions.create(
            model=body["model"],
            messages=body["messages"],
            temperature=body["temperature"],
        )
        response_json = _dump_response_json(response)
        return response_json, _extract_assistant_content(response_json)

    def _require_prompt_field(
        self,
        parsed: dict[str, Any],
        field_name: str,
    ) -> str:
        value = parsed.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"Missing {field_name} string.")
        prompt = value.strip()
        if not prompt:
            raise ValueError(f"{field_name} is empty.")
        return prompt

    def enhance_prompt(self, prompt: str) -> EnhanceResult:
        cleaned = _normalize_prompt(prompt)
        if not cleaned:
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error="No valid prompt provided.",
                provider=self.provider_label,
                model=self.model,
                latency_ms=0.0,
            )

        if self.system_prompt is None:
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error=(
                    self.unavailable_reason
                    or "Prompt enhancement system prompt is unavailable."
                ),
                provider=self.provider_label,
                model=self.model,
                latency_ms=0.0,
            )

        t0 = time.perf_counter()
        response_content: str | None = None
        try:
            _enhance_print("INFO", f"Enhancing prompt: {cleaned}")
            _, response_content = self._request_content(
                system_prompt=self.system_prompt,
                user_prompt=cleaned,
            )
            parsed = _parse_json_response(response_content)
            enhanced_prompt = self._require_prompt_field(parsed, "prompt")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt=enhanced_prompt,
                fallback_used=False,
                error=None,
                provider=self.provider_label,
                model=self.model,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            error_detail = str(exc)
            if isinstance(response_content, str) and response_content.strip():
                error_detail = (
                    f"{error_detail} | assistant_response="
                    f"{_preview_text(response_content, limit=240)}"
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error=error_detail,
                provider=self.provider_label,
                model=self.model,
                latency_ms=latency_ms,
            )
