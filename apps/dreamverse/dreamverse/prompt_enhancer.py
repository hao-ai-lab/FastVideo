from __future__ import annotations
# pyright: reportArgumentType=false
# ruff: noqa: B023,UP038
# mypy: ignore-errors

import asyncio
import json
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:  # pragma: no cover - optional dependency
    Cerebras = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from dreamverse.config import (
    PROMPT_AUTO_SYSTEM_PROMPT_FALLBACK_PATH,
    PROMPT_AUTO_SYSTEM_PROMPT_PATH,
    PROMPT_ENHANCE_SYSTEM_PROMPT_PATH,
    PROMPT_ENHANCE_SYSTEM_PROMPT_FALLBACK_PATH,
    PROMPT_API_KEY,
    PROMPT_API_KEYS,
    PROMPT_PROVIDER_API_KEY_NAMES,
    PROMPT_API_BASE_URL,
    PROMPT_API_BASE_URLS,
    PROMPT_HTTP_TIMEOUT_MS,
    PROMPT_INITIAL_STAGE_TIMEOUT_MS,
    PROMPT_MAX_COMPLETION_TOKENS,
    PROMPT_MODEL,
    PROMPT_PROVIDER,
    PROMPT_PROVIDER_MODELS,
    PROMPT_PROVIDER_PRIORITY,
    PROMPT_PROVIDER_RUNTIME_STAGES,
    PROMPT_REWRITE_MODEL,
    PROMPT_REWRITE_MODEL_OPTIONS,
    PROMPT_REWRITE_ALL_SYSTEM_PROMPT_FALLBACK_PATH,
    PROMPT_REWRITE_ALL_SYSTEM_PROMPT_PATH,
    PROMPT_REWRITE_USER_SYSTEM_PROMPT_FALLBACK_PATH,
    PROMPT_REWRITE_USER_SYSTEM_PROMPT_PATH,
    PROMPT_TEMPERATURE,
    PROMPT_TIMEOUT_MS,
)
from dreamverse.rewrite_prompt_payload import (
    build_rewrite_request_body,
    DEFAULT_REWRITE_ROLLOUT_ID,
    DEFAULT_REWRITE_ROLLOUT_LABEL,
    DEFAULT_REWRITE_SEGMENT_COUNT,
    normalize_prompt_window_prompts,
)


def _enhance_print(level: str, message: str):
    print(f"[ENHANCE][{level}] {message}", flush=True)


@dataclass
class EnhanceResult:
    prompt: str
    fallback_used: bool
    error: str | None
    provider: str
    model: str
    latency_ms: float


@dataclass
class RewriteResult:
    prompts: list[str]
    fallback_used: bool
    error: str | None
    provider: str
    model: str
    latency_ms: float
    rollout_id: str
    rollout_label: str
    raw_response_text: str | None = None


@dataclass(frozen=True)
class ProviderRuntime:
    name: str
    api_key: str
    api_base_url: str | None
    request_model: str
    client: Any


def _resolve_provider_family(provider_name: str) -> str:
    if provider_name == "cerebras":
        return "cerebras"
    if provider_name == "groq":
        return "groq"
    raise RuntimeError(f"Unsupported prompt provider: {provider_name}")


def _resolve_provider_label(provider_name: str) -> str:
    return _resolve_provider_family(provider_name)


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


def _load_prompt_required(
    path: str,
    prompt_name: str,
    fallback_path: str | None = None,
) -> str:
    text, _ = _load_prompt_required_with_path(
        path,
        prompt_name,
        fallback_path,
    )
    return text


def _load_prompt_required_with_path(
    path: str,
    prompt_name: str,
    fallback_path: str | None = None,
) -> tuple[str, str]:
    candidate_paths: list[Path] = []
    for current_path in [path, fallback_path]:
        if not current_path:
            continue
        for candidate in _expand_prompt_candidate_paths(current_path):
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            text = _normalize_prompt_file_text(candidate.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Failed to read {prompt_name} system prompt: {candidate}") from exc
        if not text:
            raise RuntimeError(f"{prompt_name} system prompt file is empty: {candidate}")
        return text, str(candidate)

    tried = ", ".join(str(candidate) for candidate in candidate_paths)
    raise RuntimeError(f"{prompt_name} system prompt file not found. Tried: {tried}")


def _load_prompt_with_prompt_fallback(
    path: str,
    prompt_name: str,
    fallback_prompt_text: str,
    fallback_prompt_source_path: str,
    fallback_path: str | None = None,
) -> tuple[str, str]:
    candidate_paths: list[Path] = []
    for current_path in [path, fallback_path]:
        if not current_path:
            continue
        for candidate in _expand_prompt_candidate_paths(current_path):
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)

    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            text = _normalize_prompt_file_text(candidate.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Failed to read {prompt_name} system prompt: {candidate}") from exc
        if text:
            return text, str(candidate)

    return fallback_prompt_text, fallback_prompt_source_path


def _normalize_prompt_file_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    wrapper_match = re.match(
        r"^SYSTEM_PROMPT\s*=\s*(?P<quote>'''|\"\"\")(?P<body>[\s\S]*?)(?P=quote)\s*$",
        normalized,
    )
    if wrapper_match:
        return wrapper_match.group("body").strip()

    return normalized


def _save_prompt(path: str, prompt_text: str, prompt_name: str) -> None:
    prompt_path = Path(path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_prompt_text = prompt_text.strip() + "\n"
    try:
        if prompt_path.is_file():
            current_text = prompt_path.read_text(encoding="utf-8")
            if current_text != normalized_prompt_text:
                backup_path = _build_prompt_backup_path(prompt_path)
                shutil.copy2(prompt_path, backup_path)
        prompt_path.write_text(normalized_prompt_text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to save {prompt_name} system prompt: {prompt_path}") from exc


def _build_prompt_backup_path(prompt_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = "".join(prompt_path.suffixes)
    stem = (prompt_path.name[:-len(suffix)] if suffix else prompt_path.name)
    return prompt_path.with_name(f"{stem}.{timestamp}.bak{suffix}")


def _resolve_prompt_save_path(
    path: str,
    fallback_path: str | None = None,
) -> str:
    candidate = Path(path)
    if candidate.is_file() or fallback_path is None:
        return str(candidate)

    for fallback_candidate in _expand_prompt_candidate_paths(fallback_path):
        if fallback_candidate.is_file():
            return str(fallback_candidate)
    return str(candidate)


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
            if isinstance(item, dict):
                # OpenAI-compatible providers may use different content item
                # shapes; keep this extractor permissive.
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
    raise ValueError("Missing assistant content in chat completion response. "
                     f"finish_reason={finish_reason!r}, refusal={refusal!r}")


def _dump_response_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif isinstance(response, dict):
        payload = response
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        raise TypeError("Unsupported chat completion response type. "
                        f"type={type(response)!r}")

    if not isinstance(payload, dict):
        raise TypeError("Chat completion response did not serialize to a JSON object.")
    return payload


def _extract_content_or_empty(response_json: dict[str, Any]) -> str:
    try:
        return _extract_assistant_content(response_json)
    except Exception as exc:
        choices = response_json.get("choices")
        choice0 = choices[0] if isinstance(choices, list) and choices else {}
        message = (choice0.get("message", {}) if isinstance(choice0, dict) else {})
        usage = response_json.get("usage")
        _enhance_print(
            "WARN",
            "Failed to extract assistant content: "
            f"{exc}; finish_reason={choice0.get('finish_reason')!r}; "
            f"usage={usage!r}; message_keys="
            f"{list(message.keys()) if isinstance(message, dict) else 'n/a'}",
        )
        return ""


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

    # Handle fenced blocks like ```json ... ``` with optional prose around it.
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

    # Fall back to scanning for the first decodable JSON object in free-form text.
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


def _resolve_rollout_id(value: Any) -> str:
    normalized = _normalize_prompt(value)
    return normalized or DEFAULT_REWRITE_ROLLOUT_ID


def _resolve_rollout_label(value: Any) -> str:
    normalized = _normalize_prompt(value)
    return normalized or DEFAULT_REWRITE_ROLLOUT_LABEL


def _format_locked_segments(locked_segments: list[str]) -> str:
    if not locked_segments:
        return "(none)"
    return "\n".join(f'segment_{i + 1} ({i * 5}-{(i + 1) * 5}s): "{segment}"'
                     for i, segment in enumerate(locked_segments))


class PromptEnhancer:

    def __init__(self):
        self.provider = PROMPT_PROVIDER
        self.provider_label = _resolve_provider_label(PROMPT_PROVIDER)
        self.api_key = PROMPT_API_KEY
        self.api_base_url = PROMPT_API_BASE_URL
        self.model = PROMPT_MODEL
        self.provider_request_models = dict(PROMPT_PROVIDER_MODELS)
        self.rewrite_default_model = PROMPT_REWRITE_MODEL or self.model
        self.rewrite_model_options = self._normalize_model_options(PROMPT_REWRITE_MODEL_OPTIONS)
        if self.rewrite_default_model not in self.rewrite_model_options:
            self.rewrite_model_options = [
                self.rewrite_default_model,
                *self.rewrite_model_options,
            ]
        self.provider_runtimes = self._build_provider_runtimes()
        self.provider_runtime_stages = self._build_provider_runtime_stages()
        self.provider_success_counts = {runtime.name: 0 for runtime in self.provider_runtimes}
        self.client: Any | None = self.provider_runtimes[0].client
        self.default_timeout_ms = PROMPT_TIMEOUT_MS
        self.http_timeout_ms = PROMPT_HTTP_TIMEOUT_MS
        self.initial_stage_timeout_ms = PROMPT_INITIAL_STAGE_TIMEOUT_MS
        self.temperature = PROMPT_TEMPERATURE
        self.rewrite_default_temperature = self.resolve_rewrite_temperature(PROMPT_TEMPERATURE)
        self.max_completion_tokens = PROMPT_MAX_COMPLETION_TOKENS
        self.enhance_system_prompt_path = PROMPT_ENHANCE_SYSTEM_PROMPT_PATH
        self.enhance_system_prompt_fallback_path = (PROMPT_ENHANCE_SYSTEM_PROMPT_FALLBACK_PATH)
        self.auto_system_prompt_path = PROMPT_AUTO_SYSTEM_PROMPT_PATH
        self.auto_system_prompt_fallback_path = (PROMPT_AUTO_SYSTEM_PROMPT_FALLBACK_PATH)
        self.rewrite_all_system_prompt_path = PROMPT_REWRITE_ALL_SYSTEM_PROMPT_PATH
        self.rewrite_all_system_prompt_fallback_path = (PROMPT_REWRITE_ALL_SYSTEM_PROMPT_FALLBACK_PATH)
        self.rewrite_user_system_prompt_path = (PROMPT_REWRITE_USER_SYSTEM_PROMPT_PATH)
        self.rewrite_user_system_prompt_fallback_path = (PROMPT_REWRITE_USER_SYSTEM_PROMPT_FALLBACK_PATH)
        self.reload_system_prompts()

    def _build_provider_runtimes(self) -> list[ProviderRuntime]:
        runtimes: list[ProviderRuntime] = []
        for provider_name in PROMPT_PROVIDER_PRIORITY:
            api_key = PROMPT_API_KEYS[provider_name]
            api_base_url = PROMPT_API_BASE_URLS[provider_name]
            if not isinstance(api_key, str) or not api_key.strip():
                env_names = ", ".join(PROMPT_PROVIDER_API_KEY_NAMES[provider_name])
                raise RuntimeError("Missing required environment variable: one of "
                                   f"{env_names}")
            runtimes.append(
                ProviderRuntime(
                    name=provider_name,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    request_model=self.provider_request_models.get(
                        provider_name,
                        self.model,
                    ),
                    client=self._build_client(
                        provider=provider_name,
                        api_key=api_key,
                        api_base_url=api_base_url,
                    ),
                ))
        return runtimes

    def _build_provider_runtime_stages(self) -> list[list[ProviderRuntime]]:
        runtime_by_name = {runtime.name: runtime for runtime in self.provider_runtimes}
        stages: list[list[ProviderRuntime]] = []
        for stage_names in PROMPT_PROVIDER_RUNTIME_STAGES:
            stage = [runtime_by_name[name] for name in stage_names if name in runtime_by_name]
            if stage:
                stages.append(stage)
        return stages

    def reload_system_prompts(self) -> None:
        (
            self.enhance_system_prompt,
            self.enhance_system_prompt_source_path,
        ) = _load_prompt_required_with_path(
            self.enhance_system_prompt_path,
            "next-segment",
            self.enhance_system_prompt_fallback_path,
        )
        (
            self.auto_system_prompt,
            self.auto_system_prompt_source_path,
        ) = _load_prompt_required_with_path(
            self.auto_system_prompt_path,
            "auto-extension",
            self.auto_system_prompt_fallback_path,
        )
        (
            self.rewrite_all_system_prompt,
            self.rewrite_all_system_prompt_source_path,
        ) = _load_prompt_required_with_path(
            self.rewrite_all_system_prompt_path,
            "rewrite-window",
            self.rewrite_all_system_prompt_fallback_path,
        )
        (
            self.rewrite_user_system_prompt,
            self.rewrite_user_system_prompt_source_path,
        ) = _load_prompt_with_prompt_fallback(
            self.rewrite_user_system_prompt_path,
            "rewrite-user",
            self.rewrite_all_system_prompt,
            self.rewrite_all_system_prompt_source_path,
            self.rewrite_user_system_prompt_fallback_path,
        )

    def get_prompt_config(self) -> dict[str, Any]:
        return {
            "next_segment_system_prompt_path":
            getattr(
                self,
                "enhance_system_prompt_source_path",
                self.enhance_system_prompt_path,
            ),
            "auto_extension_system_prompt_path":
            getattr(
                self,
                "auto_system_prompt_source_path",
                self.auto_system_prompt_path,
            ),
            "rewrite_window_system_prompt_path":
            getattr(
                self,
                "rewrite_all_system_prompt_source_path",
                self.rewrite_all_system_prompt_path,
            ),
            "rewrite_user_system_prompt_path":
            _resolve_prompt_save_path(
                self.rewrite_user_system_prompt_path,
                self.rewrite_user_system_prompt_fallback_path,
            ),
            "next_segment_system_prompt":
            self.enhance_system_prompt,
            "auto_extension_system_prompt":
            self.auto_system_prompt,
            "rewrite_window_system_prompt":
            self.rewrite_all_system_prompt,
            "rewrite_user_system_prompt":
            self.rewrite_user_system_prompt,
            "rewrite_model":
            self.rewrite_default_model,
            "rewrite_model_options":
            list(self.rewrite_model_options),
            "rewrite_temperature":
            getattr(
                self,
                "rewrite_default_temperature",
                self.temperature,
            ),
        }

    def save_prompt_config(
        self,
        *,
        next_segment_system_prompt: str | None = None,
        auto_extension_system_prompt: str | None = None,
        rewrite_window_system_prompt: str | None = None,
        rewrite_user_system_prompt: str | None = None,
        rewrite_model: str | None = None,
        rewrite_temperature: float | None = None,
    ) -> dict[str, Any]:
        if next_segment_system_prompt is not None:
            normalized = next_segment_system_prompt.strip()
            if not normalized:
                raise ValueError("next_segment_system_prompt cannot be empty.")
            _save_prompt(
                _resolve_prompt_save_path(
                    self.enhance_system_prompt_path,
                    self.enhance_system_prompt_fallback_path,
                ),
                normalized,
                "next-segment",
            )

        if auto_extension_system_prompt is not None:
            normalized = auto_extension_system_prompt.strip()
            if not normalized:
                raise ValueError("auto_extension_system_prompt cannot be empty.")
            _save_prompt(
                _resolve_prompt_save_path(
                    self.auto_system_prompt_path,
                    self.auto_system_prompt_fallback_path,
                ),
                normalized,
                "auto-extension",
            )

        if rewrite_window_system_prompt is not None:
            normalized = rewrite_window_system_prompt.strip()
            if not normalized:
                raise ValueError("rewrite_window_system_prompt cannot be empty.")
            _save_prompt(
                _resolve_prompt_save_path(
                    self.rewrite_all_system_prompt_path,
                    self.rewrite_all_system_prompt_fallback_path,
                ),
                normalized,
                "rewrite-window",
            )

        if rewrite_user_system_prompt is not None:
            normalized = rewrite_user_system_prompt.strip()
            if not normalized:
                raise ValueError("rewrite_user_system_prompt cannot be empty.")
            _save_prompt(
                _resolve_prompt_save_path(
                    self.rewrite_user_system_prompt_path,
                    self.rewrite_user_system_prompt_fallback_path,
                ),
                normalized,
                "rewrite-user",
            )

        if rewrite_model is not None:
            self.set_rewrite_default_model(rewrite_model)
        if rewrite_temperature is not None:
            self.set_rewrite_default_temperature(rewrite_temperature)

        self.reload_system_prompts()
        return self.get_prompt_config()

    def _resolve_timeout_ms(self, timeout_ms: int | None) -> int:
        if timeout_ms is None:
            return self.default_timeout_ms
        return timeout_ms

    def _resolve_initial_stage_timeout_seconds(self) -> float:
        timeout_ms = getattr(
            self,
            "initial_stage_timeout_ms",
            PROMPT_INITIAL_STAGE_TIMEOUT_MS,
        )
        if not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
            timeout_ms = PROMPT_INITIAL_STAGE_TIMEOUT_MS
        return float(timeout_ms) / 1000.0

    @staticmethod
    def _normalize_model_options(values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            clean_value = value.strip() if isinstance(value, str) else ""
            if not clean_value or clean_value in normalized:
                continue
            normalized.append(clean_value)
        return normalized

    def _build_client(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
    ) -> Any:
        provider_name = provider or self.provider
        provider_family = _resolve_provider_family(provider_name)
        provider_api_key = api_key or self.api_key
        resolved_api_base_url = (self.api_base_url if api_base_url is None else api_base_url)
        if provider_family == "cerebras":
            if Cerebras is None:
                raise RuntimeError("Cerebras SDK is not installed. Install "
                                   "'cerebras-cloud-sdk' to use prompt enhancement.")
            return Cerebras(api_key=provider_api_key)

        if provider_family == "groq":
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK is not installed. Install 'openai' to use "
                                   f"the {provider_name} prompt provider.")
            client_kwargs = {
                "api_key": provider_api_key,
            }
            if resolved_api_base_url:
                client_kwargs["base_url"] = resolved_api_base_url
            return OpenAI(**client_kwargs)

        raise RuntimeError(f"Unsupported prompt provider: {provider_name}")

    def _get_provider_runtimes(self) -> list[ProviderRuntime]:
        runtimes = getattr(self, "provider_runtimes", None)
        if isinstance(runtimes, list) and runtimes:
            return runtimes
        return []

    def _get_provider_runtime_stages(self) -> list[list[ProviderRuntime]]:
        stages = getattr(self, "provider_runtime_stages", None)
        if isinstance(stages, list) and stages:
            normalized_stages = [list(stage) for stage in stages if isinstance(stage, (list, tuple)) and stage]
            if normalized_stages:
                return normalized_stages
        runtimes = self._get_provider_runtimes()
        if not runtimes:
            return []
        return [runtimes]

    def _get_provider_success_counts_store(self) -> dict[str, int]:
        counts = getattr(self, "provider_success_counts", None)
        if isinstance(counts, dict):
            return counts
        counts = {runtime.name: 0 for runtime in self._get_provider_runtimes()}
        self.provider_success_counts = counts
        return counts

    def _record_provider_success(self, provider_name: str) -> None:
        counts = self._get_provider_success_counts_store()
        current_value = counts.get(provider_name, 0)
        counts[provider_name] = (current_value if isinstance(current_value, int) and current_value >= 0 else 0) + 1

    def get_provider_success_counts(self) -> dict[str, int]:
        counts = self._get_provider_success_counts_store()
        normalized: dict[str, int] = {}
        for provider_name in PROMPT_PROVIDER_PRIORITY:
            raw_value = counts.get(provider_name, 0)
            normalized[provider_name] = (raw_value if isinstance(raw_value, int) and raw_value >= 0 else 0)
        for provider_name, raw_value in counts.items():
            if provider_name in normalized:
                continue
            normalized[provider_name] = (raw_value if isinstance(raw_value, int) and raw_value >= 0 else 0)
        return normalized

    def _resolve_provider_request_model(
        self,
        provider_name: str,
        requested_model: str | None,
    ) -> str:
        provider_request_models = getattr(
            self,
            "provider_request_models",
            {},
        )
        candidate = (requested_model.strip() if isinstance(requested_model, str) else "")
        if not candidate:
            return provider_request_models.get(
                provider_name,
                self.model,
            )
        if candidate == self.rewrite_default_model:
            return provider_request_models.get(
                provider_name,
                candidate,
            )
        return candidate

    def _build_body(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "model":
            model or self.model,
            "temperature":
            self.temperature if temperature is None else temperature,
            "max_completion_tokens":
            self.max_completion_tokens,
            "response_format": {
                "type": "json_object"
            },
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

    async def _run_blocking_request(self, func, /, *args, **kwargs):
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _runner() -> None:
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - thread handoff
                result_queue.put(("error", exc))
            else:
                result_queue.put(("ok", result))

        threading.Thread(
            target=_runner,
            name="prompt-provider-call",
            daemon=True,
        ).start()
        while True:
            try:
                status, payload = result_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

            if status == "error":
                raise payload
            return payload

    async def _run_provider_race(
        self,
        operation_name: str,
        attempt_factory,
        *,
        initial_stage_timeout_seconds: float | None = None,
        fallback_stage_timeout_seconds: float | None = None,
    ):
        runtime_stages = self._get_provider_runtime_stages()
        if not runtime_stages:
            raise RuntimeError(f"{operation_name} is unavailable without provider runtimes.")
        errors: list[str] = []
        for stage_index, stage_runtimes in enumerate(runtime_stages):
            stage_timeout_seconds = fallback_stage_timeout_seconds
            if stage_index == 0 and len(runtime_stages) > 1:
                stage_timeout_seconds = initial_stage_timeout_seconds
            if stage_index > 0:
                _enhance_print(
                    "INFO",
                    f"{operation_name} entering fallback stage with providers="
                    f"{','.join(runtime.name for runtime in stage_runtimes)}",
                )

            queue: asyncio.Queue[tuple[str, str, Any]] = asyncio.Queue()

            async def _runner(runtime: ProviderRuntime) -> None:
                try:
                    if (isinstance(stage_timeout_seconds, (int, float)) and stage_timeout_seconds > 0):
                        result = await asyncio.wait_for(
                            attempt_factory(runtime),
                            timeout=stage_timeout_seconds,
                        )
                    else:
                        result = await attempt_factory(runtime)
                except asyncio.TimeoutError:
                    await queue.put((
                        "error",
                        runtime.name,
                        RuntimeError("timed out after "
                                     f"{stage_timeout_seconds:.2f}s"),
                    ))
                except Exception as exc:
                    await queue.put(("error", runtime.name, exc))
                else:
                    await queue.put(("success", runtime.name, result))

            tasks = [asyncio.create_task(_runner(runtime)) for runtime in stage_runtimes]
            stage_errors: list[str] = []
            try:
                for _ in tasks:
                    status, provider_name, payload = await queue.get()
                    if status == "success":
                        self._record_provider_success(provider_name)
                        for task in tasks:
                            task.cancel()
                        return payload

                    stage_errors.append(f"provider={provider_name} error={payload}")
                    _enhance_print(
                        "WARN",
                        f"{operation_name} failed for provider="
                        f"{provider_name}: {payload}",
                    )
            finally:
                for task in tasks:
                    task.cancel()

            if stage_errors:
                errors.extend(stage_errors)

        raise RuntimeError(f"{operation_name} failed for all providers. " + " | ".join(errors))

    async def _request_content(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        timeout_ms: int,
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], str]:
        timeout_seconds = max(timeout_ms, self.http_timeout_ms) / 1000.0
        body = self._build_body(
            system_prompt=system_prompt,
            user_payload=user_payload,
            model=model,
            temperature=temperature,
        )
        return await self._request_content_with_body(
            body=body,
            timeout_seconds=timeout_seconds,
        )

    async def _request_content_with_body(
        self,
        *,
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], str]:
        runtimes = self._get_provider_runtimes()
        if runtimes:
            return await self._request_content_with_body_for_runtime(
                runtime=runtimes[0],
                body=body,
                timeout_seconds=timeout_seconds,
            )

        provider_body = dict(body)
        provider_body["model"] = self._resolve_provider_request_model(
            self.provider,
            provider_body.get("model"),
        )
        provider_family = _resolve_provider_family(self.provider)
        if provider_family == "cerebras":
            response_json = await self._request_content_with_body_cerebras(
                client=self.client,
                body=provider_body,
                timeout_seconds=timeout_seconds,
            )
        elif provider_family == "groq":
            response_json = await self._request_content_with_body_openai_compatible(
                provider_name=self.provider,
                client=self.client,
                body=provider_body,
                timeout_seconds=timeout_seconds,
            )
        else:
            raise RuntimeError("Prompt enhancer has no provider runtimes configured.")

        content = _extract_content_or_empty(response_json)
        return response_json, content

    async def _request_content_with_body_for_runtime(
        self,
        *,
        runtime: ProviderRuntime,
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], str]:
        provider_body = dict(body)
        provider_body["model"] = self._resolve_provider_request_model(
            runtime.name,
            provider_body.get("model"),
        )
        provider_family = _resolve_provider_family(runtime.name)
        if provider_family == "cerebras":
            response_json = await self._request_content_with_body_cerebras(
                client=runtime.client,
                body=provider_body,
                timeout_seconds=timeout_seconds,
            )
        elif provider_family == "groq":
            response_json = await self._request_content_with_body_openai_compatible(
                provider_name=provider_family,
                client=runtime.client,
                body=provider_body,
                timeout_seconds=timeout_seconds,
            )
        else:  # pragma: no cover - config validation should reject this.
            raise RuntimeError(f"Unsupported prompt provider: {runtime.name}")

        content = _extract_content_or_empty(response_json)
        return response_json, content

    async def _request_content_with_body_cerebras(
        self,
        *,
        client: Any,
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del timeout_seconds
        cerebras_body = {
            "model": body.get("model"),
            "messages": body.get("messages"),
            "temperature": body.get("temperature"),
        }
        cerebras_body = {key: value for key, value in cerebras_body.items() if value is not None}
        response = await self._run_blocking_request(
            client.chat.completions.create,
            **cerebras_body,
        )
        return _dump_response_json(response)

    async def _request_content_with_body_openai_compatible(
        self,
        *,
        provider_name: str,
        client: Any,
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del provider_name, timeout_seconds
        openai_body = {
            "model": body.get("model"),
            "messages": body.get("messages"),
            "temperature": body.get("temperature"),
        }
        max_completion_tokens = body.get("max_completion_tokens")
        if max_completion_tokens is not None:
            openai_body["max_completion_tokens"] = max_completion_tokens
        openai_body = {key: value for key, value in openai_body.items() if value is not None}
        response = await self._run_blocking_request(
            client.chat.completions.create,
            **openai_body,
        )
        return _dump_response_json(response)

    async def _request_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        timeout_ms: int,
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        response_json, content = await self._request_content(
            system_prompt=system_prompt,
            user_payload=user_payload,
            timeout_ms=timeout_ms,
            model=model,
            temperature=temperature,
        )
        parsed = _parse_json_response(content)
        return parsed, response_json, content

    def _require_prompt_field(self, parsed: dict[str, Any], field_name: str) -> str:
        value = parsed.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"Missing {field_name} string.")
        prompt = value.strip()
        if not prompt:
            raise ValueError(f"{field_name} is empty.")
        return prompt

    def _optional_prompt_field(
        self,
        parsed: dict[str, Any],
        field_name: str,
    ) -> str | None:
        value = parsed.get(field_name)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _require_prompt_list_field(
        self,
        parsed: dict[str, Any],
        field_name: str,
        expected_len: int,
    ) -> list[str]:
        value = parsed.get(field_name)
        if not isinstance(value, list):
            raise ValueError(f"Missing {field_name} list.")

        prompts: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} contains non-string item.")
            normalized = item.strip()
            if not normalized:
                raise ValueError(f"{field_name} contains empty prompt.")
            prompts.append(normalized)

        if len(prompts) != expected_len:
            raise ValueError(f"{field_name} length mismatch: expected {expected_len}, got {len(prompts)}.")
        return prompts

    def _extract_rewrite_segment_prompts(
        self,
        parsed: dict[str, Any],
        expected_len: int,
    ) -> list[str]:
        return self._require_prompt_list_field(
            parsed,
            "segment_prompts",
            expected_len,
        )

    def _normalize_rewrite_prompt_item(self, item: Any) -> str | None:
        if isinstance(item, str):
            normalized = item.strip()
            return normalized or None
        if not isinstance(item, dict):
            return None

        for key in (
                "prompt",
                "text",
                "segment_prompt",
                "content",
                "description",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _maybe_extract_rewrite_prompt_list(
        self,
        value: Any,
        expected_len: int,
    ) -> list[str] | None:
        if not isinstance(value, list):
            return None

        prompts: list[str] = []
        for item in value:
            normalized = self._normalize_rewrite_prompt_item(item)
            if normalized is None:
                return None
            prompts.append(normalized)

        if len(prompts) != expected_len:
            return None
        return prompts

    def _extract_indexed_rewrite_prompts(
        self,
        parsed: dict[str, Any],
        expected_len: int,
    ) -> list[str] | None:
        indexed_prompts: dict[int, str] = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                continue
            match = re.fullmatch(
                r"(?:segment|prompt|scene|shot)[ _-]?(\d+)",
                key.strip(),
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            segment_idx = int(match.group(1))
            if segment_idx < 1 or segment_idx > expected_len:
                continue
            normalized = self._normalize_rewrite_prompt_item(value)
            if normalized is None:
                return None
            indexed_prompts[segment_idx] = normalized

        if any(idx not in indexed_prompts for idx in range(1, expected_len + 1)):
            return None
        return [indexed_prompts[idx] for idx in range(1, expected_len + 1)]

    def _extract_numbered_rewrite_prompts_from_text(
        self,
        content: str,
        expected_len: int,
    ) -> list[str] | None:
        lines = content.splitlines()
        if not lines:
            return None

        numbered_segments: dict[int, list[str]] = {}
        current_idx: int | None = None
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            normalized_line = re.sub(
                r"^\s*[-*]\s*",
                "",
                line,
            )
            match = re.match(
                (r'^(?:\*\*)?(?:segment|scene|shot|prompt)?\s*[_ -]?(\d+)'
                 r'(?:\*\*)?\s*[:.)-]\s*(.+)$'),
                normalized_line,
                flags=re.IGNORECASE,
            )
            if match:
                segment_idx = int(match.group(1))
                if 1 <= segment_idx <= expected_len:
                    numbered_segments[segment_idx] = [match.group(2).strip()]
                    current_idx = segment_idx
                    continue

            if current_idx is not None:
                numbered_segments[current_idx].append(normalized_line)

        if any(idx not in numbered_segments for idx in range(1, expected_len + 1)):
            return None

        prompts: list[str] = []
        for idx in range(1, expected_len + 1):
            prompt = " ".join(part for part in numbered_segments[idx] if part).strip()
            prompt = prompt.strip().strip('"').strip("'").strip()
            if not prompt:
                return None
            prompts.append(prompt)
        return prompts

    def _extract_rewrite_segment_prompts_lenient(
        self,
        parsed: dict[str, Any],
        expected_len: int,
        *,
        raw_content: str | None = None,
    ) -> list[str]:
        candidate_dicts: list[dict[str, Any]] = [parsed]
        for key in ("rollout", "current_rollout", "rewritten_rollout"):
            value = parsed.get(key)
            if isinstance(value, dict):
                candidate_dicts.append(value)

        for candidate in candidate_dicts:
            for key in (
                    "segment_prompts",
                    "rewritten_prompts",
                    "prompts",
                    "segments",
            ):
                prompts = self._maybe_extract_rewrite_prompt_list(
                    candidate.get(key),
                    expected_len,
                )
                if prompts is not None:
                    return prompts

            indexed_prompts = self._extract_indexed_rewrite_prompts(
                candidate,
                expected_len,
            )
            if indexed_prompts is not None:
                return indexed_prompts

        if isinstance(raw_content, str) and raw_content.strip():
            numbered_prompts = self._extract_numbered_rewrite_prompts_from_text(
                raw_content,
                expected_len,
            )
            if numbered_prompts is not None:
                return numbered_prompts

        raise ValueError("No rewrite segment prompts found in assistant response.")

    def _extract_rewrite_rollout(
        self,
        parsed: dict[str, Any],
        expected_len: int,
        *,
        raw_content: str | None = None,
        preset_id: str | None = None,
        preset_label: str | None = None,
    ) -> tuple[str, str, list[str]]:
        candidate_dicts: list[dict[str, Any]] = [parsed]
        for key in ("rollout", "current_rollout", "rewritten_rollout"):
            value = parsed.get(key)
            if isinstance(value, dict):
                candidate_dicts.append(value)

        for candidate in candidate_dicts:
            try:
                rollout_id = self._require_prompt_field(candidate, "id")
                rollout_label = self._require_prompt_field(candidate, "label")
                prompts = self._extract_rewrite_segment_prompts(
                    candidate,
                    expected_len,
                )
                return rollout_id, rollout_label, prompts
            except ValueError:
                continue

        rollout_id = next(
            (value for value in (self._optional_prompt_field(candidate, "id")
                                 for candidate in candidate_dicts) if value is not None),
            _resolve_rollout_id(preset_id),
        )
        rollout_label = next(
            (value for value in (self._optional_prompt_field(candidate, "label")
                                 for candidate in candidate_dicts) if value is not None),
            _resolve_rollout_label(preset_label),
        )
        prompts = self._extract_rewrite_segment_prompts_lenient(
            parsed,
            expected_len,
            raw_content=raw_content,
        )
        return rollout_id, rollout_label, prompts

    def _extract_rewrite_rollout_from_content(
        self,
        response_content: str,
        expected_len: int,
        *,
        preset_id: str | None = None,
        preset_label: str | None = None,
    ) -> tuple[str, str, list[str]]:
        parse_error: ValueError | None = None
        try:
            parsed = _parse_json_response(response_content)
        except ValueError as exc:
            parsed = {}
            parse_error = exc

        try:
            return self._extract_rewrite_rollout(
                parsed,
                expected_len,
                raw_content=response_content,
                preset_id=preset_id,
                preset_label=preset_label,
            )
        except ValueError as exc:
            if (parse_error is not None and str(exc) == "No rewrite segment prompts found in assistant response."):
                raise parse_error from exc
            raise

    def _extract_single_clip_prompt(self, content: str) -> str:
        text = content.strip()
        if not text:
            raise ValueError("Assistant response is empty.")

        parsed = _parse_json_response(text)
        return self._require_prompt_field(parsed, "prompt")

    def set_rewrite_default_model(self, rewrite_model: str) -> str:
        normalized = (rewrite_model.strip() if isinstance(rewrite_model, str) else "")
        if not normalized:
            raise ValueError("rewrite_model cannot be empty.")
        if normalized not in self.rewrite_model_options:
            allowed = ", ".join(self.rewrite_model_options)
            raise ValueError(f"Unsupported rewrite_model {normalized!r}. "
                             f"Expected one of: {allowed}.")
        self.rewrite_default_model = normalized
        return self.rewrite_default_model

    def set_rewrite_default_temperature(
        self,
        rewrite_temperature: float,
    ) -> float:
        if not isinstance(rewrite_temperature, (int, float)):
            raise ValueError("rewrite_temperature must be numeric.")
        self.rewrite_default_temperature = self.resolve_rewrite_temperature(float(rewrite_temperature))
        return self.rewrite_default_temperature

    def resolve_rewrite_system_prompt(
        self,
        system_prompt_override: str | None,
    ) -> str:
        normalized = _normalize_prompt(system_prompt_override)
        if normalized:
            return normalized
        return self.rewrite_all_system_prompt

    def resolve_rewrite_new_rollout_system_prompt(
        self,
        system_prompt_override: str | None,
    ) -> str:
        normalized = _normalize_prompt(system_prompt_override)
        if normalized:
            return normalized
        return getattr(
            self,
            "rewrite_user_system_prompt",
            self.rewrite_all_system_prompt,
        )

    def resolve_rewrite_model(self, requested_model: str | None) -> str:
        candidate = requested_model.strip() if isinstance(requested_model, str) else ""
        if candidate and candidate in self.rewrite_model_options:
            return candidate
        return self.rewrite_default_model

    def get_rewrite_model_config(self) -> dict[str, Any]:
        return {
            "default_model": self.rewrite_default_model,
            "options": list(self.rewrite_model_options),
        }

    def resolve_rewrite_temperature(self, requested_temperature: float | None) -> float:
        if requested_temperature is None:
            return getattr(
                self,
                "rewrite_default_temperature",
                self.temperature,
            )
        try:
            numeric_value = float(requested_temperature)
        except (TypeError, ValueError):
            return getattr(
                self,
                "rewrite_default_temperature",
                self.temperature,
            )
        return min(2.0, max(0.0, numeric_value))

    async def enhance_prompt(
        self,
        conditioning_prompt: str,
        *,
        locked_segments: list[str] | None = None,
        next_segment_idx: int | None = None,
        preset_id: str | None = None,
        mode: str = "user",
        model: str | None = None,
        timeout_ms: int | None = None,
    ) -> EnhanceResult:
        cleaned = _normalize_prompt(conditioning_prompt)
        resolved_model = (self.resolve_rewrite_model(model)
                          if isinstance(model, str) and model.strip() else self.resolve_rewrite_model(None))
        is_single_clip_mode = mode in {"single_clip", "simple5s", "single5s"}
        if not cleaned:
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error="No valid prompt provided.",
                provider=self.provider_label,
                model=resolved_model,
                latency_ms=0.0,
            )

        effective_timeout_ms = self._resolve_timeout_ms(timeout_ms)
        if is_single_clip_mode:
            request_system_prompt = self.auto_system_prompt
            user_payload = {
                "request": (
                    "Expand the user prompt into one detailed prompt for a "
                    "single 5-second LTX-2.3 video clip. Respond with "
                    'valid JSON only as {"prompt": "..."}.'  # noqa: E501
                ),
                "user_prompt":
                cleaned,
            }
        else:
            locked_segments_clean = [
                segment for segment in (_normalize_prompt(item) for item in (locked_segments or [])) if segment
            ]
            resolved_next_segment_idx = (next_segment_idx if isinstance(next_segment_idx, int) else None)
            if resolved_next_segment_idx is None or resolved_next_segment_idx < 1:
                resolved_next_segment_idx = len(locked_segments_clean) + 1
            next_segment_key = f"segment_{resolved_next_segment_idx}"
            locked_text = _format_locked_segments(locked_segments_clean)
            request_system_prompt = self.enhance_system_prompt
            user_payload = {
                "request": (
                    "<locked_segments>\n"
                    f"{locked_text}\n"
                    "</locked_segments>\n\n"
                    f"<conditioning_prompt>{cleaned}</conditioning_prompt>\n\n"
                    f"Write exactly one new segment ({next_segment_key}) "
                    "continuing from the locked segments. "
                    'Respond with valid JSON only as {"next_prompt": "..."}.'  # noqa: E501
                ),
            }

        t0 = time.perf_counter()
        try:
            _enhance_print(
                "INFO",
                "Enhancing prompt: "
                f"{cleaned}, system_prompt: {request_system_prompt}, "
                f"mode={mode}",
            )
            _enhance_print("INFO", f"user_payload: {user_payload}")
            runtimes = self._get_provider_runtimes()
            if runtimes:
                timeout_seconds = max(
                    effective_timeout_ms,
                    self.http_timeout_ms,
                ) / 1000.0
                request_body = self._build_body(
                    system_prompt=request_system_prompt,
                    user_payload=user_payload,
                    model=resolved_model,
                )

                async def _attempt(runtime: ProviderRuntime):
                    response_content: str | None = None
                    try:
                        _, response_content = await self._request_content_with_body_for_runtime(
                            runtime=runtime,
                            body=request_body,
                            timeout_seconds=timeout_seconds,
                        )
                        if is_single_clip_mode:
                            prompt = self._extract_single_clip_prompt(response_content)
                        else:
                            parsed = _parse_json_response(response_content)
                            prompt = self._require_prompt_field(
                                parsed,
                                "next_prompt",
                            )
                        return _resolve_provider_label(runtime.name), prompt
                    except Exception as exc:
                        error_detail = str(exc)
                        if (isinstance(response_content, str) and response_content.strip()):
                            error_detail = (f"{error_detail} | assistant_response="
                                            f"{_preview_text(response_content, limit=240)}")
                        raise RuntimeError(error_detail) from exc

                provider_name, enhanced_prompt = await self._run_provider_race(
                    "enhance_prompt",
                    _attempt,
                    initial_stage_timeout_seconds=(self._resolve_initial_stage_timeout_seconds()),
                    fallback_stage_timeout_seconds=timeout_seconds,
                )
            else:
                response_json, response_content = await self._request_content(
                    system_prompt=request_system_prompt,
                    user_payload=user_payload,
                    timeout_ms=effective_timeout_ms,
                    model=resolved_model,
                )
                del response_json
                if is_single_clip_mode:
                    enhanced_prompt = self._extract_single_clip_prompt(response_content)
                else:
                    parsed = _parse_json_response(response_content)
                    enhanced_prompt = self._require_prompt_field(parsed, "next_prompt")
                provider_name = self.provider_label
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt=enhanced_prompt,
                fallback_used=False,
                error=None,
                provider=provider_name,
                model=resolved_model,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error=str(exc),
                provider=self.provider_label,
                model=resolved_model,
                latency_ms=latency_ms,
            )

    async def generate_auto_prompt(
        self,
        *,
        locked_segments: list[str] | None = None,
        next_segment_idx: int | None = None,
        model: str | None = None,
        timeout_ms: int | None = None,
    ) -> EnhanceResult:
        locked_segments_clean = [
            segment for segment in (_normalize_prompt(item) for item in (locked_segments or [])) if segment
        ]
        resolved_next_segment_idx = (next_segment_idx if isinstance(next_segment_idx, int) else None)
        if resolved_next_segment_idx is None or resolved_next_segment_idx < 1:
            resolved_next_segment_idx = len(locked_segments_clean) + 1
        next_segment_key = f"segment_{resolved_next_segment_idx}"
        locked_text = _format_locked_segments(locked_segments_clean)

        effective_timeout_ms = self._resolve_timeout_ms(timeout_ms)
        user_payload = {
            "request": (
                "<locked_segments>\n"
                f"{locked_text}\n"
                "</locked_segments>\n\n"
                f"Write exactly one new segment ({next_segment_key}) "
                "that continues linearly from the locked segments. "
                "Infer the next narrative beat from this history. "
                'Respond with valid JSON only as {"next_prompt": "..."}.'  # noqa: E501
            ),
        }

        t0 = time.perf_counter()
        resolved_model = (self.resolve_rewrite_model(model)
                          if isinstance(model, str) and model.strip() else self.resolve_rewrite_model(None))
        try:
            _enhance_print(
                "INFO",
                "Auto extension request: "
                f"next_segment={resolved_next_segment_idx} "
                f"locked_count={len(locked_segments_clean)}",
            )
            _enhance_print("INFO", f"auto_user_payload: {user_payload}")
            runtimes = self._get_provider_runtimes()
            if runtimes:
                timeout_seconds = max(
                    effective_timeout_ms,
                    self.http_timeout_ms,
                ) / 1000.0
                request_body = self._build_body(
                    system_prompt=self.auto_system_prompt,
                    user_payload=user_payload,
                    model=resolved_model,
                )

                async def _attempt(runtime: ProviderRuntime):
                    response_content: str | None = None
                    try:
                        _, response_content = await self._request_content_with_body_for_runtime(
                            runtime=runtime,
                            body=request_body,
                            timeout_seconds=timeout_seconds,
                        )
                        parsed = _parse_json_response(response_content)
                        prompt = self._require_prompt_field(
                            parsed,
                            "next_prompt",
                        )
                        return _resolve_provider_label(runtime.name), prompt
                    except Exception as exc:
                        error_detail = str(exc)
                        if (isinstance(response_content, str) and response_content.strip()):
                            error_detail = (f"{error_detail} | assistant_response="
                                            f"{_preview_text(response_content, limit=240)}")
                        raise RuntimeError(error_detail) from exc

                provider_name, next_prompt = await self._run_provider_race(
                    "generate_auto_prompt",
                    _attempt,
                    initial_stage_timeout_seconds=(self._resolve_initial_stage_timeout_seconds()),
                    fallback_stage_timeout_seconds=timeout_seconds,
                )
            else:
                response_json, response_content = await self._request_content(
                    system_prompt=self.auto_system_prompt,
                    user_payload=user_payload,
                    timeout_ms=effective_timeout_ms,
                    model=resolved_model,
                )
                del response_json
                parsed = _parse_json_response(response_content)
                next_prompt = self._require_prompt_field(parsed, "next_prompt")
                provider_name = self.provider_label
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt=next_prompt,
                fallback_used=False,
                error=None,
                provider=provider_name,
                model=resolved_model,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EnhanceResult(
                prompt="",
                fallback_used=True,
                error=str(exc),
                provider=self.provider_label,
                model=resolved_model,
                latency_ms=latency_ms,
            )

    async def rewrite_prompt_sequence(
        self,
        prompts: list[str],
        *,
        preset_id: str | None = None,
        preset_label: str | None = None,
        rewrite_instruction: str | None = None,
        rewrite_model: str | None = None,
        rewrite_temperature: float | None = None,
        timeout_ms: int | None = None,
        system_prompt_override: str | None = None,
    ) -> RewriteResult:
        resolved_rewrite_model = self.resolve_rewrite_model(rewrite_model)
        resolved_rewrite_temperature = self.resolve_rewrite_temperature(rewrite_temperature)
        resolved_system_prompt = self.resolve_rewrite_system_prompt(system_prompt_override)
        cleaned_prompts = normalize_prompt_window_prompts(prompts)
        rewrite_instruction_text = _normalize_prompt(rewrite_instruction)
        if not cleaned_prompts and not rewrite_instruction_text:
            return RewriteResult(
                prompts=[],
                fallback_used=True,
                error="No valid prompts to rewrite or generate.",
                provider=self.provider_label,
                model=resolved_rewrite_model,
                latency_ms=0.0,
                rollout_id=_resolve_rollout_id(preset_id),
                rollout_label=_resolve_rollout_label(preset_label),
                raw_response_text=None,
            )
        expected_len = (len(cleaned_prompts) if cleaned_prompts else DEFAULT_REWRITE_SEGMENT_COUNT)

        effective_timeout_ms = self._resolve_timeout_ms(timeout_ms)
        timeout_seconds = max(effective_timeout_ms, self.http_timeout_ms) / 1000.0
        rewrite_body = build_rewrite_request_body(
            system_prompt=resolved_system_prompt,
            prompt_window_prompts=cleaned_prompts,
            preset_id=preset_id,
            preset_label=preset_label,
            rewrite_instruction=rewrite_instruction_text,
            model=resolved_rewrite_model,
            temperature=resolved_rewrite_temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

        t0 = time.perf_counter()
        raw_response_text: str | None = None
        try:
            runtimes = self._get_provider_runtimes()
            if runtimes:

                async def _attempt(runtime: ProviderRuntime):
                    response_content: str | None = None
                    try:
                        response_json, response_content = await self._request_content_with_body_for_runtime(
                            runtime=runtime,
                            body=rewrite_body,
                            timeout_seconds=timeout_seconds,
                        )
                        if not response_content:
                            response_content = json.dumps(
                                response_json,
                                ensure_ascii=False,
                            )
                        (
                            rollout_id,
                            rollout_label,
                            rewritten_prompts,
                        ) = self._extract_rewrite_rollout_from_content(
                            response_content,
                            expected_len,
                            preset_id=preset_id,
                            preset_label=preset_label,
                        )
                        return (
                            _resolve_provider_label(runtime.name),
                            response_content,
                            rollout_id,
                            rollout_label,
                            rewritten_prompts,
                        )
                    except Exception as exc:
                        error_detail = str(exc)
                        if (isinstance(response_content, str) and response_content.strip()):
                            error_detail = (f"{error_detail} | assistant_response="
                                            f"{_preview_text(response_content, limit=240)}")
                        raise RuntimeError(error_detail) from exc

                (
                    provider_name,
                    raw_response_text,
                    rollout_id,
                    rollout_label,
                    rewritten_prompts,
                ) = await self._run_provider_race(
                    "rewrite_prompt_sequence",
                    _attempt,
                    initial_stage_timeout_seconds=(self._resolve_initial_stage_timeout_seconds()),
                    fallback_stage_timeout_seconds=timeout_seconds,
                )
            else:
                response_json, raw_response_text = await self._request_content_with_body(
                    body=rewrite_body,
                    timeout_seconds=timeout_seconds,
                )
                if not raw_response_text:
                    raw_response_text = json.dumps(
                        response_json,
                        ensure_ascii=False,
                    )
                (
                    rollout_id,
                    rollout_label,
                    rewritten_prompts,
                ) = self._extract_rewrite_rollout_from_content(
                    raw_response_text,
                    expected_len,
                    preset_id=preset_id,
                    preset_label=preset_label,
                )
                provider_name = self.provider_label
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RewriteResult(
                prompts=rewritten_prompts,
                fallback_used=False,
                error=None,
                provider=provider_name,
                model=resolved_rewrite_model,
                latency_ms=latency_ms,
                rollout_id=rollout_id,
                rollout_label=rollout_label,
                raw_response_text=raw_response_text,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RewriteResult(
                prompts=cleaned_prompts,
                fallback_used=True,
                error=str(exc),
                provider=self.provider_label,
                model=resolved_rewrite_model,
                latency_ms=latency_ms,
                rollout_id=_resolve_rollout_id(preset_id),
                rollout_label=_resolve_rollout_label(preset_label),
                raw_response_text=raw_response_text,
            )
