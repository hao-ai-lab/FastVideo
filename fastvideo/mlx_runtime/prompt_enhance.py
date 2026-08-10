# SPDX-License-Identifier: Apache-2.0
"""Local prompt enrichment for the MLX Wan runtime (H3 Context-IR-style).

Wan's training captions are long and cinematic; short user prompts leave
quality on the table. This module expands a raw prompt into Wan-style
shot language **on device** — no remote API, no training.

Backends (first match wins):

1. **mlx-lm** — optional local LLM (``--enhance-prompt-model``).
2. **template** — deterministic cinematic expansion (always available).

System-prompt contract matches the streaming server's enhancer defaults
in ``fastvideo/entrypoints/streaming/prompt/enhancer.py`` so remote and
local paths stay interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Keep in lockstep with streaming PromptEnhancer defaults (enhance op).
DEFAULT_ENHANCE_SYSTEM_PROMPT = (
    "You are a prompt enhancer for cinematic video generation. Given "
    "a user prompt, produce an enhanced prompt that is more vivid, "
    "specific, and concrete. Keep the subject intact; add lighting, "
    "camera, and motion detail. Reply with just the enhanced prompt."
)

# Small default that fits 16 GB Macs alongside the 1.3B DiT when the user
# opts into mlx-lm. Override with --enhance-prompt-model.
DEFAULT_MLX_LM_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

_CAMERA_CUES = (
    "cinematic",
    "camera",
    "lens",
    "shot",
    "bokeh",
    "dolly",
    "tracking",
    "close-up",
    "wide shot",
    "handheld",
    "steadicam",
)
_LIGHT_CUES = (
    "light",
    "lighting",
    "sun",
    "golden hour",
    "neon",
    "rim light",
    "softbox",
    "overcast",
    "moonlight",
    "volumetric",
)
_MOTION_CUES = (
    "moving",
    "motion",
    "walk",
    "run",
    "flies",
    "flying",
    "drifts",
    "sails",
    "flows",
    "pan",
    "tilt",
    "zoom",
)


@dataclass(frozen=True)
class EnhanceResult:
    """Outcome of a prompt enrichment call."""

    original: str
    enhanced: str
    backend: str
    elapsed_s: float
    model: str | None = None

    @property
    def changed(self) -> bool:
        return self.enhanced.strip() != self.original.strip()


def _normalize_user_prompt(prompt: str) -> str:
    text = " ".join(prompt.strip().split())
    if not text:
        raise ValueError("prompt must be non-empty")
    return text


def _already_rich(prompt: str) -> bool:
    """Heuristic: long prompts that already name camera/light/motion pass through."""
    lower = prompt.lower()
    has_camera = any(c in lower for c in _CAMERA_CUES)
    has_light = any(c in lower for c in _LIGHT_CUES)
    has_motion = any(c in lower for c in _MOTION_CUES)
    return len(prompt) >= 160 and has_camera and has_light


def enhance_prompt_template(prompt: str) -> str:
    """Deterministic cinematic expansion (no model weights).

    Keeps the subject verbatim and appends Wan-style shot grammar only when
    the prompt looks thin. Idempotent on already-rich prompts.
    """
    text = _normalize_user_prompt(prompt)
    if _already_rich(text):
        return text

    lower = text.lower()
    parts = [text.rstrip(".")]

    if not any(c in lower for c in _CAMERA_CUES):
        parts.append(
            "shot on a 35mm anamorphic lens, gentle handheld micro-movement, "
            "shallow depth of field"
        )
    if not any(c in lower for c in _LIGHT_CUES):
        parts.append(
            "natural cinematic lighting with soft volumetric haze and subtle "
            "rim light separating subject from background"
        )
    if not any(c in lower for c in _MOTION_CUES):
        parts.append("smooth continuous motion with grounded physics")

    parts.append(
        "highly detailed, coherent temporal continuity, film grain, "
        "color graded like a contemporary drama"
    )
    enhanced = ", ".join(parts)
    # Single trailing period; collapse duplicate whitespace.
    enhanced = re.sub(r"\s+", " ", enhanced).strip()
    if not enhanced.endswith("."):
        enhanced += "."
    return enhanced


def enhance_prompt_mlx_lm(
    prompt: str,
    *,
    model: str = DEFAULT_MLX_LM_MODEL,
    system_prompt: str = DEFAULT_ENHANCE_SYSTEM_PROMPT,
    max_tokens: int = 128,
    temp: float = 0.6,
) -> str:
    """Run a local mlx-lm instruct model. Raises if mlx-lm is unavailable."""
    try:
        from mlx_lm import generate, load
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "mlx-lm is not installed. `uv pip install mlx-lm` or use "
            "--enhance-prompt-backend template."
        ) from exc

    text = _normalize_user_prompt(prompt)
    logger.info("[MLX enhance] loading %s", model)
    mlx_model, tokenizer = load(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:  # pragma: no cover - ancient tokenizers
        chat = f"{system_prompt}\n\nUser: {text}\nAssistant:"

    raw = generate(
        mlx_model,
        tokenizer,
        prompt=chat,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    enhanced = _clean_llm_output(raw, original=text)
    if not enhanced:
        raise RuntimeError("mlx-lm returned an empty enhance result")
    return enhanced


def _clean_llm_output(raw: str, *, original: str) -> str:
    """Strip chat debris; fall back to original if the model refused."""
    text = raw.strip()
    # Drop common prefatory phrases.
    for prefix in (
        "enhanced prompt:",
        "here's the enhanced prompt:",
        "here is the enhanced prompt:",
        "sure:",
        "sure,",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    # Keep first non-empty paragraph only.
    para = text.split("\n\n")[0].strip()
    para = " ".join(para.split())
    if len(para) < max(12, len(original) // 4):
        return original
    return para


def enhance_prompt(
    prompt: str,
    *,
    backend: str = "auto",
    model: str | None = None,
    system_prompt: str = DEFAULT_ENHANCE_SYSTEM_PROMPT,
    max_tokens: int = 128,
) -> EnhanceResult:
    """Enhance ``prompt`` with the requested backend.

    ``backend``:
      * ``"auto"`` — try mlx-lm, fall back to template
      * ``"mlx-lm"`` — require mlx-lm (error if missing)
      * ``"template"`` — deterministic expansion only
    """
    text = _normalize_user_prompt(prompt)
    backend_norm = (backend or "auto").lower()
    if backend_norm not in {"auto", "mlx-lm", "template"}:
        raise ValueError(f"Unknown enhance backend: {backend}")

    start = time.perf_counter()
    used_model: str | None = None

    if backend_norm in {"auto", "mlx-lm"}:
        try:
            used_model = model or DEFAULT_MLX_LM_MODEL
            enhanced = enhance_prompt_mlx_lm(
                text,
                model=used_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return EnhanceResult(
                original=text,
                enhanced=enhanced,
                backend="mlx-lm",
                elapsed_s=time.perf_counter() - start,
                model=used_model,
            )
        except Exception as exc:
            if backend_norm == "mlx-lm":
                raise
            logger.info(
                "[MLX enhance] mlx-lm unavailable (%s); using template backend",
                exc,
            )

    enhanced = enhance_prompt_template(text)
    return EnhanceResult(
        original=text,
        enhanced=enhanced,
        backend="template",
        elapsed_s=time.perf_counter() - start,
        model=None,
    )


def enhance_cache_path(
    prompt: str,
    *,
    backend: str,
    model: str | None,
    cache_dir: Path | None = None,
) -> Path:
    """Content-addressed cache file for an enhanced prompt string."""
    root = cache_dir or (Path.home() / ".cache" / "fastvideo" / "enhanced_prompts")
    key = "\0".join([prompt, backend, model or ""])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return root / f"{digest}.json"


def load_or_enhance_prompt(
    prompt: str,
    *,
    backend: str = "auto",
    model: str | None = None,
    system_prompt: str = DEFAULT_ENHANCE_SYSTEM_PROMPT,
    max_tokens: int = 128,
    cache: bool = True,
    cache_dir: Path | None = None,
) -> EnhanceResult:
    """Enhance with an optional on-disk cache (skip model on repeat prompts)."""
    text = _normalize_user_prompt(prompt)
    path = enhance_cache_path(text, backend=backend, model=model, cache_dir=cache_dir)
    if cache and path.is_file():
        try:
            payload = json.loads(path.read_text())
            return EnhanceResult(
                original=str(payload.get("original", text)),
                enhanced=str(payload["enhanced"]),
                # Mark cache hits explicitly so metrics/logs can distinguish
                # a free replay from a fresh template/mlx-lm call.
                backend="cache",
                elapsed_s=0.0,
                model=payload.get("model"),
            )
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    result = enhance_prompt(
        text,
        backend=backend,
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "original": result.original,
                        "enhanced": result.enhanced,
                        "backend": result.backend,
                        "model": result.model,
                    },
                    indent=2,
                )
            )
        except OSError as exc:  # pragma: no cover - cache is best-effort
            logger.info("[MLX enhance] cache write skipped: %s", exc)
    return result


def enhance_result_as_metrics(result: EnhanceResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "enhance_prompt": False,
            "enhance_backend": None,
            "enhance_model": None,
            "enhance_elapsed_s": None,
            "prompt_original": None,
            "prompt_enhanced": None,
        }
    return {
        "enhance_prompt": True,
        "enhance_backend": result.backend,
        "enhance_model": result.model,
        "enhance_elapsed_s": result.elapsed_s,
        "prompt_original": result.original,
        "prompt_enhanced": result.enhanced,
    }


__all__ = [
    "DEFAULT_ENHANCE_SYSTEM_PROMPT",
    "DEFAULT_MLX_LM_MODEL",
    "EnhanceResult",
    "enhance_cache_path",
    "enhance_prompt",
    "enhance_prompt_mlx_lm",
    "enhance_prompt_template",
    "enhance_result_as_metrics",
    "load_or_enhance_prompt",
]
