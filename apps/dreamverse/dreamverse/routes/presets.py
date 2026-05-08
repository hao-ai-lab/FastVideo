"""Prompt-system-config and curated-presets HTTP routes.

Exports two routers:
- ``prompt_config_router``: always registered.
- ``curated_presets_router``: registered only when ``DEVTOOLS_ENABLED``.
"""
from __future__ import annotations
# pyright: reportMissingTypeArgument=false

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dreamverse.config import (
    CURATED_PRESETS_FILE_PATH,
    CURATED_PRESETS_FALLBACK_FILE_PATH,
)

import dreamverse.runtime as runtime

prompt_config_router = APIRouter()
curated_presets_router = APIRouter()


class PromptConfigUpdateRequest(BaseModel):
    next_segment_system_prompt: str | None = None
    auto_extension_system_prompt: str | None = None
    rewrite_window_system_prompt: str | None = None
    rewrite_user_system_prompt: str | None = None
    rewrite_model: str | None = None
    rewrite_temperature: float | None = None


class AppendCuratedPresetRequest(BaseModel):
    id: str
    label: str
    segment_prompts: list[str]


def _sanitize_preset_id(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower())
    normalized = normalized.strip("_")
    return normalized or "custom_editable"


def _load_curated_presets_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in curated presets file: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read curated presets file: {path}") from exc

    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Curated presets file must contain a JSON array: {path}")


def _write_curated_presets_file(path: Path, presets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to write curated presets file: {path}") from exc


def _merge_curated_presets(*preset_groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    index_by_id: dict[str, int] = {}
    for presets in preset_groups:
        for item in presets:
            if not isinstance(item, dict):
                continue
            preset_id = str(item.get("id", "")).strip()
            if not preset_id:
                continue
            normalized_id = preset_id.lower()
            normalized_item = dict(item)
            if normalized_id in index_by_id:
                merged[index_by_id[normalized_id]] = normalized_item
            else:
                index_by_id[normalized_id] = len(merged)
                merged.append(normalized_item)
    return merged


@prompt_config_router.get("/prompt-system-config")
async def get_prompt_system_config():
    """Get editable prompt-system configuration."""
    if runtime.prompt_enhancer is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancer not initialized",
        )
    return runtime.prompt_enhancer.get_prompt_config()


@prompt_config_router.post("/prompt-system-config")
async def save_prompt_system_config(payload: PromptConfigUpdateRequest):
    """Save prompt-system configuration to disk and reload runtime prompts."""
    if runtime.prompt_enhancer is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancer not initialized",
        )
    try:
        return runtime.prompt_enhancer.save_prompt_config(
            next_segment_system_prompt=payload.next_segment_system_prompt,
            auto_extension_system_prompt=payload.auto_extension_system_prompt,
            rewrite_window_system_prompt=payload.rewrite_window_system_prompt,
            rewrite_user_system_prompt=payload.rewrite_user_system_prompt,
            rewrite_model=payload.rewrite_model,
            rewrite_temperature=payload.rewrite_temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@curated_presets_router.get("/curated-presets")
async def get_curated_presets():
    """Get curated presets with devtools overlays applied."""
    file_path = Path(CURATED_PRESETS_FILE_PATH)
    fallback_path = (Path(CURATED_PRESETS_FALLBACK_FILE_PATH) if CURATED_PRESETS_FALLBACK_FILE_PATH else None)
    try:
        overlay_presets = _load_curated_presets_file(file_path)
        fallback_presets = (_load_curated_presets_file(fallback_path) if fallback_path is not None else [])
        presets = _merge_curated_presets(
            fallback_presets,
            overlay_presets,
        )
        return {
            "presets": presets,
            "count": len(presets),
            "file_path": str(file_path),
            "fallback_file_path": (str(fallback_path) if fallback_path is not None else None),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@curated_presets_router.post("/curated-presets/append")
async def append_curated_preset(payload: AppendCuratedPresetRequest):
    """Append a curated preset to the configured presets JSON file."""
    preset_id = _sanitize_preset_id(payload.id)
    label = (payload.label or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="label must be non-empty.")

    prompts = [prompt.strip() for prompt in payload.segment_prompts if isinstance(prompt, str) and prompt.strip()]
    if len(prompts) < 2:
        raise HTTPException(
            status_code=400,
            detail="segment_prompts must contain at least 2 non-empty prompts.",
        )

    file_path = Path(CURATED_PRESETS_FILE_PATH)
    fallback_path = (Path(CURATED_PRESETS_FALLBACK_FILE_PATH) if CURATED_PRESETS_FALLBACK_FILE_PATH else None)
    try:
        overlay_presets = _load_curated_presets_file(file_path)
        fallback_presets = (_load_curated_presets_file(fallback_path) if fallback_path is not None else [])
        existing_ids = {
            str(item.get("id", "")).strip().lower()
            for item in _merge_curated_presets(
                fallback_presets,
                overlay_presets,
            ) if isinstance(item, dict)
        }
        if preset_id.lower() in existing_ids:
            raise HTTPException(
                status_code=409,
                detail=("Preset id already exists in curated presets file: "
                        f"{preset_id}"),
            )

        next_entry = {
            "id": preset_id,
            "label": label,
            "segment_prompts": prompts,
        }
        overlay_presets.append(next_entry)
        _write_curated_presets_file(file_path, overlay_presets)
        return {
            "type": "curated_preset_appended",
            "preset": next_entry,
            "count": len(_merge_curated_presets(
                fallback_presets,
                overlay_presets,
            )),
            "file_path": str(file_path),
            "fallback_file_path": (str(fallback_path) if fallback_path is not None else None),
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
