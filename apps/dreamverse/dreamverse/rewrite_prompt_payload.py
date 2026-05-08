from __future__ import annotations

import json
from typing import Any

REWRITE_REQUEST_TEXT = ("Rewrite all segment prompts with improved continuity and cinematic detail. "
                        "Keep count and ordering identical.")
DEFAULT_REWRITE_SEGMENT_COUNT = 6
REWRITE_MODE_NEW = "new_rollout"
REWRITE_MODE_EDIT_EXISTING = "edit_existing_rollout"
DEFAULT_REWRITE_ROLLOUT_ID = "current_rollout"
DEFAULT_REWRITE_ROLLOUT_LABEL = "Current rollout"


def normalize_prompt_window_prompts(values: list[Any] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        clean_item = item.strip()
        if not clean_item:
            continue
        normalized.append(clean_item)
    return normalized


def build_rewrite_user_payload(
    *,
    prompt_window_prompts: list[str],
    preset_id: str | None = None,
    preset_label: str | None = None,
    rewrite_instruction: str | None = None,
) -> dict[str, Any]:
    rollout_id = (preset_id or "").strip() or DEFAULT_REWRITE_ROLLOUT_ID
    rollout_label = (preset_label or "").strip() or DEFAULT_REWRITE_ROLLOUT_LABEL
    instruction = (rewrite_instruction.strip() if isinstance(rewrite_instruction, str) else "")
    if len(prompt_window_prompts) == 0:
        return {
            "mode": REWRITE_MODE_NEW,
            "request": REWRITE_REQUEST_TEXT,
            "user_instruction": instruction,
            "desired_segment_count": DEFAULT_REWRITE_SEGMENT_COUNT,
            "rollout_id_hint": rollout_id,
            "rollout_label_hint": rollout_label,
        }
    return {
        "mode": REWRITE_MODE_EDIT_EXISTING,
        "request": REWRITE_REQUEST_TEXT,
        "user_instruction": instruction,
        "current_rollout": {
            "id": rollout_id,
            "label": rollout_label,
            "segment_prompts": list(prompt_window_prompts),
        },
    }


def build_rewrite_request_body(
    *,
    system_prompt: str,
    prompt_window_prompts: list[str],
    preset_id: str | None,
    preset_label: str | None,
    rewrite_instruction: str | None,
    model: str,
    temperature: float,
    max_completion_tokens: int,
) -> dict[str, Any]:
    user_payload = build_rewrite_user_payload(
        prompt_window_prompts=prompt_window_prompts,
        preset_id=preset_id,
        preset_label=preset_label,
        rewrite_instruction=rewrite_instruction,
    )
    return {
        "model":
        model,
        "temperature":
        temperature,
        "max_completion_tokens":
        max_completion_tokens,
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
