# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker critic reward utilities.

Adapted from InterleaveThinker's
``train/EasyR1/verl/reward_function/interleave_thinker_reward.py``. The
networked edit API and Gemini scorer are intentionally left outside this module
so FastVideo training methods can inject those services without making reward
parsing depend on external credentials.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

import torch

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_TAG_SPACE_PATTERN = re.compile(r"\s*(<|>|/)\s*")


@dataclass(frozen=True, slots=True)
class InterleaveThinkerAnswer:
    """Parsed critic answer payload."""

    previous_step_success: bool
    refine_prompt: str


@dataclass(frozen=True, slots=True)
class InterleaveThinkerEditRequest:
    """Metadata needed by an external image-edit scorer."""

    index: int
    origin_prompt: str
    previous_prompt: str
    refine_prompt: str
    origin_image_path: str | None
    previous_image_path: str | None
    previous_step_success: bool
    previous_semantic_score: float
    previous_quality_score: float


@dataclass(frozen=True, slots=True)
class InterleaveThinkerEditScore:
    """Edit scorer output.

    ``semantic_score`` and ``quality_score`` are absolute post-edit scores on
    the same 0-10 scale used by InterleaveThinker. ``semantic_reward`` and
    ``quality_reward`` are normalized reward components in [0, 1]. Callers can
    provide either absolute scores or normalized rewards.
    """

    semantic_score: float | None = None
    quality_score: float | None = None
    semantic_reward: float | None = None
    quality_reward: float | None = None


@dataclass(frozen=True, slots=True)
class InterleaveThinkerRewardResult:
    """One scored InterleaveThinker rollout."""

    overall: float
    format_reward: float
    judge_accuracy_reward: float
    edited_image_reward_semantic: float
    edited_image_reward_quality: float
    predicted_previous_step_success: bool | None
    refine_prompt: str
    index: int

    def as_dict(self) -> dict[str, float]:
        return {
            "overall": float(self.overall),
            "format_reward": float(self.format_reward),
            "judge_accuracy_reward": float(self.judge_accuracy_reward),
            "edited_image_reward_semantic": float(self.edited_image_reward_semantic),
            "edited_image_reward_quality": float(self.edited_image_reward_quality),
            "idx": float(self.index),
        }


EditScoreProvider = Callable[[InterleaveThinkerEditRequest], InterleaveThinkerEditScore | Mapping[str, Any] | None]


def normalize_interleave_response(response: str) -> str:
    """Match upstream tag-spacing normalization before parsing."""
    return _TAG_SPACE_PATTERN.sub(r"\1", str(response or ""))


def _jsonish_loads(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def extract_interleave_answer(response: str) -> InterleaveThinkerAnswer | None:
    """Extract the ``<answer>`` JSON object expected by InterleaveThinker."""
    match = _ANSWER_PATTERN.search(normalize_interleave_response(response))
    if not match:
        return None
    payload = _jsonish_loads(match.group(1))
    if payload is None:
        return None
    previous_step_success = payload.get("previous_step_success")
    refine_prompt = payload.get("refine_prompt")
    if not isinstance(previous_step_success, bool) or not isinstance(refine_prompt, str):
        return None
    return InterleaveThinkerAnswer(
        previous_step_success=previous_step_success,
        refine_prompt=refine_prompt,
    )


def interleave_format_reward(response: str) -> float:
    """Return 1 when response has valid ``<think>`` then ``<answer>`` format."""
    normalized = normalize_interleave_response(response)
    think_match = _THINK_PATTERN.search(normalized)
    answer_match = _ANSWER_PATTERN.search(normalized)
    if think_match is None or answer_match is None:
        return 0.0
    if think_match.start() >= answer_match.start():
        return 0.0
    return 1.0 if extract_interleave_answer(normalized) is not None else 0.0


def interleave_judge_accuracy_reward(
    predicted_previous_step_success: bool | None,
    ground_truth_previous_step_success: bool | None,
) -> float:
    """Reward correct previous-step success prediction."""
    if predicted_previous_step_success is None or ground_truth_previous_step_success is None:
        return 0.0
    return 1.0 if bool(predicted_previous_step_success) == bool(ground_truth_previous_step_success) else 0.0


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return float(value.detach().cpu())
    return float(value)


def _mapping_get_bool(mapping: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if isinstance(value, bool):
                return bool(value)
    return None


def _ground_truth(mapping: Mapping[str, Any]) -> tuple[bool | None, float, float]:
    raw = mapping.get("ground_truth", mapping.get("evaluation", {}))
    if isinstance(raw, str):
        raw = _jsonish_loads(raw) or {}
    if not isinstance(raw, Mapping):
        return None, 0.0, 0.0
    success = _mapping_get_bool(raw, "success", "previous_step_success")
    return (
        success,
        _coerce_float(raw.get("semantics", raw.get("semantic_score")), default=0.0),
        _coerce_float(raw.get("quality", raw.get("quality_score")), default=0.0),
    )


def _make_edit_request(
    item: Mapping[str, Any],
    *,
    index: int,
    answer: InterleaveThinkerAnswer | None,
    previous_success: bool,
    previous_semantics: float,
    previous_quality: float,
) -> InterleaveThinkerEditRequest:
    previous_prompt = str(item.get("previous_prompt", item.get("rewritten_prompt", "")) or "")
    refine_prompt = answer.refine_prompt if answer is not None and answer.refine_prompt else previous_prompt
    return InterleaveThinkerEditRequest(
        index=index,
        origin_prompt=str(item.get("origin_prompt", item.get("prompt", "")) or ""),
        previous_prompt=previous_prompt,
        refine_prompt=refine_prompt,
        origin_image_path=item.get("origin_image_path"),
        previous_image_path=item.get("previous_image_path", item.get("edited_image_path")),
        previous_step_success=previous_success,
        previous_semantic_score=previous_semantics,
        previous_quality_score=previous_quality,
    )


def _coerce_edit_score(raw: InterleaveThinkerEditScore | Mapping[str, Any] | None) -> InterleaveThinkerEditScore | None:
    if raw is None:
        return None
    if isinstance(raw, InterleaveThinkerEditScore):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"edit score must be a mapping or InterleaveThinkerEditScore, got {type(raw).__name__}")
    return InterleaveThinkerEditScore(
        semantic_score=raw.get("semantic_score", raw.get("semantics")),
        quality_score=raw.get("quality_score", raw.get("quality")),
        semantic_reward=raw.get("semantic_reward", raw.get("edited_image_reward_semantic")),
        quality_reward=raw.get("quality_reward", raw.get("edited_image_reward_quality")),
    )


def _normalized_edit_rewards(
    score: InterleaveThinkerEditScore | None,
    *,
    previous_semantics: float,
    previous_quality: float,
    fallback: float,
) -> tuple[float, float]:
    if score is None:
        return fallback, fallback
    semantic_reward = score.semantic_reward
    quality_reward = score.quality_reward
    if semantic_reward is None and score.semantic_score is not None:
        semantic_reward = ((_coerce_float(score.semantic_score) - previous_semantics) / 10.0 + 1.0) / 2.0
    if quality_reward is None and score.quality_score is not None:
        quality_reward = ((_coerce_float(score.quality_score) - previous_quality) / 10.0 + 1.0) / 2.0
    return (
        _coerce_float(semantic_reward, default=fallback),
        _coerce_float(quality_reward, default=fallback),
    )


class InterleaveThinkerRewardScorer:
    """Batch scorer for InterleaveThinker critic outputs.

    The default weights match upstream InterleaveThinker's ``compute_score``:
    ``0.5 * format + 0.5 * (0.2 * judge + 0.6 * semantic + 0.2 * quality)``.
    """

    def __init__(
        self,
        *,
        format_weight: float = 0.5,
        judge_accuracy_weight: float = 0.2,
        semantic_weight: float = 0.6,
        quality_weight: float = 0.2,
        fallback_edit_reward: float = 0.5,
        edit_scorer: EditScoreProvider | None = None,
    ) -> None:
        self.format_weight = float(format_weight)
        self.judge_accuracy_weight = float(judge_accuracy_weight)
        self.semantic_weight = float(semantic_weight)
        self.quality_weight = float(quality_weight)
        self.fallback_edit_reward = float(fallback_edit_reward)
        self.edit_scorer = edit_scorer
        if not 0.0 <= self.format_weight <= 1.0:
            raise ValueError("format_weight must be in [0, 1]")
        inner_total = self.judge_accuracy_weight + self.semantic_weight + self.quality_weight
        if inner_total <= 0.0:
            raise ValueError("At least one non-format reward weight must be positive")

    def __call__(
        self,
        reward_inputs: Sequence[Mapping[str, Any]],
    ) -> list[InterleaveThinkerRewardResult]:
        results: list[InterleaveThinkerRewardResult] = []
        inner_total = self.judge_accuracy_weight + self.semantic_weight + self.quality_weight
        for index, item in enumerate(reward_inputs):
            response = str(item.get("response", "") or "")
            answer = extract_interleave_answer(response)
            format_reward = interleave_format_reward(response)
            previous_success, previous_semantics, previous_quality = _ground_truth(item)
            predicted_success = answer.previous_step_success if answer is not None else None
            judge_reward = interleave_judge_accuracy_reward(predicted_success, previous_success)
            edit_request = _make_edit_request(
                item,
                index=index,
                answer=answer,
                previous_success=bool(previous_success),
                previous_semantics=previous_semantics,
                previous_quality=previous_quality,
            )
            edit_score = _coerce_edit_score(item.get("edit_score", item.get("edit_scores")))
            if edit_score is None and self.edit_scorer is not None:
                edit_score = _coerce_edit_score(self.edit_scorer(edit_request))
            semantic_reward, quality_reward = _normalized_edit_rewards(
                edit_score,
                previous_semantics=previous_semantics,
                previous_quality=previous_quality,
                fallback=self.fallback_edit_reward,
            )
            non_format_reward = (
                self.judge_accuracy_weight * judge_reward + self.semantic_weight * semantic_reward +
                self.quality_weight * quality_reward
            ) / inner_total
            overall = self.format_weight * format_reward + (1.0 - self.format_weight) * non_format_reward
            results.append(
                InterleaveThinkerRewardResult(
                    overall=float(overall),
                    format_reward=float(format_reward),
                    judge_accuracy_reward=float(judge_reward),
                    edited_image_reward_semantic=float(semantic_reward),
                    edited_image_reward_quality=float(quality_reward),
                    predicted_previous_step_success=predicted_success,
                    refine_prompt=edit_request.refine_prompt,
                    index=index,
                ))
        return results

    def as_tensors(
        self,
        reward_inputs: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        results = self(reward_inputs)
        names = [
            "overall",
            "format_reward",
            "judge_accuracy_reward",
            "edited_image_reward_semantic",
            "edited_image_reward_quality",
        ]
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            tensors[name] = torch.tensor([getattr(result, name) for result in results],
                                         device=device,
                                         dtype=torch.float32)
        tensors["avg"] = tensors["overall"]
        return tensors


def score_interleave_thinker_rewards(
    reward_inputs: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Convenience wrapper matching upstream's list-of-dicts return shape."""
    scorer = InterleaveThinkerRewardScorer(**kwargs)
    return [result.as_dict() for result in scorer(reward_inputs)]


__all__ = [
    "EditScoreProvider",
    "InterleaveThinkerAnswer",
    "InterleaveThinkerEditRequest",
    "InterleaveThinkerEditScore",
    "InterleaveThinkerRewardResult",
    "InterleaveThinkerRewardScorer",
    "extract_interleave_answer",
    "interleave_format_reward",
    "interleave_judge_accuracy_reward",
    "normalize_interleave_response",
    "score_interleave_thinker_rewards",
]
