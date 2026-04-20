# SPDX-License-Identifier: Apache-2.0
"""Optional prompt safety filter.

Uses a fastText classifier to score prompts against a banned-content
rubric. Only loaded when ``ServeConfig.streaming.safety.enabled`` is
True and fastText is installed — users who don't need it see no
runtime cost.

Install: ``pip install fastvideo[prompt-safety]`` (ships fasttext as an
optional extra) or install fasttext directly.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SafetyDecision(enum.Enum):
    ALLOW = "allow"
    BLOCK = "block"
    UNAVAILABLE = "unavailable"
    """Returned when the classifier can't run (not configured, fastText
    missing). Safety is opt-in; the server treats ``UNAVAILABLE`` as
    ``ALLOW`` but logs it so operators know the filter is off."""


@dataclass
class SafetyResult:
    prompt: str
    decision: SafetyDecision
    score: float = 0.0
    label: str | None = None
    reason: str | None = None


class PromptSafetyFilter:
    """Minimal fastText-backed prompt safety filter.

    Loads the classifier lazily on first use so the streaming server
    can construct the filter eagerly at startup without paying the
    model-load cost when safety is disabled.
    """

    def __init__(
        self,
        *,
        classifier_path: str | None,
        enabled: bool = True,
        block_threshold: float = 0.5,
    ) -> None:
        self._classifier_path = classifier_path
        self._enabled = enabled
        self._block_threshold = block_threshold
        self._model: Any | None = None
        self._load_attempted = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._classifier_path is not None

    def classify(self, prompt: str) -> SafetyResult:
        if not self.enabled:
            return SafetyResult(
                prompt=prompt,
                decision=SafetyDecision.UNAVAILABLE,
                reason="safety filter not enabled",
            )
        model = self._ensure_loaded()
        if model is None:
            return SafetyResult(
                prompt=prompt,
                decision=SafetyDecision.UNAVAILABLE,
                reason="fastText model unavailable",
            )
        try:
            labels, probs = model.predict(prompt.replace("\n", " "), k=1)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("safety: classifier failed: %s", exc)
            return SafetyResult(
                prompt=prompt,
                decision=SafetyDecision.UNAVAILABLE,
                reason=f"classifier error: {exc}",
            )
        label = labels[0].removeprefix("__label__") if labels else None
        score = float(probs[0]) if len(probs) else 0.0
        decision = (SafetyDecision.BLOCK if
                    (label == "unsafe" and score >= self._block_threshold) else SafetyDecision.ALLOW)
        return SafetyResult(
            prompt=prompt,
            decision=decision,
            score=score,
            label=label,
        )

    def _ensure_loaded(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None
        self._load_attempted = True
        if self._classifier_path is None:
            return None
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("safety: fasttext not installed; safety filter disabled. "
                           "Install fastvideo[prompt-safety] to enable.")
            return None
        try:
            self._model = fasttext.load_model(self._classifier_path)
        except Exception as exc:  # pragma: no cover - requires real model
            logger.warning("safety: failed to load %s: %s", self._classifier_path, exc)
            return None
        return self._model


def first_blocked(
    filter_: PromptSafetyFilter,
    prompts: list[str],
) -> SafetyResult | None:
    """Return the first prompt the filter blocks, or ``None``."""
    for prompt in prompts:
        result = filter_.classify(prompt)
        if result.decision is SafetyDecision.BLOCK:
            return result
    return None


__all__ = [
    "PromptSafetyFilter",
    "SafetyDecision",
    "SafetyResult",
    "first_blocked",
]
