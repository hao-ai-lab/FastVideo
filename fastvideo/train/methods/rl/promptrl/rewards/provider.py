# SPDX-License-Identifier: Apache-2.0
"""Reward provider contract + built-in HTTP client.

The trainer depends only on the :class:`RewardProvider` protocol::

    score(samples: Sequence[RewardSample]) -> Sequence[RewardResult]

``RewardResult`` carries the composite ``score``, per-component
``details``, the ``sample_id`` and the ``request_id``.  The built-in
:class:`HttpRewardProvider` uploads videos to an external service
(e.g. VideoScore2 on a ninth GPU) with bounded retries and strict
response validation. Any timeout, duplicate/missing sample, non-finite
score, or cardinality mismatch raises :class:`RewardServiceError`.
Judge-specific adapters may explicitly neutralize an unscorable output
for the complete comparison group so it contributes no relative advantage.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from collections.abc import Sequence

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class RewardServiceError(RuntimeError):
    """Raised when the reward service fails a step consistently."""


@dataclass(frozen=True, slots=True)
class RewardSample:
    """One generated video submitted for scoring."""

    group_id: str
    request_id: str
    sample_id: str
    expected_group_size: int
    original_prompt: str
    reward_tag: str
    fps: int
    video_bytes: bytes
    video_filename: str = "sample.mp4"


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Score for one sample returned by a provider."""

    score: float
    details: dict[str, float]
    sample_id: str
    request_id: str


@runtime_checkable
class RewardProvider(Protocol):
    """Pluggable reward contract used by the PromptRL trainer."""

    def score(self, samples: Sequence[RewardSample]) -> Sequence[RewardResult]:
        """Score *samples*; exactly one result per input sample."""
        ...


def validate_reward_results(
    samples: Sequence[RewardSample],
    results: Sequence[RewardResult],
) -> None:
    """Enforce cardinality, identity, and finiteness of *results*.

    Raises :class:`RewardServiceError` on any mismatch so every rank
    fails the step consistently.
    """
    if len(results) != len(samples):
        raise RewardServiceError(f"Reward cardinality mismatch: sent {len(samples)} "
                                 f"sample(s), got {len(results)} result(s)")
    expected_ids = [s.sample_id for s in samples]
    got_ids = [r.sample_id for r in results]
    if sorted(got_ids) != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(got_ids))
        duplicate = sorted({g for g in got_ids if got_ids.count(g) > 1})
        raise RewardServiceError(f"Reward sample id mismatch: missing={missing}, "
                                 f"duplicates={duplicate}, got={sorted(got_ids)}")
    for result in results:
        if not isinstance(result.score, float | int) or not math.isfinite(float(result.score)):
            raise RewardServiceError(f"Non-finite reward score for sample "
                                     f"{result.sample_id!r}: {result.score!r}")
        for key, value in result.details.items():
            if not math.isfinite(float(value)):
                raise RewardServiceError(f"Non-finite reward detail {key!r} for sample "
                                         f"{result.sample_id!r}: {value!r}")


@dataclass(slots=True)
class HttpRewardProvider:
    """HTTP client for the PromptRL reward service."""

    endpoint_url: str
    timeout_sec: float = 300.0
    retries: int = 2
    score_path: str = "/v1/rewards:score"
    health_path: str = "/healthz"
    # Test hook: sleep between retries (seconds).
    retry_backoff_sec: float = 1.0
    _session: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.endpoint_url = self.endpoint_url.rstrip("/")

    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self._session is not None:
            return self._session
        # The module-level helpers create an independent Session per request,
        # avoiding shared mutable Session state across concurrent uploads.
        import requests
        return requests

    def health(self) -> bool:
        try:
            response = self._client().get(
                f"{self.endpoint_url}{self.health_path}",
                timeout=min(self.timeout_sec, 30.0),
            )
            return bool(response.status_code == 200)
        except Exception:
            return False

    def score(self, samples: Sequence[RewardSample]) -> Sequence[RewardResult]:
        if not samples:
            return []
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                results = self._score_once(samples)
                validate_reward_results(samples, results)
                return results
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                logger.warning(
                    "Reward request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.retries + 1,
                    exc,
                )
                if attempt < self.retries and self.retry_backoff_sec > 0:
                    time.sleep(self.retry_backoff_sec)
        raise RewardServiceError(f"Reward service failed after "
                                 f"{self.retries + 1} attempt(s): {last_error}") from last_error

    # ------------------------------------------------------------------

    def _score_once(self, samples: Sequence[RewardSample]) -> list[RewardResult]:
        # A service-side group does not score until every expected sample
        # arrives. Submit all samples owned by this rank concurrently so a
        # rank with multiple group slots cannot block on its first request.
        # ``executor.map`` preserves input order for deterministic validation.
        max_workers = min(len(samples), 32)
        with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="promptrl-reward",
        ) as executor:
            return list(executor.map(self._score_sample_once, samples))

    def _score_sample_once(self, sample: RewardSample) -> RewardResult:
        """POST one sample; the service batches the group server-side."""
        form = {
            "group_id": sample.group_id,
            "request_id": sample.request_id,
            "sample_id": sample.sample_id,
            "expected_group_size": str(sample.expected_group_size),
            "original_prompt": sample.original_prompt,
            "reward_tag": sample.reward_tag,
            "fps": str(sample.fps),
        }
        files = {
            "video": (sample.video_filename, sample.video_bytes, "video/mp4"),
        }
        response = self._client().post(
            f"{self.endpoint_url}{self.score_path}",
            data=form,
            files=files,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise RewardServiceError(f"Reward service returned HTTP {response.status_code}: "
                                     f"{response.text[:500]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RewardServiceError(f"Reward service returned non-JSON response: "
                                     f"{response.text[:500]}") from exc
        return _parse_reward_payload(payload, sample=sample)


def _parse_reward_payload(payload: Any, *, sample: RewardSample) -> RewardResult:
    if not isinstance(payload, dict):
        raise RewardServiceError(f"Reward payload must be a JSON object, got "
                                 f"{type(payload).__name__}")
    if payload.get("request_id") != sample.request_id:
        raise RewardServiceError(f"Reward payload request_id mismatch: expected "
                                 f"{sample.request_id!r}, got {payload.get('request_id')!r}")
    if payload.get("sample_id") != sample.sample_id:
        raise RewardServiceError(f"Reward payload sample_id mismatch: expected "
                                 f"{sample.sample_id!r}, got {payload.get('sample_id')!r}")
    raw_score = payload.get("score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise RewardServiceError(f"Reward payload has invalid score: {raw_score!r}") from exc
    raw_details = payload.get("details", {}) or {}
    if not isinstance(raw_details, dict):
        raise RewardServiceError(f"Reward payload details must be an object, got "
                                 f"{type(raw_details).__name__}")
    details: dict[str, float] = {}
    for key, value in raw_details.items():
        try:
            details[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise RewardServiceError(f"Reward detail {key!r} is not numeric: {value!r}") from exc
    return RewardResult(
        score=score,
        details=details,
        sample_id=str(payload["sample_id"]),
        request_id=str(payload["request_id"]),
    )
