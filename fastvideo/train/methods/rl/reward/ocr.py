# SPDX-License-Identifier: Apache-2.0
"""OCR-based reward for video-text alignment."""

from __future__ import annotations

import re

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward.utils import (
    prepare_images,
)

logger = init_logger(__name__)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(
                min(insertions, deletions, substitutions)
            )
        prev_row = curr_row
    return prev_row[-1]


def _extract_text_from_prompt(prompt: str) -> str:
    """Extract expected text from prompt (within quotes)."""
    match = re.search(r'["\'](.+?)["\']', prompt)
    if match:
        return match.group(1)
    return prompt


def video_ocr_score():
    """Return an OCR-based reward function (CPU)."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        msg = (
            "paddleocr not installed. "
            "Install via: pip install paddleocr"
        )
        raise ImportError(msg) from exc

    ocr = PaddleOCR(
        use_angle_cls=True, lang="en", use_gpu=False
    )

    def _score(images, prompts, metadata, only_strict=False):
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]

            expected = _extract_text_from_prompt(
                prompts[b] if prompts else ""
            ).lower()

            # Sample every 4th frame.
            sample_indices = list(
                range(0, len(frames), 4)
            )
            if not sample_indices:
                sample_indices = [0]

            best_score = 0.0
            for idx in sample_indices:
                frame = frames[idx]
                result = ocr.ocr(frame, cls=True)
                detected = ""
                if result and result[0]:
                    texts = [
                        line[1][0]
                        for line in result[0]
                        if line[1]
                    ]
                    detected = " ".join(texts).lower()

                if not expected:
                    score = 1.0 if detected else 0.0
                elif not detected:
                    score = 0.0
                else:
                    dist = _levenshtein_distance(
                        detected, expected
                    )
                    max_len = max(
                        len(detected), len(expected)
                    )
                    score = 1.0 - (dist / max_len)
                best_score = max(best_score, score)

            batch_scores.append(best_score)

        reward = torch.tensor(batch_scores).float()
        return {"avg": reward}, {}

    return _score
