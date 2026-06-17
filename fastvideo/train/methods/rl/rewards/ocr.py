# SPDX-License-Identifier: Apache-2.0
"""OCR-based video/text reward."""

from __future__ import annotations

import re

import torch

from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array


def _levenshtein_distance(s1: str, s2: str) -> int:
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
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _extract_text_from_prompt(prompt: str) -> str:
    match = re.search(r'["\'](.+?)["\']', prompt)
    return match.group(1) if match else prompt


class VideoOCRScorer:
    """Score text rendering quality with PaddleOCR."""

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError("video_ocr reward requires paddleocr. Install with `pip install paddleocr`.") from exc
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        images_np = media_to_uint8_array(media)
        batch_scores = []
        for sample_idx, sample in enumerate(images_np):
            frames = sample if sample.ndim == 4 else sample[None]
            expected = _extract_text_from_prompt(prompts[sample_idx] if sample_idx < len(prompts) else "").lower()
            sample_indices = list(range(0, len(frames), 4)) or [0]
            best_score = 0.0
            for idx in sample_indices:
                result = self.ocr.ocr(frames[idx], cls=True)
                detected = ""
                if result and result[0]:
                    detected = " ".join(line[1][0] for line in result[0] if line[1]).lower()
                if not expected:
                    score = 1.0 if detected else 0.0
                elif not detected:
                    score = 0.0
                else:
                    distance = _levenshtein_distance(detected, expected)
                    score = 1.0 - (distance / max(len(detected), len(expected)))
                best_score = max(best_score, score)
            batch_scores.append(best_score)
        return torch.tensor(batch_scores, device=media.device, dtype=torch.float32)
