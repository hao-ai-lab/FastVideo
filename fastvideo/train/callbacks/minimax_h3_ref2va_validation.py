# SPDX-License-Identifier: Apache-2.0
"""Validation adapter for raw MiniMax H3 Ref2VA dataset records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import (
    build_minimax_h3_references,
    parse_minimax_h3_ref2va_raw_sample,
)
from fastvideo.train.callbacks.validation import ValidationCallback


class MiniMaxH3Ref2VAValidationCallback(ValidationCallback):
    """Attach ordered raw references before running the standard validation loop."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("use_validation_media_conditioning", False)
        super().__init__(**kwargs)

    def _prepare_validation_batch(
        self,
        sampling_param: SamplingParam,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
    ) -> ForwardBatch:
        batch = super()._prepare_validation_batch(
            sampling_param,
            validation_batch,
            num_inference_steps,
        )
        manifest_path = Path(self.dataset_file).expanduser().resolve()
        sample = parse_minimax_h3_ref2va_raw_sample(
            validation_batch,
            manifest_path=manifest_path,
            # ValidationDataset adds fields such as prompt after loading the
            # strict raw document; the five schema fields remain authoritative.
            allow_extra_fields=True,
        )
        batch.references = build_minimax_h3_references(sample.references)
        return batch


__all__ = ["MiniMaxH3Ref2VAValidationCallback"]
