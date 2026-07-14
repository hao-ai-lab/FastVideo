# SPDX-License-Identifier: Apache-2.0
"""Sole HF-import boundary for GLM-Image's AR encoder (lazy-wrapper exception E001)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class GlmImageARLoader(nn.Module):

    def __init__(self, model_path: str, processor_path: str | None = None,
                 *, torch_dtype: torch.dtype = torch.bfloat16,
                 trust_remote_code: bool = True) -> None:
        super().__init__()
        from transformers import (AutoProcessor,
                                  GlmImageForConditionalGeneration)
        logger.info("Loading GLM-Image AR encoder from %s", model_path)
        self._model = GlmImageForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if processor_path is not None:
            logger.info("Loading GLM-Image processor from %s", processor_path)
            self.processor = AutoProcessor.from_pretrained(
                processor_path, trust_remote_code=trust_remote_code)
        else:
            self.processor = None

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._model.generate(*args, **kwargs)

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor,
                           image_grid_thw: torch.Tensor) -> Any:
        return self._model.get_image_features(pixel_values, image_grid_thw)

    @torch.no_grad()
    def get_image_tokens(self, image_embeds: torch.Tensor,
                         image_grid_thw: torch.Tensor) -> torch.Tensor:
        return self._model.get_image_tokens(image_embeds, image_grid_thw)

    @property
    def config(self):  # type: ignore[no-untyped-def]
        return self._model.config

    @property
    def generation_config(self):  # type: ignore[no-untyped-def]
        return self._model.generation_config

    def to(self, *args, **kwargs):  # type: ignore[override]
        self._model = self._model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def eval(self):  # type: ignore[override]
        self._model = self._model.eval()
        return super().eval()
