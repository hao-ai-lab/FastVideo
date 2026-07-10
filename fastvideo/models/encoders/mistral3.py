# SPDX-License-Identifier: Apache-2.0
"""HF-backed Mistral3 text encoder wrapper for full Flux2."""
from typing import Any

import torch
from torch import nn

from fastvideo.configs.models.encoders.mistral3 import Mistral3TextConfig
from fastvideo.models.encoders.base import TextEncoder


class Mistral3ForConditionalGeneration(TextEncoder):
    """Loads the Transformers Mistral3 implementation for Flux2 text encoding."""

    supports_hf_from_pretrained = True

    def __init__(self, config: Mistral3TextConfig) -> None:
        super().__init__(config)
        self.config = config

    @classmethod
    def from_pretrained_local(
        cls,
        model_path: str,
        model_config: Mistral3TextConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> nn.Module:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
        if device.type != "cpu":
            model = model.to(device)
        return model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Mistral3ForConditionalGeneration is loaded through Transformers "
            "via from_pretrained_local()."
        )


EntryClass = Mistral3ForConditionalGeneration
