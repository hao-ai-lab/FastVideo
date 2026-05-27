# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.models.encoders.base import TextEncoder


class T5EncoderModel(TextEncoder):
    
    supports_hf_from_pretrained: bool = True

    def __init__(self, config: T5Config, hf_model: Any | None = None) -> None:
        super().__init__(config)
        self.hf_model = hf_model

    @classmethod
    def from_pretrained_local(
        cls,
        model_path: str,
        config: T5Config,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "T5EncoderModel":
        from transformers import T5EncoderModel as HFT5EncoderModel

        kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        try:
            hf = HFT5EncoderModel.from_pretrained(model_path, dtype=dtype, **kwargs)
        except TypeError:
            # Backward-compatible fallback for older Transformers versions.
            hf = HFT5EncoderModel.from_pretrained(
                model_path, torch_dtype=dtype, **kwargs
            )

        hf = hf.eval().to(device=device, dtype=dtype)
        return cls(config=config, hf_model=hf).eval()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        if self.hf_model is None:
            raise RuntimeError(
                "T5EncoderModel(HF) is not initialized. Use "
                "`from_pretrained_local(...)` to construct a loaded instance."
            )

        out = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=bool(output_hidden_states)
            if output_hidden_states is not None
            else False,
            return_dict=True,
        )
        return BaseEncoderOutput(
            last_hidden_state=out.last_hidden_state,
            hidden_states=out.hidden_states if output_hidden_states else None,
            attention_mask=attention_mask,
        )
