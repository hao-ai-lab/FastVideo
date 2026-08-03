# SPDX-License-Identifier: Apache-2.0
"""HF-backed Qwen3-VL encoder used by MiniMax H3."""

from typing import Any

import torch
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.models.encoders.base import TextEncoder


def _module_dtype(module: nn.Module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else torch.float32


class MiniMaxH3Qwen3VLConditioner(TextEncoder):
    """Thin FastVideo adapter around Transformers' base Qwen3-VL model."""

    supports_hf_from_pretrained = True

    def __init__(self, config: MiniMaxH3Qwen3VLConfig, hf_model: nn.Module | None = None) -> None:
        super().__init__(config)
        self.hf_model = hf_model

    @classmethod
    def from_pretrained_local(
        cls,
        model_path: str,
        model_config: MiniMaxH3Qwen3VLConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> nn.Module:
        from transformers import Qwen3VLModel

        hf_model = Qwen3VLModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
        if device.type != "cpu":
            hf_model = hf_model.to(device)
        return cls(model_config, hf_model=hf_model).eval()

    @property
    def dtype(self) -> torch.dtype:
        if self.hf_model is None:
            return torch.float32
        return _module_dtype(self.hf_model)

    @property
    def num_hidden_layers(self) -> int | None:
        if self.hf_model is None:
            return None
        text_config = getattr(self.hf_model.config, "text_config", None)
        if isinstance(text_config, dict):
            return text_config.get("num_hidden_layers")
        return getattr(text_config, "num_hidden_layers", None)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        if self.hf_model is None:
            raise RuntimeError("The MiniMax H3 Qwen3-VL encoder has not been loaded.")
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        outputs = self.hf_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return BaseEncoderOutput(
            last_hidden_state=getattr(outputs, "last_hidden_state", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            attention_mask=attention_mask,
        )


EntryClass = MiniMaxH3Qwen3VLConditioner

__all__ = ["MiniMaxH3Qwen3VLConditioner"]
