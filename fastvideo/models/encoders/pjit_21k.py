# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Linum Inc.
"""P-JiT's caption conditioning recipe for the native Qwen3.5 text tower.

P-JiT was trained on the concatenated outputs of blocks 7, 15 and 27 of
Qwen3.5-4B, including the chat-template tokens. The model and tokenizer are
loaded by the pipeline; this adapter owns only formatting and extraction.
"""
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class PJiTTextConditioner:

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        extraction_layers: Sequence[int] = (7, 15, 27),
        max_length: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("P-JiT max_length must be positive")
        if not extraction_layers:
            raise ValueError("P-JiT requires at least one extraction layer")
        self.model = model
        self.tokenizer = tokenizer
        self.extraction_layers = tuple(extraction_layers)
        self.max_length = max_length
        self.device = torch.device(device) if device is not None else None

        # Native encoder configs expose the resolved hybrid layer schedule.
        # Accept the HF text tower too, for direct reference comparisons.
        config = model.config
        arch = getattr(config, "arch_config", getattr(config, "text_config", config))
        layer_types = arch.resolved_layer_types() if hasattr(arch, "resolved_layer_types") else arch.layer_types
        full = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
        for index in self.extraction_layers:
            if index < 0 or index >= len(layer_types) or layer_types[index] != "full_attention":
                raise ValueError(f"Layer {index} is not a full-attention layer; Qwen3.5's full-attention layers "
                                 f"are {full}.")

    @torch.inference_mode()
    def __call__(self, text: list[str]) -> list[torch.Tensor]:
        if not text:
            return []
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) for caption in text
        ]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        device = self.device
        if device is None:
            device = getattr(self.model, "_fastvideo_input_device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("P-JiT's text encoder must return hidden states")
        results = []
        for index in range(len(text)):
            seq_len = int(inputs["attention_mask"][index].sum().item())
            # HF hidden_states[0] is the embedding output, hence block i + 1.
            layers = [
                hidden_states[layer + 1][index, :seq_len].to(torch.bfloat16) for layer in self.extraction_layers
            ]
            results.append(torch.cat(layers, dim=-1))
        return results
