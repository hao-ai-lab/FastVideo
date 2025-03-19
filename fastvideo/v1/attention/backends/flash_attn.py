import torch
from typing import Optional

from fastvideo.v1.attention.backends.abstract import AttentionImpl

from flash_attn import flash_attn_func

class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        dropout_rate: float,
        causal: bool,
        softmax_scale: float,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_rate,
            softmax_scale=self.softmax_scale,
            causal=self.causal
        )
        return output