# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/input_quant_fp8.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    GroupShape,
    get_fp8_min_max,
)

_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


class QuantFP8(nn.Module):
    """FP8 activation quantization (per-tensor or per-token, static or dynamic)."""

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
    ):
        super().__init__()
        self.static = static
        self.group_shape = group_shape
        self.num_token_padding = num_token_padding
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN

        if not static:
            assert group_shape in (
                GroupShape.PER_TOKEN, GroupShape.PER_TENSOR), (
                    "Only per-token or per-tensor scales are supported for "
                    "dynamic non-group quantization.")

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (scale is not None) == self.static

        if scale is None:
            if self.use_per_token_if_dynamic:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

        out = (x.to(torch.float32) / scale.to(torch.float32))
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(FP8_DTYPE)

        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale
