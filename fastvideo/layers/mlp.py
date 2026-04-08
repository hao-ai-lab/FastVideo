# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import QuantizationConfig


class MLP(nn.Module):
    """
    MLP for DiT blocks, NO gated linear units
    """

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: torch.dtype | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.fc_in = ReplicatedLinear(
            input_dim,
            mlp_hidden_dim,  # For activation func like SiLU that need 2x width
            bias=bias,
            params_dtype=dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        if quant_config is not None:
            quant_method = self.fc_in.quant_config.get_quant_method(self.fc_in, f"{prefix}.fc_in")
            if quant_method is not None:
                quant_method.process_weights_after_loading(self.fc_in)

        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = ReplicatedLinear(mlp_hidden_dim,
                                       output_dim,
                                       bias=bias,
                                       params_dtype=dtype,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.fc_out")
        if quant_config is not None:
            quant_method = self.fc_out.quant_config.get_quant_method(self.fc_out, f"{prefix}.fc_out")
            if quant_method is not None:
                quant_method.process_weights_after_loading(self.fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x