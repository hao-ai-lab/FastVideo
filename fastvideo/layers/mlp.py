from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.activation import get_act_fn
import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """
    MLP for DiT blocks, NO gated linear units
    TODO: add Tensor Parallel
    """
    
    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dim: int,
        output_dim: int = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.fc_in = ReplicatedLinear(
            input_dim,
            mlp_hidden_dim,  # For activation functions like SiLU that need 2x width
            bias=bias,
            params_dtype=dtype
        )
        
        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = ReplicatedLinear(
            mlp_hidden_dim,
            output_dim,
            bias=bias,
            params_dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x

