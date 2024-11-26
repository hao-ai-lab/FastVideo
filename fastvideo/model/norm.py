import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.ops.rms_norm import LigerRMSNormFunction

from liger_kernel.ops.layer_norm import LigerLayerNormFunction

class RMSNorm(nn.Module):
    # mbedding_dim, eps=eps, elementwise_affine=elementwise_affine)
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            weight = torch.ones(dim)
            self.register_buffer("weight", weight, persistent=False)
            
    def forward(self, hidden_states):
  
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.eps,
            0.0,
            "gemma",
            True,
        )
        
class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        else:
            weight = torch.ones(dim)
            bias = torch.zeros(dim) 
            self.register_buffer("weight", weight, persistent=False)
            self.register_buffer("bias", bias, persistent=False)
    def forward(self, hidden_states):
        return LigerLayerNormFunction.apply(
            hidden_states, self.weight, self.bias, self.eps
        )

class MochiRMSNormZero(nn.Module):
    r"""
    Adaptive RMS Norm used in Mochi.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, eps: float = 1e-5, elementwise_affine: bool = False
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        # TODO: There is a bug in HF. Remove hardcode
        self.norm = RMSNorm(hidden_dim//4, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None])

        return hidden_states, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)
    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x
            
class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
