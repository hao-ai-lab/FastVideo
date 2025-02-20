# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn


class MochiModulatedRMSNorm(nn.Module):

    def __init__(self, eps: float):
        super().__init__()

        self.eps = eps

    def forward(self, hidden_states, scale=None):
        hidden_states_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if scale is not None:
            hidden_states = hidden_states * scale

        hidden_states = hidden_states.to(hidden_states_dtype)

        return hidden_states


class MochiRMSNorm(nn.Module):

    def __init__(self, dim, eps: float, elementwise_affine=True):
        super().__init__()

        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        hidden_states_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        hidden_states = hidden_states.to(hidden_states_dtype)

        return hidden_states


class MochiLayerNormContinuous(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps=1e-5,
        bias=True,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)
        self.norm = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x, (1 + scale.unsqueeze(1).to(torch.float32)))

        return x.to(input_dtype)


class MochiRMSNormZero(nn.Module):
    r"""
    Adaptive RMS Norm used in Mochi.
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.norm = MochiModulatedRMSNorm(eps=eps)

    def forward(self, hidden_states: torch.Tensor,
                emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states_dtype = hidden_states.dtype

        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)

        hidden_states = self.norm(hidden_states, (1 + scale_msa[:, None].to(torch.float32)))
        hidden_states = hidden_states.to(hidden_states_dtype)

        return hidden_states, gate_msa, scale_mlp, gate_mlp
