# SPDX-License-Identifier: Apache-2.0

from torch import nn


# TODO
class BaseDiT(nn.Module):
    _fsdp_shard_conditions: list = []
    attention_head_dim: int | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        pass
