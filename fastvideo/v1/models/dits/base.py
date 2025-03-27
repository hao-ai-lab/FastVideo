# SPDX-License-Identifier: Apache-2.0

from torch import nn

# TODO
class BaseDiT(nn.Module):
    _fsdp_shard_conditions = []
    attention_head_dim: int = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass
