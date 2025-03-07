from torch import nn


class BaseDiT(nn.Module):
    _fsdp_shard_conditions = []
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass