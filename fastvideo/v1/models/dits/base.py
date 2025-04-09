# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch import nn


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    attention_head_dim: int | None = None
    _param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int

    def __init_subclass__(cls):
        required_class_attrs = [
            "_fsdp_shard_conditions", "_param_names_mapping"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                guidance=None,
                **kwargs) -> torch.Tensor:
        pass

    def __post_init__(self):
        required_attrs = ["hidden_size", "num_attention_heads"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )
