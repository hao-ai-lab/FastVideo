# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from fastvideo.v1.platforms import _Backend


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    attention_head_dim: int | None = None
    _param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    # always supports torch_sdpa
    _supported_attention_backends: Tuple[_Backend,
                                         ...] = (_Backend.TORCH_SDPA, )

    def __init_subclass__(cls) -> None:
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
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        pass

    def __post_init__(self) -> None:
        required_attrs = ["hidden_size", "num_attention_heads"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )

    @property
    def supported_attention_backends(self) -> Tuple[_Backend, ...]:
        return self._supported_attention_backends
