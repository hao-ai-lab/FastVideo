from typing import Tuple

from torch import nn

from fastvideo.v1.platforms import _Backend


class BaseEncoder(nn.Module):
    _supported_attention_backends: Tuple[_Backend,
                                         ...] = (_Backend.TORCH_SDPA, )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    def forward(self, *args, **kwargs):
        pass

    @property
    def supported_attention_backends(self) -> Tuple[_Backend, ...]:
        return self._supported_attention_backends
