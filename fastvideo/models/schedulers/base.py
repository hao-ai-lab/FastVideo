# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch


class BaseScheduler(ABC):
    timesteps: torch.Tensor
    order: int

    def __init__(self, *args, **kwargs) -> None:
        # Check if subclass has defined all required properties
        required_attributes = ['timesteps', 'order']

        for attr in required_attributes:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseScheduler must define '{attr}' property"
                )

    @abstractmethod
    def set_shift(self, shift: float) -> None:
        pass

    @abstractmethod
    def set_timesteps(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def scale_model_input(self,
                          sample: torch.Tensor,
                          timestep: int | None = None) -> torch.Tensor:
        pass
