# Copied from another repo, but I can't remember exactly which one.

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float = 0.9999,
            update_step_interval: int = 1,
            device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

        # TODO: add an automatic decay calculation based on this formula:
        # The impact of the last n steps can be calculated as:
        #     impact = 1-(decay^n)
        # The number of steps needed to reach a specific impact is:
        #     n = log_decay(1-impact)
        # The decay needed to reach a specific impact after n steps is:
        #     decay = (1-impact)^(1/n)

    def get_current_decay(self, optimization_step) -> float:
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        parameters = list(parameters)

        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # in place calculations to save memory
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.data)

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }
