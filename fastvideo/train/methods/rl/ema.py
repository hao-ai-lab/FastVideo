# SPDX-License-Identifier: Apache-2.0
"""Exponential Moving Average wrapper for RL training."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    """Maintains EMA copies of model parameters."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [
            p.clone().detach().to(device) for p in parameters
        ]
        self.temp_stored_parameters = None
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        return min(
            (1 + optimization_step)
            / (10 + optimization_step),
            self.decay,
        )

    @torch.no_grad()
    def step(
        self,
        parameters: Iterable[torch.nn.Parameter],
        optimization_step,
    ):
        parameters = list(parameters)
        one_minus_decay = (
            1 - self.get_current_decay(optimization_step)
        )

        if (
            optimization_step + 1
        ) % self.update_step_interval == 0:
            for ema_p, p in zip(
                self.ema_parameters,
                parameters,
                strict=True,
            ):
                if p.requires_grad:
                    if ema_p.device == p.device:
                        ema_p.add_(
                            one_minus_decay * (p - ema_p)
                        )
                    else:
                        p_copy = p.detach().to(ema_p.device)
                        p_copy.sub_(ema_p)
                        p_copy.mul_(one_minus_decay)
                        ema_p.add_(p_copy)
                        del p_copy

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        self.device = device
        self.ema_parameters = [
            (
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
            )
            for p in self.ema_parameters
        ]

    def copy_ema_to(
        self,
        parameters: Iterable[torch.nn.Parameter],
        store_temp: bool = True,
    ) -> None:
        if store_temp:
            self.temp_stored_parameters = [
                p.detach().cpu() for p in parameters
            ]
        parameters = list(parameters)
        for ema_p, p in zip(
            self.ema_parameters, parameters, strict=True
        ):
            p.data.copy_(ema_p.to(p.device).data)

    def copy_temp_to(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> None:
        for temp_p, p in zip(
            self.temp_stored_parameters,
            parameters,
            strict=True,
        ):
            p.data.copy_(temp_p.data)
        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get(
            "ema_parameters"
        )
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }
