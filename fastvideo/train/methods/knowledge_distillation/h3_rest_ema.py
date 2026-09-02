# SPDX-License-Identifier: Apache-2.0
"""Device-local EMA over only trainable H3 parameter shards."""

from __future__ import annotations

from contextlib import AbstractContextManager
import math
from typing import Any

import torch


class TrainableShardEMA:
    """EMA that preserves DTensor/FSDP placement and ignores frozen weights.

    FastH3 REST trains only a quality LoRA. Cloning the frozen 35B backbone is
    unnecessary; this object stores one float32 shadow for each trainable local
    parameter shard. Keeping shadows on the model device avoids a CPU-to-GPU
    copy of every LoRA shard for each REST EMA teacher forward.
    """

    def __init__(self, module: torch.nn.Module, *, decay: float) -> None:
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must satisfy 0 <= decay < 1, got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone().float()
        if not self.shadow:
            raise ValueError("TrainableShardEMA requires at least one trainable parameter")

    @staticmethod
    def _trainable_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        return {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        parameters = self._trainable_parameters(module)
        if set(parameters) != set(self.shadow):
            raise RuntimeError(
                "Trainable parameter set changed after EMA initialization: "
                f"missing={sorted(set(self.shadow) - set(parameters))}, "
                f"new={sorted(set(parameters) - set(self.shadow))}"
            )
        one_minus_decay = 1.0 - self.decay
        for name, parameter in parameters.items():
            shadow = self.shadow[name]
            value = parameter.detach().to(dtype=shadow.dtype)
            if shadow.shape != value.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: shadow={tuple(shadow.shape)}, "
                    f"parameter={tuple(value.shape)}"
                )
            shadow.mul_(self.decay).add_(value, alpha=one_minus_decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": torch.tensor(self.decay, dtype=torch.float64),
            "num_updates": torch.tensor(self.num_updates, dtype=torch.int64),
            "shadow": self.shadow,
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        observed_decay = float(torch.as_tensor(state_dict["decay"]).item())
        if not math.isclose(observed_decay, self.decay, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "EMA decay in checkpoint does not match the run config: "
                f"checkpoint={observed_decay}, config={self.decay}"
            )
        loaded_shadow = state_dict.get("shadow")
        if not isinstance(loaded_shadow, dict) or set(loaded_shadow) != set(self.shadow):
            raise ValueError(
                "EMA checkpoint parameter set mismatch: "
                f"checkpoint={sorted(loaded_shadow) if isinstance(loaded_shadow, dict) else type(loaded_shadow).__name__}, "
                f"model={sorted(self.shadow)}"
            )
        for name, target in self.shadow.items():
            raw_value = loaded_shadow[name]
            if not torch.is_tensor(raw_value):
                raise ValueError(
                    f"EMA checkpoint value for {name} is not a tensor: "
                    f"{type(raw_value).__name__}"
                )
            value = raw_value.to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"EMA checkpoint shape mismatch for {name}: "
                    f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                )
            target.copy_(value)
        self.num_updates = int(torch.as_tensor(state_dict["num_updates"]).item())

    class _ApplyContext(AbstractContextManager[torch.nn.Module]):
        def __init__(self, ema: "TrainableShardEMA", module: torch.nn.Module) -> None:
            self.ema = ema
            self.module = module
            self.saved: dict[str, torch.Tensor] = {}

        @torch.no_grad()
        def __enter__(self) -> torch.nn.Module:
            parameters = TrainableShardEMA._trainable_parameters(self.module)
            if set(parameters) != set(self.ema.shadow):
                raise RuntimeError("Trainable parameter set changed before EMA application")
            for name, parameter in parameters.items():
                self.saved[name] = parameter.detach().clone()
                parameter.copy_(self.ema.shadow[name].to(dtype=parameter.dtype))
            return self.module

        @torch.no_grad()
        def __exit__(self, exc_type, exc, traceback) -> bool:
            parameters = TrainableShardEMA._trainable_parameters(self.module)
            for name, value in self.saved.items():
                parameters[name].copy_(value)
            self.saved.clear()
            return False

    def apply_to_model(self, module: torch.nn.Module) -> _ApplyContext:
        """Temporarily place EMA trainable shards into ``module``."""
        return self._ApplyContext(self, module)


__all__ = ["TrainableShardEMA"]
