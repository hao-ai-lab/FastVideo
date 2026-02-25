# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal

from fastvideo.distillation.roles import RoleHandle


@dataclass(slots=True)
class ValidationRequest:
    """Method-provided validation configuration overrides.

    Validators are model-plugin-specific (e.g. Wan sampling), but should remain
    method-agnostic. A method may override key sampling parameters by passing a
    request object here.
    """

    sample_handle: RoleHandle | None = None
    sampling_steps: list[int] | None = None
    sampler_kind: Literal["ode", "sde"] | None = None
    sampling_timesteps: list[int] | None = None
    guidance_scale: float | None = None
    output_dir: str | None = None


class DistillValidator(ABC):
    @abstractmethod
    def log_validation(self, step: int, *, request: ValidationRequest | None = None) -> None:
        raise NotImplementedError
