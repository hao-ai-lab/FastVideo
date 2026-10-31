# SPDX-License-Identifier: Apache-2.0
"""Lightweight training-role abstraction.

``TrainRoleBase`` is the minimal contract every training role must
satisfy, whether the role wraps a diffusion transformer
(:class:`~fastvideo.train.models.base.ModelBase`) or a non-diffusion
model such as a language model prompt refiner.

A role owns:

* its checkpoint-visible modules (``checkpoint_modules``),
* its trainable parameter set (``trainable_parameters``),
* lifecycle hooks (``init_preprocessors``, ``on_train_start``),
* device metadata (``device``).

The builder, checkpointing, gradient clipping, and optimizer plumbing
all operate on this interface so non-diffusion roles can participate in
training without pretending to be a diffusion ``ModelBase``.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

from fastvideo.distributed import get_local_torch_device

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig


class TrainRoleBase(ABC):
    """Minimal per-role training interface.

    Subclasses must set ``_trainable`` and expose the modules that
    should be visible to DCP checkpointing and FSDP wrapping through
    :meth:`checkpoint_modules`.
    """

    _trainable: bool

    # ------------------------------------------------------------------
    # Device metadata
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The local training device for this rank."""
        return get_local_torch_device()

    # ------------------------------------------------------------------
    # Checkpoint / optimizer plumbing
    # ------------------------------------------------------------------

    def checkpoint_modules(self) -> dict[str, torch.nn.Module]:
        """Modules owned by this role, keyed by a stable module name.

        The keys appear in DCP checkpoints as ``roles.<role>.<name>`` and
        in the method's ``role_modules`` ModuleDict used for FSDP
        wrapping.  Return an empty dict for roles that hold no modules.
        """
        return {}

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters the role's optimizer should update."""
        params: list[torch.nn.Parameter] = []
        for module in self.checkpoint_modules().values():
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def init_preprocessors(  # noqa: B027
        self,
        training_config: TrainingConfig,
    ) -> None:
        """Load heavyweight resources (VAE, text encoders, dataloaders).

        Called only on roles the method designates as resource-owning.
        Default is a no-op.
        """

    def on_train_start(self) -> None:  # noqa: B027
        """Hook fired once before the first training step."""
