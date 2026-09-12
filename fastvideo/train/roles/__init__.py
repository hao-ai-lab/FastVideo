# SPDX-License-Identifier: Apache-2.0
"""Training-role abstractions and non-diffusion role implementations."""

from fastvideo.train.roles.base import TrainRoleBase

__all__ = ["QwenPromptRefinerRole", "TrainRoleBase"]


def __getattr__(name: str):
    # Lazy: importing the Qwen role pulls in transformers/peft.
    if name == "QwenPromptRefinerRole":
        from fastvideo.train.roles.qwen_refiner import QwenPromptRefinerRole

        return QwenPromptRefinerRole
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
