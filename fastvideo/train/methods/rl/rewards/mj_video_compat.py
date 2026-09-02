# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for the pinned MJ-VIDEO source tree."""

from __future__ import annotations


def install_mj_video_transformers_compat() -> None:
    """Provide a removed, unused Llama documentation constant.

    The pinned MJ-VIDEO ``moe_reward.py`` imports this private docstring value
    but does not use it in model execution. Transformers 5 removed the value.
    Restoring an empty documentation string changes no weights or forward math;
    all functional incompatibilities remain hard failures in the GPU preflight.
    """
    from transformers.models.llama import modeling_llama

    if getattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING", None) is None:
        setattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING", "")


__all__ = ["install_mj_video_transformers_compat"]
