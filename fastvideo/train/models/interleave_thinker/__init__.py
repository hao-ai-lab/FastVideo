# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker training model adapters."""

from fastvideo.train.models.interleave_thinker.critic import (
    INTERLEAVE_CRITIC_PROMPT,
    InterleaveThinkerCriticModel,
)

__all__ = ["INTERLEAVE_CRITIC_PROMPT", "InterleaveThinkerCriticModel"]
