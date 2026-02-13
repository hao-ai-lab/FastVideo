# SPDX-License-Identifier: Apache-2.0
# Minimal utils for Stable Audio DiT (no torch.compile by default).


def compile(function, *args, **kwargs):
    """No-op compile for stable-audio transformer; avoids torch.compile deps."""
    return function
