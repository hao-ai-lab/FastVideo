# SPDX-License-Identifier: MIT
"""Vendored TAEHV (Tiny AutoEncoder for Hunyuan/Wan video latents).

Upstream: https://github.com/madebyollin/taehv -- see ``taehv.py``'s header for
provenance and ``LICENSE`` for the MIT license text.
"""

from fastvideo.third_party.taehv.taehv import TAEHV

__all__ = ["TAEHV"]
