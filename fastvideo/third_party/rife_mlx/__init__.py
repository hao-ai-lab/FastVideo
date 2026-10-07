# SPDX-License-Identifier: MIT
"""Vendored rife-mlx (RIFE frame interpolation on Apple MLX).

Upstream: https://github.com/xocialize/rife-mlx, itself an MLX port of
Practical-RIFE (https://github.com/hzwer/Practical-RIFE, (c) hzwer). Both are
MIT; see ``LICENSE`` for the text.

Only the inference path is vendored, which is what ``--fast`` calls:
``utils.weights.build_model`` and ``pipeline_mlx.interpolate_pair``. Upstream's
video I/O and torch-to-MLX conversion helpers are omitted, so this package
needs no ``av``, ``PIL``, or torch import to load. Weights are still fetched at
runtime from the ``mlx-community`` Hugging Face org, exactly as upstream does.
"""

from fastvideo.third_party.rife_mlx.config import DEFAULT_VERSION, VERSIONS, VersionConfig

__all__ = ["DEFAULT_VERSION", "VERSIONS", "VersionConfig"]
