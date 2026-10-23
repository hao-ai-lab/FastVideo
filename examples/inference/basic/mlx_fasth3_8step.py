# SPDX-License-Identifier: Apache-2.0
"""Eight-forward FastH3 8-Step V2 generation on Apple Silicon.

``FastVideo/FastVideo-FastH3-8-Step-V2`` trains eight DMD forwards at video
shift 10 and audio shift 3, VSA sparsity 0.8, and 64-token tiles. Point
``--model-root`` at a snapshot that contains that checkpoint's
``fastvideo_inference.json`` plus the shared VAE, audio VAE, text encoder, and
tokenizer. Point ``--mlx-checkpoint`` at an MLX DiT converted from this
checkpoint (the AdaLN cache must be the contract ladder, not the four-step
uniform grid).

``--steps`` accepts 8 transformer forwards or 9 sigma-grid points. Either
value runs the trained rungs. The four-step preview entrypoint is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    from . import mlx_fasth3
except ImportError:
    import mlx_fasth3  # type: ignore[no-redef]

FORWARDS = 8
GRID_POINTS = FORWARDS + 1


def parse_args(argv: Sequence[str] | None = None):
    parser = mlx_fasth3.build_parser()
    parser.set_defaults(steps=FORWARDS, vsa=True, vsa_sparsity=0.8, vsa_tile_size=64)
    args = parser.parse_args(argv)
    if args.steps not in (FORWARDS, GRID_POINTS):
        parser.error(f"--steps must be {FORWARDS} or {GRID_POINTS} for FastH3 8-Step V2")
    return args


def main() -> None:
    mlx_fasth3.run(parse_args())


if __name__ == "__main__":
    main()
