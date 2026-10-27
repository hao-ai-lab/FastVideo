# SPDX-License-Identifier: Apache-2.0
"""Eight-forward video+audio generation with the FastH3 8-Step V2 checkpoint.

``FastVideo/FastVideo-FastH3-8-Step-V2`` is a DMD-distilled MiniMax H3 T2AV
student trained with video/audio scheduler shifts 10/3, VSA sparsity 0.8 and
64-token tiles, on the explicit rung ladder ``[999, 874, 749, 624, 500, 375,
250, 125]``. Its ``fastvideo_inference.json`` sidecar carries that ladder; the
pipeline loads it, checks the declared shifts against the checkpoint scheduler
configs, and runs exactly eight transformer forwards from nine sigma-grid
points. This script only sets those defaults; it does not change the
four-forward preview example.

``--steps`` must stay at 9 for this checkpoint: the trained ladder has eight
rungs, and a different grid is not a valid recipe for it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from . import basic_fasth3
except ImportError:
    # Direct script execution puts this directory, rather than ``examples``, on
    # sys.path. Keep both ``python file.py`` and module/importlib use working.
    import basic_fasth3  # type: ignore[no-redef]

MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2"
GRID_POINTS = 9  # nine sigma-grid points = eight DiT forwards
VSA_SPARSITY = 0.8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = basic_fasth3.build_parser(description=__doc__)
    parser.set_defaults(
        model_path=MODEL,
        output="outputs/fasth3_8step",
        steps=GRID_POINTS,
        vsa_sparsity=VSA_SPARSITY,
        vsa_tile_size=64,
    )
    args = basic_fasth3.validate_args(parser, parser.parse_args(argv))
    if args.steps != GRID_POINTS:
        parser.error(f"--steps must be {GRID_POINTS} for {MODEL}: the checkpoint's trained ladder has "
                     f"{GRID_POINTS - 1} rungs and the pipeline rejects any other grid")
    return args


def main() -> None:
    basic_fasth3.run(parse_args())


if __name__ == "__main__":
    main()
