# SPDX-License-Identifier: Apache-2.0
"""Stub example for daVinci-MagiHuman DMD-2 distilled TI2V — NOT YET PORTED.

Mirrors upstream `daVinci-MagiHuman/example/distill/run_TI2V.sh`. Same arch
as `magi_human_distill` (8 steps, no CFG, ~30 GB bf16 weights) but in
TI2V mode — see `basic_magi_human_ti2v.py` for the missing pipeline-side
work that applies to both base and distill TI2V variants.
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "MagiHuman distilled TI2V is not yet ported. See "
        "`basic_magi_human_ti2v.py` for the required image-conditioning "
        "pipeline work and `tests/local_tests/magi-human.md` for the "
        "port-status journal.",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
