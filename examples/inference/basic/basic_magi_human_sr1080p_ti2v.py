# SPDX-License-Identifier: Apache-2.0
"""Stub example for daVinci-MagiHuman SR-1080p TI2V — NOT YET PORTED.

Mirrors upstream `daVinci-MagiHuman/example/sr_1080p/run_TI2V.sh`. Combines
the SR-1080p two-stage local-window-attention flow
(`basic_magi_human_sr1080p.py`) with the TI2V image conditioning
(`basic_magi_human_ti2v.py`).
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "MagiHuman SR-1080p TI2V is not yet ported. Requires both the "
        "SR-1080p two-stage pipeline with local-window attention "
        "(see `basic_magi_human_sr1080p.py`) and the TI2V image-conditioning "
        "stage (see `basic_magi_human_ti2v.py`).",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
