# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 UniPC flow_shift vs the framework.

The framework selects the UniPC ``shift`` purely from the named resolution
bucket the (H, W) belongs to, via ``OmniSampleArgs._RESOLUTION_SHIFT_DEFAULTS``
(keyed by the VLM model size — Cosmos3-Nano uses the 8B backbone — and the
resolution string), NOT from the task (T2V/I2V/T2I share a shift at a given
resolution). FastVideo gets raw pixel ``height``/``width`` and must map back to
the same shift.

This pins ``Cosmos3DenoisingStage._flow_shift_for_resolution`` against the
framework's own tables: for every (resolution, aspect) entry in
``VIDEO_RES_SIZE_INFO`` whose resolution has an 8B shift default, the FastVideo
shift for that exact pixel size must equal the framework default.

The framework tables are the parity ORACLE.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_flow_shift_parity.py -q
"""
from __future__ import annotations

import pytest

# The official framework provides the parity oracle for the resolution->pixel
# tables. (``cosmos_framework.inference.args`` — which holds the shift constant —
# can't be imported here: it transitively requires ``multistorageclient``. The
# small shift table is mirrored verbatim below with its source location.)
_utils = pytest.importorskip(
    "cosmos_framework.data.vfm.utils",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from fastvideo.pipelines.stages.cosmos3_stages import (  # noqa: E402
    Cosmos3DenoisingStage,
)

pytestmark = [pytest.mark.local]

# Cosmos3-Nano's VLM backbone is Qwen3-VL-8B (checkpoint config.json).
_MODEL_SIZE = "8B"
# Verbatim from cosmos_framework.inference.args.OmniSampleArgs
# ._RESOLUTION_SHIFT_DEFAULTS (args.py:770), restricted to the 8B rows.
_SHIFT_DEFAULTS = {
    ("8B", "256"): 3.0,
    ("8B", "480"): 5.0,
    ("8B", "720"): 10.0,
    ("8B", "768"): 10.0,
    ("32B", "256"): 5.0,
    ("32B", "480"): 5.0,
    ("32B", "720"): 5.0,
    ("32B", "768"): 5.0,
}
_VIDEO_RES = _utils.VIDEO_RES_SIZE_INFO
_IMAGE_RES = _utils.IMAGE_RES_SIZE_INFO


def _cases():
    seen = set()
    for resolution, by_aspect in {**_VIDEO_RES, **_IMAGE_RES}.items():
        key = (_MODEL_SIZE, resolution)
        if key not in _SHIFT_DEFAULTS:
            continue
        expected = _SHIFT_DEFAULTS[key]
        for aspect, (a, b) in by_aspect.items():
            cid = f"{resolution}_{aspect.replace(',', '-')}_{a}x{b}"
            if cid in seen:
                continue
            seen.add(cid)
            yield pytest.param(a, b, expected, id=cid)


class TestCosmos3FlowShiftParity:

    @pytest.mark.parametrize(("dim_a", "dim_b", "expected_shift"), list(_cases()))
    def test_flow_shift_matches_framework(self, dim_a, dim_b, expected_shift):
        got = Cosmos3DenoisingStage._flow_shift_for_resolution(dim_a, dim_b)
        assert got == expected_shift, (
            f"shift for {dim_a}x{dim_b}: got {got}, framework default {expected_shift}")
