#!/usr/bin/env bash
# Canonical Slurm CI selection for the transformer lane.
set -euo pipefail

# The existing block reference records an absent FASTVIDEO_FA4 (FA2). Keep
# that reference identity; the component lane also selects FA2 explicitly.
env -u FASTVIDEO_FA4 pytest ./fastvideo/tests/golden_gate/test_wan_t2v.py -xvs
pytest ./fastvideo/tests/golden_gate/test_wan_causal.py -xvs
exec pytest ./fastvideo/tests/transformers -vs
