#!/bin/bash
# Step 04 — MERGE. Graft the overfit's co-adapted track pathway (track_encoder + patch-embed track
# slot [36:52]) onto a PRISTINE 14B base, discarding the overfit's base degradation. CPU only,
# ~62GB RAM -> run on a compute node. Needs 03_export.sh to have produced the overfit diffusers dir.
set -uo pipefail
cd ~/FastVideo

BASE=${BASE:-/home/hal-kevin/models/Wan2.1-I2V-14B-720P-Diffusers}
SRC=${SRC:-/home/hal-kevin/models/overfit14b_export}     # produced by 03_export.sh
OUT=${OUT:-/home/hal-kevin/models/trackwan_14b_i2v_d64_merged_from_overfit_bias}
ID_DIM=${ID_DIM:-64}

[ -d "$SRC/transformer" ] || { echo "[merge] $SRC/transformer not found — run 03_export.sh first" >&2; exit 1; }

# --track-src + --pe-src from the SAME export = the merge: encoder AND its co-adapted track slot
# lifted together (bias copied through); everything else comes from the pristine --base.
python data_pipeline/convert_trackwan_init_v2.py \
  --base "$BASE" \
  --out  "$OUT" \
  --id-dim "$ID_DIM" --pe-init random \
  --track-src "$SRC/transformer" \
  --pe-src    "$SRC/transformer"

echo "[merge] built merged init -> $OUT"
python -c "
import json; c=json.load(open('$OUT/transformer/config.json'))
print('[merge] in_channels', c['in_channels'], '| id_dim', c['track_config']['id_dim'])
"
echo "[merge] next: 05_run_openvid_stage1.sh (init_from defaults to this dir)"
