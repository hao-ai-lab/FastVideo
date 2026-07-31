#!/bin/bash
# Phase 0, step 1 — build the d64 + bias 14B WanTrack init for the overfit.
# CPU only, needs ~62GB RAM (loads the 14B base) -> run on a COMPUTE node, not the login node.
# pretrained channels are preserved; the added track slot is ZERO-init (--pe-init zero, matching
# upstream's trackwan_14b_i2v_d64_zero_init_bias) and the track_encoder gets default init WITH bias
# (--use-bias-defaults), matching TRACKWAN_TRACK_BIAS=1 at train time.
set -uo pipefail
cd ~/FastVideo

BASE=${BASE:-/home/hal-kevin/models/Wan2.1-I2V-14B-720P-Diffusers}
OUT=${OUT:-/home/hal-kevin/models/trackwan_14b_i2v_d64_bias_init}
ID_DIM=${ID_DIM:-64}
PE_INIT=${PE_INIT:-zero}

python data_pipeline/convert_trackwan_init_v2.py \
  --base "$BASE" \
  --out  "$OUT" \
  --id-dim "$ID_DIM" \
  --pe-init "$PE_INIT" \
  --use-bias-defaults

echo "[init] built $OUT (id_dim=$ID_DIM, pe-init=$PE_INIT, bias=on)"
echo "[init] expect: 'added 4 track_encoder tensors' (2 weights + 2 bias)"
python -c "
from safetensors import safe_open
ks=[k for k in safe_open('$OUT/transformer/diffusion_pytorch_model.safetensors','pt').keys() if 'track_encoder' in k]
import json; c=json.load(open('$OUT/transformer/config.json'))
print('[init] in_channels', c['in_channels'], '| track_config.id_dim', c['track_config']['id_dim'])
print('[init] track_encoder keys:', sorted(ks))
"
