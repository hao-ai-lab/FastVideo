#!/usr/bin/env bash
# STEP C — build the 14B "merged" init from the step-B overfit.
#
# Takes the 14B-native overfit (steps A+B) and transplants BOTH halves of the track pathway
# into a FRESH 14B base:
#   * track_encoder.{proj,temporal_conv}.{weight,bias}   (--track-src)
#   * patch_embedding.weight[:, 36:]  — the track slot   (--pe-src)
# while patch_embedding.weight[:, :36] stays the pretrained I2V weights from --base.
#
# Both halves come from the SAME source on purpose: they were co-adapted during the overfit.
# Lifting the encoder alone (the "partial_merged" experiment) measured WORSE than random init,
# because the trained encoder ends up feeding a random projection.
#
# Usage: [STEP=2000] bash examples/train/run_stepC_merge.sh
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
REPO=$WORK/FastVideo
STEP="${STEP:-2000}"
SRC_CKPT="${SRC_CKPT:-$WORK/wantrack_14b_synth_sparse_random_out/checkpoint-$STEP}"
EXPORT_DIR="${EXPORT_DIR:-$WORK/exports/overfit_14b_random_ckpt$STEP}"
OUT="${OUT:-$WORK/models/trackwan_14b_i2v_d64_merged_from_overfit_bias}"
BASE="${BASE:-$WORK/models/Wan2.1-I2V-14B-720P-Diffusers}"

cd "$REPO"
source .venv/bin/activate
export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
       PYTHONPATH=$REPO TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 \
       TRACKWAN_TRACK_BIAS=1

[ -d "$SRC_CKPT" ] || { echo "[C] missing $SRC_CKPT"; exit 1; }
[ -f "$SRC_CKPT/dcp/.metadata" ] || { echo "[C] $SRC_CKPT is an INCOMPLETE checkpoint (no dcp/.metadata)"; exit 1; }

# 1) DCP -> diffusers (single process; DCP reshards automatically)
if [ ! -d "$EXPORT_DIR/transformer" ]; then
  echo "[C] exporting $SRC_CKPT -> $EXPORT_DIR"
  python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint "$SRC_CKPT" --output-dir "$EXPORT_DIR"
else
  echo "[C] reusing existing export $EXPORT_DIR"
fi

# 2) fresh 14B base + co-adapted encoder AND track slot from that export
echo "[C] building merged init -> $OUT"
python data_pipeline/convert_trackwan_init_v2.py \
  --base "$BASE" --out "$OUT" --id-dim 64 --pe-init random \
  --track-src "$EXPORT_DIR" --pe-src "$EXPORT_DIR"

# 3) verify: pretrained channels untouched, track slot + encoder match the overfit
echo "[C] verifying merged init against source ..."
python - "$OUT" "$EXPORT_DIR" "$BASE" <<'PY'
import sys, glob
import torch
from safetensors.torch import load_file

def load(d):
    st = {}
    for f in sorted(glob.glob(f"{d}/transformer/*.safetensors")):
        st.update(load_file(f))
    return st

out, src, base = (load(p) for p in sys.argv[1:4])
pe_o, pe_s, pe_b = out["patch_embedding.weight"], src["patch_embedding.weight"], base["patch_embedding.weight"]
print(f"  pe shape {tuple(pe_o.shape)}  (base {tuple(pe_b.shape)})")
print(f"  pe[:, :36] == base           : {torch.equal(pe_o[:, :36], pe_b)}   std={pe_o[:, :36].float().std():.5f}")
print(f"  pe[:, 36:] == overfit slot   : {torch.equal(pe_o[:, 36:], pe_s[:, 36:])}   std={pe_o[:, 36:].float().std():.5f}")
ok = torch.equal(pe_o[:, :36], pe_b) and torch.equal(pe_o[:, 36:], pe_s[:, 36:])
for k in sorted(k for k in out if "track_encoder" in k):
    same = k in src and torch.equal(out[k], src[k])
    ok &= same
    print(f"  {k:42s} lifted={same}  norm={out[k].float().norm():.5f}")
print("  RESULT:", "OK — encoder and track slot are co-adapted" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PY
echo "[C] done -> $OUT"
