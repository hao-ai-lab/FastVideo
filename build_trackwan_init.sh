#!/usr/bin/env bash
set +e
export HF_HOME=/mnt/lustre/vlm-s4duan/.hf
export TOKENIZERS_PARALLELISM=false
cd /mnt/lustre/vlm-s4duan/FastVideo || exit 1
source .venv/bin/activate

BASE=/mnt/lustre/vlm-s4duan/models/Wan2.1-Fun-1.3B-InP-Diffusers
OUT=/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_init
mkdir -p /mnt/lustre/vlm-s4duan/models

echo "==================== download Fun-InP diffusers base ===================="
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers", local_dir="$BASE")
print("downloaded to", p)
PY
echo "DL_EXIT=$?"

echo "==================== base transformer sanity (I2V => in_channels 36) ===================="
python - <<PY
import json
c = json.load(open("$BASE/transformer/config.json"))
print("in_channels:", c.get("in_channels"), "image_dim:", c.get("image_dim"), "num_layers:", c.get("num_layers"), "hidden:", c.get("hidden_size") or c.get("dim"))
PY

echo "==================== convert_trackwan_init.py ===================="
python data_pipeline/convert_trackwan_init.py --base "$BASE" --out "$OUT"
echo "CONVERT_EXIT=$?"

echo "==================== trackwan init result ===================="
ls -la "$OUT" 2>&1 | head -25
python - <<PY
import json
c = json.load(open("$OUT/transformer/config.json"))
print("trackwan in_channels:", c.get("in_channels"), "| has track_config:", "track_config" in c)
PY
echo "==================== DONE build_trackwan_init ===================="
