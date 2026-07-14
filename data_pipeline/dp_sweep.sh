#!/usr/bin/env bash
# DP concurrency sweep on ONE GPU: K workers sharing GPU 3, aggregate throughput.
set +e
WORK=/mnt/lustre/vlm-s4duan
source "$WORK/FastVideo/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_HOME=$WORK/.torch
export TOKENIZERS_PARALLELISM=false
V=$WORK/dp_test/videos; T=$WORK/dp_test/tracks; L=$WORK/dp_test/videos.txt
mkdir -p "$V" "$T"
if [ ! -f "$V/clip_24.mp4" ]; then
  for i in $(seq -w 1 24); do cp -f "$WORK/wan_overfit/data/videos/vid_000000.mp4" "$V/clip_$i.mp4"; done
fi
ls "$V"/*.mp4 > "$L"
echo "test set: $(wc -l < "$L") videos on GPU 3 (121f 720p, grid 50)"
for K in 1 2 3 4; do
  rm -f "$T"/*.npz
  echo "===================== K=$K procs/GPU ====================="
  pids=()
  for s in $(seq 0 $((K-1))); do
    CUDA_VISIBLE_DEVICES=3 python data_pipeline/extract_tracks_mp.py \
      --video-list "$L" --out-dir "$T" --gpus-per-node 1 \
      --shard "$s" --num-shards "$K" --fps 24 --num-frames 121 --grid-size 50 &
    pids+=($!)
  done
  wait "${pids[@]}"
done
echo "SWEEP_DONE"
