#!/usr/bin/env bash
# Self-healing OpenVidHD download: HF unauthenticated throttles ~per-200GB then
# resets after a cooldown. Run N idempotent download shards; watchdog detects a
# stall (no zip growth) and kills+cooldowns+relaunches (resume) until all 98 parts
# done. Runs as ONE srun on the node so local kill works. Honors $HF_TOKEN if set.
set +e
WORK=/mnt/lustre/vlm-s4duan
export HF_HOME=$WORK/.hf
cd "$WORK/FastVideo"; source .venv/bin/activate
NSHARD=${NSHARD:-4}
DC(){ ls "$WORK"/openvid_1m/_extracted/*.done 2>/dev/null | wc -l; }
ZB(){ find "$WORK"/openvid_1m/_zips -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{print s+0}'; }

pkill -f openvid_download_hd.py 2>/dev/null; sleep 3
round=0
while [ "$(DC)" -lt 98 ]; do
  round=$((round+1))
  echo "[$(date +%H:%M)] ROUND $round start parts=$(DC)/98 ${HF_TOKEN:+(token set)}"
  pids=()
  for s in $(seq 0 $((NSHARD-1))); do
    python data_pipeline/openvid_download_hd.py --shard "$s" --num-shards "$NSHARD" \
      --videos-dir "$WORK"/openvid_1m/videos --zip-dir "$WORK"/openvid_1m/_zips/"$s" \
      --only-list "$WORK"/openvid/OpenVidHD_filtered.txt &
    pids+=($!)
  done
  last=$(ZB); stall=0
  while :; do
    sleep 120
    [ "$(DC)" -ge 98 ] && break
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && { echo "[$(date +%H:%M)] round done naturally parts=$(DC)/98"; break; }
    now=$(ZB)
    if [ "$now" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
    last=$now
    echo "[$(date +%H:%M)] parts=$(DC)/98 zip=$((now/1000000000))GB stall=$stall"
    if [ "$stall" -ge 2 ]; then
      echo "[$(date +%H:%M)] STALL -> kill + cooldown"
      kill "${pids[@]}" 2>/dev/null; sleep 5; kill -9 "${pids[@]}" 2>/dev/null
      pkill -9 -f openvid_download_hd.py 2>/dev/null
      break
    fi
  done
  wait 2>/dev/null
  [ "$(DC)" -ge 98 ] && break
  echo "[$(date +%H:%M)] cooldown 180s (parts $(DC)/98)"; sleep 180
done
echo "ALL_PARTS_DONE $(DC)/98 videos=$(ls "$WORK"/openvid_1m/videos/*.mp4 2>/dev/null | wc -l)"
