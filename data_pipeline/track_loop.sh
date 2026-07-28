#!/usr/bin/env bash
# Overlap tracking with the slow self-healing download: idempotent multi-node passes
# over the growing videos/ dir until download done (98 parts) AND ~all tracked.
# MUST run inside srun on a compute node (login-node `sleep` gets killed by the harness).
# Waits for any in-flight tracking pass first, so it's safe to relaunch (no double-submit).
set +e
WORK=/mnt/lustre/vlm-s4duan
cd "$WORK/FastVideo"
DC(){ ls "$WORK"/openvid_1m/_extracted/*.done 2>/dev/null | wc -l; }
RUNNING(){ squeue -u vlm-s4duan -h -n cotracker_dp -o %i 2>/dev/null; }
NODES=${NODES:-8}
while true; do
  while [ -n "$(RUNNING)" ]; do sleep 90; done            # let in-flight pass finish
  ls "$WORK"/openvid_1m/videos/*.mp4 > "$WORK"/openvid_1m/videos_all.txt 2>/dev/null
  nv=$(wc -l < "$WORK"/openvid_1m/videos_all.txt 2>/dev/null); nv=${nv:-0}
  nt=$(ls "$WORK"/openvid_1m/tracks/*.npz 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] parts=$(DC)/98 videos=$nv tracks=$nt"
  if [ "$(DC)" -ge 98 ] && [ "$nt" -ge "$((nv - nv/40 - 200))" ]; then
    echo "TRACK_LOOP_DONE videos=$nv tracks=$nt"; break; fi
  if [ "$nv" -le "$((nt + 300))" ]; then echo "  little new; wait 300s"; sleep 300; continue; fi
  jid=$(CLIPS_DIR="$WORK"/openvid_1m/clips VIDEO_LIST="$WORK"/openvid_1m/videos_all.txt \
        OUT_DIR="$WORK"/openvid_1m/tracks NODES=$NODES PROCS_PER_GPU=2 FPS=24 NUM_FRAMES=121 GRID=50 \
        bash data_pipeline/run_tracks_slurm.sh 2>&1 | grep -oE "[0-9]+$" | tail -1)
  echo "  submitted track job $jid over $nv videos"
  [ -z "$jid" ] && sleep 180
done
