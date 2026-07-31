#!/usr/bin/env bash
# Wait for the in-flight wave to flush its .done markers, then restart the 720p
# preprocess on a 12-node subset of alloc 881 (frees hpc-rack-3-[2,5,10,15]).
#
# Safe because the run is resumable: per-shard .done markers are skipped on
# restart, and the launcher releases any claim lacking a .done at pass start.
set -uo pipefail
W=/mnt/lustre/vlm-s4duan
OUT=$W/openvid_1m/combined_parquet_dataset_720p
KEEP12="hpc-rack-2-[0-2,4-6,8-9,12,14],hpc-rack-3-[0-1]"
TARGET=${TARGET:-60}   # wave 1 is 64 shards; don't wait forever on a straggler

echo "[switch] waiting for >=$TARGET shards to flush before restarting on 12 nodes"
for _ in $(seq 1 90); do
  D=$(find "$OUT" -maxdepth 2 -name .done 2>/dev/null | wc -l)
  echo "[switch] $(date +%T) banked $D shards"
  [ "$D" -ge "$TARGET" ] && break
  sleep 60
done
BANKED=$(find "$OUT" -maxdepth 2 -name .done 2>/dev/null | wc -l)
echo "[switch] proceeding with $BANKED shards banked"

# 1) Stop the supervisor FIRST so it cannot launch another pass.
#    Character-class in the pattern so this script's own cmdline never matches.
pkill -f 'run_preprocess_720p_hel[d]' 2>/dev/null
sleep 3
# 2) Stop the 16 per-node srun steps.
pkill -f 'jobid=881 --nodelis[t]=hpc-rack' 2>/dev/null
sleep 15
pkill -9 -f 'jobid=881 --nodelis[t]=hpc-rack' 2>/dev/null
sleep 5
echo "[switch] supervisor procs left: $(pgrep -cf 'run_preprocess_720p_hel[d]' || echo 0)"
echo "[switch] srun step procs left:  $(pgrep -f 'jobid=881 --nodelis[t]=hpc-rack' | wc -l)"

# 3) Confirm no worker python survives anywhere in the alloc.
srun --overlap --jobid=881 --nodes=16 --ntasks=16 --ntasks-per-node=1 --chdir=$W \
  bash -lc 'N=$(ps -eo args --no-headers | grep -c "[p]ython fastvideo/pipelines"); [ "$N" -gt 0 ] && echo "  STILL RUNNING $(hostname): $N procs"' 2>/dev/null | grep -v '^srun\|error:'
echo "[switch] alloc drained"

# 4) Relaunch on 12 nodes. Resume skips banked shards; stale claims are released.
cd $W/FastVideo
JOBID=881 NODELIST="$KEEP12" nohup bash data_pipeline/run_preprocess_720p_held.sh \
  > $W/logs/prep720_supervisor_12n.log 2>&1 &
echo "[switch] relaunched on 12 nodes (pid $!), log prep720_supervisor_12n.log"
sleep 40
head -6 $W/logs/prep720_supervisor_12n.log
echo "[switch] FREE FOR TRAINING: hpc-rack-3-[2,5,10,15]"
