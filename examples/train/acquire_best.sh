#!/usr/bin/env bash
# Acquire the LARGEST held allocation we can, targeting 16 nodes but falling back 16->12->8.
# The cluster's topology/block quirk refuses -N<big> even when many nodes are "idle" (freshly
# k8s-scaled nodes register outside the static block defs -> unallocatable). Only nodes held by a
# RUNNING job are reliably allocatable. So: warm ~20 nodes with 1-node warmups, wait until they're
# RUNNING, then scancel them and immediately race a real -N request, trying the biggest size first.
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
JOB=wan14b_bidir_hold
TIME=120:00:00
say(){ echo "[best $(date +%H:%M:%S)] $*"; }

# 1) warm ~20 nodes
say "submitting 20 warmups to warm the pool"
for _ in $(seq 1 20); do
  sbatch -N1 --gres=gpu:4 --exclusive -p all -t 00:20:00 -J warmup -o /dev/null \
    --chdir="$WORK" --wrap='srun sleep 900' >/dev/null 2>&1
done
# 2) wait until >=16 warmups are RUNNING
for i in $(seq 1 40); do
  r=$(squeue -h -u "$USER" -n warmup -t R -o '%i' 2>/dev/null | wc -l)
  say "warmups running: $r"
  [ "$r" -ge 16 ] && break
  sleep 15
done
# 3) cancel warmups and race the biggest -N that the scheduler accepts
say "cancelling warmups and racing a real allocation"
scancel -u "$USER" -n warmup 2>/dev/null
got=""
for i in $(seq 1 90); do
  idle=$(sinfo -h -p all -N -o '%T' 2>/dev/null | grep -c idle)
  for N in 16 12 8; do
    [ "$idle" -ge "$N" ] || continue
    out=$(sbatch -N"$N" --gres=gpu:4 --ntasks-per-node=1 --exclusive -t "$TIME" -p all \
      -J "$JOB" --requeue --chdir="$WORK" -o "$WORK/logs/${JOB}_%j.out" \
      --wrap='srun sleep infinity' 2>&1 | head -1)
    if echo "$out" | grep -q "Submitted"; then
      jid=$(echo "$out" | grep -oE '[0-9]+' | head -1)
      say "GOT N=$N alloc jid=$jid"; got="$jid:$N"; break 2
    fi
  done
  say "iter $i: idle=$idle, no size accepted yet"
  sleep 4
done
[ -n "$got" ] || { say "FAILED to acquire any of 16/12/8"; exit 1; }
jid="${got%%:*}"; N="${got##*:}"
# 4) wait for it to start
for i in $(seq 1 120); do
  [ "$(squeue -h -j "$jid" -o '%t' 2>/dev/null)" = R ] && { say "RUNNING jid=$jid N=$N nodes=$(squeue -h -j "$jid" -o '%N')"; echo "ALLOC=$jid NODES=$N"; exit 0; }
  sleep 5
done
say "jid=$jid submitted but not running yet"; echo "ALLOC=$jid NODES=$N"; exit 0
