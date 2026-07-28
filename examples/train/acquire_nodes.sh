#!/usr/bin/env bash
# Acquire a single gang allocation of N nodes on the Slinky (k8s-backed) pool.
#
# Why this is not just `sbatch -N<N>`: the k8s autoscaler destroys idle nodes, so the pool only
# contains as many nodes as there is current demand. Slurm REJECTS (does not queue) a job asking
# for more nodes than physically exist in the partition:
#     "Batch job submission failed: Requested node configuration is not available"
# So we have to manufacture demand first, then grab the gang allocation:
#   1. keep N+SLACK short 1-node warmups queued -> autoscaler grows the pool
#   2. probe with `sbatch --test-only -N<N>` until Slurm says it *could* schedule it
#   3. submit the real -N<N> hold (now accepted; pends behind the warmups)
#   4. drop the warmups so the hold starts immediately
#
# Note: other people's jobs (e.g. the demo hold) occupy nodes too, so the pool has to grow to
# roughly N + (nodes held by others) before a -N<N> request becomes satisfiable.
#
# Usage: NODES=12 JOB=wan14b_hold bash examples/train/acquire_nodes.sh
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
NODES="${NODES:-12}"
JOB="${JOB:-wan14b_hold}"
TIME="${TIME:-120:00:00}"
SLACK="${SLACK:-3}"          # extra warmups beyond NODES, to out-pace other users' holds
WARM_SECS="${WARM_SECS:-900}"
MAX_ROUNDS="${MAX_ROUNDS:-60}"
mkdir -p "$WORK/logs"
say() { echo "[acquire $(date +%H:%M:%S)] $*"; }

EXIST=$(squeue -h -u "$USER" -n "${JOB}" -o '%i' 2>/dev/null | head -1)
if [ -n "$EXIST" ]; then say "reusing existing hold $EXIST"; echo "ALLOC=$EXIST"; exit 0; fi

WANT_WARM=$(( NODES + SLACK ))

for round in $(seq 1 "$MAX_ROUNDS"); do
  pool=$(sinfo -h -p all -N -o '%N' 2>/dev/null | sort -u | wc -l)

  # Probe: can Slurm schedule -N$NODES at all? (--test-only never actually submits)
  probe=$(sbatch --test-only -N"$NODES" --gres=gpu:4 --ntasks-per-node=1 --exclusive \
            -t "$TIME" -p all --wrap='hostname' 2>&1 | head -1)

  if echo "$probe" | grep -q "to start at"; then
    say "pool=$pool — probe OK, submitting real -N$NODES hold"
    OUT=$(sbatch -N"$NODES" --gres=gpu:4 --ntasks-per-node=1 --exclusive -t "$TIME" \
      -p all -J "$JOB" --requeue --chdir="$WORK" -o "$WORK/logs/${JOB}_%j.out" \
      --wrap='srun sleep infinity' 2>&1)
    if echo "$OUT" | grep -q "Submitted batch"; then
      ALLOC=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
      say "SUBMITTED hold jobid=$ALLOC — clearing warmups so it can start"
      scancel -u "$USER" -n warmup 2>/dev/null
      for i in $(seq 1 180); do
        st=$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)
        if [ "$st" = R ]; then
          say "RUNNING on: $(squeue -h -j "$ALLOC" -o '%N')"
          echo "ALLOC=$ALLOC"; exit 0
        fi
        [ -z "$st" ] && { say "hold $ALLOC vanished"; break; }
        sleep 10
      done
      say "hold $ALLOC queued but not started yet (jobid kept)"; echo "ALLOC=$ALLOC"; exit 0
    fi
    say "real submit rejected despite OK probe: $(echo "$OUT" | tail -1)"
  else
    say "pool=$pool — probe says not schedulable yet"
  fi

  # Keep warmup pressure up so the autoscaler keeps growing the pool.
  have=$(squeue -h -u "$USER" -n warmup -o '%i' 2>/dev/null | wc -l)
  need=$(( WANT_WARM - have ))
  if [ "$need" -gt 0 ]; then
    say "submitting $need warmup(s) (have $have, want $WANT_WARM) to drive scale-up"
    for _ in $(seq 1 "$need"); do
      sbatch -N1 --gres=gpu:4 --exclusive -p all -t 00:20:00 -J warmup \
        -o /dev/null --chdir="$WORK" --wrap="srun sleep $WARM_SECS" >/dev/null 2>&1
    done
  fi
  sleep 30
done

say "FAILED to acquire $NODES nodes after $MAX_ROUNDS rounds"
exit 1
