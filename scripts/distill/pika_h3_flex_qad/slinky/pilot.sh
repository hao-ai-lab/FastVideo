#!/bin/bash
# One-tray QAD pilot (SP 4): pilot.sh <tray> <name> <amax table> [--dotted.key value ...]; logs/pilot-<name>.log, runs/pilot-<name>
R=/mnt/nfs/vlm-aryan/pika-flex-qad; T=$1; N=$2; A=$3; shift 3
nohup srun --overlap --jobid=$T --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre:/mnt/lustre:ro \
  bash $R/env.sh bash $R/jobs/smoke.sh $R/ref2va-qad.yaml $R/runs/pilot-$N \
  --models.student.qad.linears.amax_json $A --training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4 \
  --training.loop.max_train_steps 60 --method.eval_every 15 --method.full_eval_every 30 --method.patience 99 \
  --method.noise_floor_eval false --method.max_rollout_rows 4 \
  --models.student.data.eval_per_group.480p 1 --models.student.data.eval_per_group.768p 1 \
  --training.tracker.run_name pilot-$N --training.checkpoint.training_state_checkpointing_steps 1000 "$@" \
  > $R/logs/pilot-$N.log 2>&1 &
echo "pilot $N on tray $T (pid $!) -> $R/logs/pilot-$N.log"
