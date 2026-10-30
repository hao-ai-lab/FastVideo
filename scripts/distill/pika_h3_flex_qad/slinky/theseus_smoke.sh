#!/bin/bash
# GPU smoke of the Theseus warm-up: 7 warm-up steps (one per width 1..50) + 2 normal steps, provisional scales.
R=/mnt/nfs/vlm-aryan/pika-flex-qad; T=$1
nohup srun --overlap --jobid=$T --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre:/mnt/lustre:ro \
  bash $R/env.sh bash $R/jobs/smoke.sh $R/smoke-ref2va.yaml $R/runs/theseus-smoke \
  --method.theseus_steps 7 --training.loop.max_train_steps 9 --method.eval_every 9 --method.full_eval_every 9 \
  --models.student.data.max_train_rows 9 > $R/logs/theseus-smoke.log 2>&1 &
echo "theseus smoke on tray $T (pid $!)"
