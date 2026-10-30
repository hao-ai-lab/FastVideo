#!/bin/bash
# usage: launch_o.sh <tray jobid> <timeout seconds> <log> <cmd...>  (Pika Flex env; detached on the login node)
J=$1; T=$2; LOG=$3; shift 3
nohup timeout $((T+120)) srun --overlap --jobid=$J --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre:/mnt/lustre:ro \
  timeout $T bash /mnt/nfs/vlm-aryan/pika-flex-qad/env.sh "$@" > $LOG 2>&1 &
echo "launched pid $! on tray $J -> $LOG"
