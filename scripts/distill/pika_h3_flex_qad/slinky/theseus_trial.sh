#!/bin/bash
# Bounded Theseus warm-up trial on tray 13510 once it holds a node (login node, tmux). Two arms, 30 min each, sequential.
R=/mnt/nfs/vlm-aryan/pika-flex-qad; T=13510; S=$R/logs/theseus-trial-summary.txt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $S; }
say "waiting for tray $T"
for i in $(seq 1 600); do [ "$(squeue -h -j $T -o %T)" = RUNNING ] && break; sleep 60; done
[ "$(squeue -h -j $T -o %T)" = RUNNING ] || { say "tray $T never started; giving up"; exit 0; }
say "tray $T running on $(squeue -h -j $T -o %N)"
COMMON="--models.student.qad.linears.amax_json null --models.student.qad.linears.provisional_calibration_rows 2 \
 --models.student.data.max_frames 124 --models.student.data.max_train_rows 8 --models.student.data.eval_resolutions [480p] \
 --models.student.data.eval_per_group 1 --training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4 \
 --training.loop.max_train_steps 8 --training.optimizer.learning_rate 2e-5 --training.optimizer.lr_warmup_steps 0 \
 --training.optimizer.lr_scheduler constant --method.eval_every 8 --method.full_eval_every 8 --method.patience 99 \
 --method.noise_floor_eval false --method.rollout_fraction 0 --training.tracker.trackers []"
arm() {  # arm <name> [extra overrides]
  local name=$1; shift
  timeout 1800 srun --overlap --jobid=$T --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
    --container-image=nvcr.io/nvidia/pytorch:25.06-py3 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre:/mnt/lustre:ro \
    bash $R/env.sh bash $R/jobs/smoke.sh $R/ref2va-qad.yaml $R/runs/trial-$name $COMMON "$@" > $R/logs/trial-$name.log 2>&1
  say "$name exit $?: $(grep -o "QAD eval @[0-9]*: score [0-9.]*" $R/logs/trial-$name.log | tr "\n" " ") $(grep -c Traceback $R/logs/trial-$name.log) tracebacks"
  grep -o "Theseus warm-up step [0-9]* (width [0-9]*)[^:]*: rel [0-9.]*" $R/logs/trial-$name.log | sed "s/^/  /" >> $S
  rm -rf $R/runs/trial-$name/checkpoint-* $R/runs/trial-$name/best
}
arm baseline
arm theseus --method.theseus_steps 4 --method.theseus_widths [1,4,16,50] --method.theseus_gamma 1.0
say "TRIAL_DONE"
