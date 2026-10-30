#!/bin/bash
# Overnight Flex Ref2VA calibration on trays 13509 + 13483 (login node, in tmux): finalize coverage rows,
# collect, candidates, MSE, choose, then step-0 PTQ evals of each table. Log: logs/calib-pipeline.log
R=/mnt/nfs/vlm-aryan/pika-flex-qad; C=$R/calib/ref2va; CFG=$R/ref2va-qad.yaml; L=$R/logs
T0=13509; T1=13483
run() {  # run <tray> <log> <cmd...>: one container step under env.sh, foreground
  local t=$1 log=$2; shift 2
  srun --overlap --jobid=$t --nodes=1 --ntasks=1 --kill-on-bad-exit=1 --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
    --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre:/mnt/lustre:ro bash $R/env.sh "$@" > $log 2>&1
}
say() { echo "$(date -u +%FT%TZ) $*"; }
say "start: coverage rows final (retry stopped)"
run $T0 $L/build-finalize.log bash $R/jobs/build_rows.sh finalize; tail -3 $L/build-finalize.log
A="--models.student.data.per_group.480p 120 --models.student.data.per_group.768p 30"
say "collect"
run $T0 $L/calib-collect-0.log bash $R/jobs/calib.sh collect $CFG $C 0 2 $A & run $T1 $L/calib-collect-1.log bash $R/jobs/calib.sh collect $CFG $C 1 2 $A & wait
grep -h "CALIB_STAGE_DONE\|Traceback" $L/calib-collect-*.log; wc -l $C/collect-*.jsonl
run $T0 $L/calib-candidates.log bash $R/jobs/calib.sh candidates - $C --outlier-rows 8; tail -2 $L/calib-candidates.log
say "mse"
run $T0 $L/calib-mse-0.log bash $R/jobs/calib.sh mse $CFG $C 0 2 --mse-rows-per-group 3 & run $T1 $L/calib-mse-1.log bash $R/jobs/calib.sh mse $CFG $C 1 2 --mse-rows-per-group 3 & wait
run $T0 $L/calib-choose.log bash $R/jobs/calib.sh choose - $C; tail -3 $L/calib-choose.log
say "step-0 PTQ evals"
ptq() {  # ptq <tray> <name> <table>
  run $1 $L/ptq-$2.log bash $R/jobs/smoke.sh $CFG $R/runs/ptq-$2 --models.student.qad.linears.amax_json $3 \
    --training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4 --training.loop.max_train_steps 1 \
    --training.optimizer.learning_rate 0.0 --method.noise_floor_eval false --training.tracker.trackers "[]"
  grep -o "QAD eval @0: score [0-9.]* (PTQ [0-9.]*)" $L/ptq-$2.log | head -1 | sed "s/^/$2: /"
}
ptq $T0 max $C/amax-max.json & ptq $T1 mse $C/amax-mse.json & wait
ptq $T0 balanced $C/amax-mse-balanced.json
say "CALIB_PIPELINE_DONE"
