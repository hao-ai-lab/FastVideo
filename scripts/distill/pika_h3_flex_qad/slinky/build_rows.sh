#!/bin/bash
# usage: build_rows.sh plan | encode <num_shards> <shard_base> | finalize   (runs under env.sh; encode = one process per GPU)
O=/mnt/nfs/vlm-aryan/pika-flex-qad/data/flex-ref2va; V4=/mnt/lustre/vlm-wlsaidhi/fastvideo/data/omniref-v4-processed/20260919T001000Z
EVAL=/mnt/nfs/vlm-aryan/h3-nvfp4-decoder/gen-latents/omniref-eval/manifest.json
case $1 in
  plan) $PY /mnt/nfs/vlm-aryan/pika-flex-qad/code/scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py plan --output-dir $O --v4-manifest $V4/encoded/full/MANIFEST.jsonl --plan-config /mnt/nfs/vlm-aryan/pika-flex-qad/ref2va-qad.yaml --models.student.data.max_frames 345 --models.student.data.per_group.480p 160 --models.student.data.per_group.768p 40 --eval-manifest $EVAL --extra-per-group 480p=80 768p=20 ;;
  encode) for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g MASTER_PORT=$((29700 + g)) $PY /mnt/nfs/vlm-aryan/pika-flex-qad/code/scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py encode --output-dir $O --model-path /mnt/nfs/vlm-aryan/pika-flex-qad/models/pika-flex-ref2va-merged --shard $(($3 + g)) --num-shards $2 ${@:4} >> $O/encode-$(($3 + g)).log 2>&1 & done; wait ;;
  finalize) $PY /mnt/nfs/vlm-aryan/pika-flex-qad/code/scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py finalize --output-dir $O --eval-manifest $EVAL ;;
esac
echo BUILD_DONE $1
