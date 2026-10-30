#!/bin/bash
# usage: calib.sh <collect|mse|candidates|choose> <config> <out_dir> [shard num_shards] [--dotted.key value ...]  (runs under env.sh)
set -uo pipefail
STAGE=$1; CFG=$2; OUT=$3; shift 3
S=/mnt/nfs/vlm-aryan/pika-flex-qad/code/scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py
mkdir -p $OUT
case $STAGE in
  collect|mse)
    SH=$1; NS=$2; shift 2
    FASTVIDEO_PG_TIMEOUT_MIN=120 $PY -m torch.distributed.run --nproc_per_node=${NPROC:-4} --master_port=$((29600 + SH)) \
      --log-dir $OUT/torchlogs-$STAGE-$SH --redirects 3 --tee 3 $S $STAGE --config $CFG --output-dir $OUT \
      --shard $SH --num-shards $NS --training.distributed.num_gpus ${NPROC:-4} --training.distributed.hsdp_shard_dim ${NPROC:-4} --training.distributed.sp_size ${NPROC:-4} "$@" ;;
  *) $PY $S $STAGE --output-dir $OUT "$@" ;;
esac
echo "CALIB_STAGE_DONE $STAGE"
