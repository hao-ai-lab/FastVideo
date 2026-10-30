#!/bin/bash
# 1-node QAD smoke: usage smoke.sh <config> <out>   (runs under env.sh)
CFG=$1; OUT=$2; shift 2; rm -rf $OUT; mkdir -p $OUT
FASTVIDEO_PG_TIMEOUT_MIN=120 $PY -m torch.distributed.run --nproc_per_node=4 --master_port=29571 --log-dir $OUT/torchlogs --redirects 3 --tee 3 -m fastvideo.train.entrypoint.train --config $CFG --training.checkpoint.output_dir $OUT --method.eval_save_dir $OUT/samples --method.step0_cache $OUT/step0.json "$@"
