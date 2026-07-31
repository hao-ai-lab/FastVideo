#!/usr/bin/env bash
# 720p (1280x720) i2v_track PREPROCESS of OpenVid-1M, run INSIDE a held allocation.
#
# Differs from run_preprocess_track_slurm.sh (which sbatch's its own job array):
#   * runs via `srun --overlap --jobid=$JOBID` inside an existing held alloc, so it
#     shares the nodes with (and does not disturb) the training hold,
#   * ONE srun step PER NODE instead of one gang step, so the k8s operator cordoning
#     a node kills only that node's step; a supervisor pass relaunches it,
#   * shards are CLAIMED with an atomic `mkdir` under <combined>/.claims/, so any
#     worker can steal any not-yet-claimed shard -> a dead node's remaining work is
#     picked up by the survivors instead of being silently skipped.
#   * per-shard `.done` markers make every (re)start skip finished work.
#
# Usage:  JOBID=881 bash data_pipeline/run_preprocess_720p_held.sh
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
DATA_DIR=${DATA_DIR:-$WORK/openvid_1m}
MODEL=${MODEL:-$WORK/models/trackwan_1.3b_i2v_d64_nobias_init}   # same encoders as the 480p run
CLIPS_DIR=${CLIPS_DIR:-$DATA_DIR/clips}
MANIFEST=${MANIFEST:-$DATA_DIR/videos2caption.json}
COMBINED=${COMBINED:-$DATA_DIR/combined_parquet_dataset_720p}
SHARDS_DIR=${SHARDS_DIR:-$DATA_DIR/preprocess_shards}   # reuse the 480p partition
NUM_SHARDS=${NUM_SHARDS:-1500}
JOBID=${JOBID:-881}
GPUS=${GPUS:-4}
MAX_H=${MAX_H:-720}; MAX_W=${MAX_W:-1280}; NUM_FRAMES=${NUM_FRAMES:-121}
TRAIN_FPS=${TRAIN_FPS:-24}; NUM_LATENT_T=${NUM_LATENT_T:-31}
BATCH=${BATCH:-1}          # must be 1: pad-free T5 tokenizer can't batch variable-length captions
VAE_PREC=${VAE_PREC:-fp32} # match the 480p run
NW=${NW:-8}                # decode-prefetch workers
MAXPASS=${MAXPASS:-8}
LOGDIR=$WORK/logs/prep720
CLAIMS=$COMBINED/.claims
mkdir -p "$LOGDIR" "$COMBINED" "$CLAIMS"

# By default use every node in the held alloc; set NODELIST to run on a SUBSET
# (e.g. NODELIST=hpc-rack-2-[0-2,4] to leave the rest free for training).
NODELIST=${NODELIST:-$(squeue -h -j "$JOBID" -o %N)}
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
NNODES=${#NODES[@]}
WORKERS=$(( NNODES * GPUS ))
echo "[sup] jobid=$JOBID nodes=$NNODES (${NODES[*]}) workers=$WORKERS"
echo "[sup] geometry ${MAX_W}x${MAX_H} ${NUM_FRAMES}f -> latent [16,$NUM_LATENT_T,$((MAX_H/8)),$((MAX_W/8))]"
echo "[sup] out=$COMBINED shards=$SHARDS_DIR ($NUM_SHARDS)"

count_done() { find "$COMBINED" -maxdepth 2 -name .done 2>/dev/null | wc -l; }

for PASS in $(seq 1 "$MAXPASS"); do
  DONE=$(count_done)
  echo "[sup] pass $PASS: $DONE/$NUM_SHARDS shards done"
  [ "$DONE" -ge "$NUM_SHARDS" ] && { echo "[sup] all shards done"; break; }

  # No workers are alive at this point, so any claim without a .done is stale
  # (its worker died) -> release it so this pass can re-run that shard.
  RELEASED=0
  for C in "$CLAIMS"/shard_*; do
    [ -d "$C" ] || continue
    S=$(basename "$C")
    [ -f "$COMBINED/$S/.done" ] || { rmdir "$C" 2>/dev/null && RELEASED=$((RELEASED+1)); }
  done
  [ "$RELEASED" -gt 0 ] && echo "[sup] released $RELEASED stale claim(s)"

  PIDS=()
  for I in $(seq 0 $(( NNODES - 1 ))); do
    NODE=${NODES[$I]}
    srun --overlap --jobid="$JOBID" --nodelist="$NODE" --nodes=1 --ntasks="$GPUS" \
         --ntasks-per-node="$GPUS" --gres=gpu:"$GPUS" --cpus-per-task=$(( 128 / GPUS )) \
         --chdir="$WORK/FastVideo" \
         -o "$LOGDIR/pass${PASS}_${NODE}.out" -e "$LOGDIR/pass${PASS}_${NODE}.out" \
      bash -lc "
        set -uo pipefail
        source .venv/bin/activate
        export HOME=$WORK TRITON_CACHE_DIR=$WORK/.triton XDG_CACHE_HOME=$WORK/.cache \
          HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
          PYTHONPATH=$WORK/FastVideo TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0
        export CUDA_VISIBLE_DEVICES=\$(( SLURM_LOCALID % $GPUS ))
        export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
        export MASTER_PORT=\$(( 29500 + SLURM_LOCALID ))
        W=\$(( $I * $GPUS + SLURM_LOCALID ))
        echo \"[worker \$W] host=\$(hostname) gpu=\$CUDA_VISIBLE_DEVICES\"
        # Start spread out across the shard list, then walk the whole list claiming
        # whatever is free -> automatic work-stealing, no idle workers at the tail.
        OFF=\$(( W * $NUM_SHARDS / $WORKERS ))
        for K in \$(seq 0 $(( NUM_SHARDS - 1 ))); do
          IDX=\$(( (OFF + K) % $NUM_SHARDS ))
          SHARD=\$(printf shard_%05d \$IDX)
          ODIR=$COMBINED/\$SHARD
          [ -f \"\$ODIR/.done\" ] && continue
          mkdir \"$CLAIMS/\$SHARD\" 2>/dev/null || continue   # atomic claim; someone else has it
          rm -rf \"\$ODIR\"
          T0=\$(date +%s)
          if python fastvideo/pipelines/preprocess/v1_preprocess.py \
              --model_path $MODEL --preprocess_task i2v_track \
              --data_merge_path $SHARDS_DIR/\$SHARD/merge.txt --output_dir \"\$ODIR\" \
              --max_height $MAX_H --max_width $MAX_W --num_frames $NUM_FRAMES \
              --train_fps $TRAIN_FPS --num_latent_t $NUM_LATENT_T --vae_precision $VAE_PREC \
              --preprocess_video_batch_size $BATCH --dataloader_num_workers $NW \
              --video_length_tolerance_range 5 --seed 1000 > \"$LOGDIR/\$SHARD.log\" 2>&1; then
            touch \"\$ODIR/.done\"
            rm -f \"$LOGDIR/\$SHARD.log\"   # keep only failures' logs
            echo \"[worker \$W] \$SHARD done in \$(( \$(date +%s) - T0 ))s\"
          else
            echo \"[worker \$W] \$SHARD FAILED (\$(( \$(date +%s) - T0 ))s), log $LOGDIR/\$SHARD.log -> releasing claim\"
            tail -5 \"$LOGDIR/\$SHARD.log\" | sed \"s/^/[worker \$W]   /\"
            rmdir \"$CLAIMS/\$SHARD\" 2>/dev/null
          fi
        done
        echo \"[worker \$W] no shards left to claim\"" &
    PIDS+=($!)
  done
  echo "[sup] pass $PASS: launched ${#PIDS[@]} node steps; waiting"
  wait "${PIDS[@]}"
  echo "[sup] pass $PASS finished: $(count_done)/$NUM_SHARDS done"
done

echo "[sup] FINAL: $(count_done)/$NUM_SHARDS shards done"
