#!/usr/bin/env bash
# Multi-node data-parallel i2v_track PREPROCESS over OpenVid-1M.
# v1_preprocess.py asserts WORLD_SIZE==1 (one GPU per process), so scale-out is
# N nodes x 4 GPU x 1 proc/GPU = 4N independent single-GPU processes, each over a
# SHARD of the clips, writing parquet into <combined>/shard_<idx>/. The trainer
# reads <combined> as ONE dataset (get_parquet_files_and_length os.walks recursively).
#
# Usage:
#   NODES=16 bash data_pipeline/run_preprocess_track_slurm.sh
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
DATA_DIR=${DATA_DIR:-$WORK/openvid_1m}
MODEL=${MODEL:-$WORK/models/trackwan_1.3b_i2v_d64_nobias_init}
CLIPS_DIR=${CLIPS_DIR:-$DATA_DIR/clips}
MANIFEST=${MANIFEST:-$DATA_DIR/videos2caption.json}
COMBINED=${COMBINED:-$DATA_DIR/combined_parquet_dataset}   # trainer points here
SHARDS_DIR=${SHARDS_DIR:-$DATA_DIR/preprocess_shards}
NODES=${NODES:-14}; GPUS=${GPUS:-4}
PARTITION=${PARTITION:-all}
CONC=${CONC:-$(( NODES * GPUS ))}          # max concurrent shards (= total GPUs)
# MANY small shards (not one-per-GPU) so a node glitch only loses its in-flight
# ~CONC/NODES shards, and .done markers make a resubmit skip finished work.
NUM_SHARDS=${NUM_SHARDS:-1024}             # ~253 clips/shard (fine-grained .done resume)
# Training-target geometry (must match trainer):
MAX_H=${MAX_H:-480}; MAX_W=${MAX_W:-832}; NUM_FRAMES=${NUM_FRAMES:-121}
TRAIN_FPS=${TRAIN_FPS:-24}; NUM_LATENT_T=${NUM_LATENT_T:-31}
# batch_size MUST be 1: the T5 tokenizer_kwargs pad-free config (configs/models/
# encoders/t5.py) can't batch variable-length captions with return_tensors="pt".
BATCH=${BATCH:-1}
VAE_PREC=${VAE_PREC:-fp32}   # bf16 gives NO speedup here (decode-bound, not VAE-compute-bound)
NW=${NW:-8}                  # decode-prefetch workers. SAFE now: patched VideoCaptionMergedDataset.__iter__
                             # to shard by get_worker_info() (else num_workers>0 duplicates). Validated:
                             # 16 clips -> 16 rows, latents byte-identical to nw=0. Overlaps decode w/ GPU encode.
TIME=${TIME:-24:00:00}
CPUS_PER_TASK=$(( 128 / GPUS )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
mkdir -p "$WORK/logs" "$COMBINED"

# 1) Build/refresh shard manifests (idempotent; cheap).
source "$WORK/FastVideo/.venv/bin/activate"
python "$WORK/FastVideo/data_pipeline/split_manifest_shards.py" \
  --manifest "$MANIFEST" --clips-dir "$CLIPS_DIR" \
  --out-dir "$SHARDS_DIR" --num-shards "$NUM_SHARDS"

echo "array of $NUM_SHARDS shards, up to $CONC concurrent (~$((NODES)) nodes x $GPUS GPU); out=$COMBINED"

# 2) Launch. This cluster gives a GPU job the WHOLE node (all 4 GPU + 144 CPU), so we
#    use -N nodes x ntasks-per-node=4 (= 4N=56 workers) rather than a job array (which
#    would get 1 whole node per task = only 14 concurrent). Each worker loops over its
#    slice shards[PROCID::NW], running a FRESH python per SMALL (~260-clip) shard:
#    the fresh process bounds the pipeline's ~0.75 GB/clip RSS growth (VAE feature cache)
#    -> 4 workers x ~214 GB peak = ~856 GB < 979 GB/node. Per-shard .done makes resubmit
#    skip finished work; --no-requeue avoids a destructive auto-restart on node glitches.
WORKERS=$(( NODES * GPUS ))
# CORDON-PROOF: the k8s operator cordons worker pods mid-run (node-pool scale-down);
# a gang -N$NODES job dies entirely if ANY one node is cordoned. So submit a JOB ARRAY
# of single-node tasks (--array=0..NODES-1, each -N1): a cordon kills+requeues only ONE
# array task; the other nodes keep running. Global worker idx = ARRAY_TASK_ID*GPUS+LOCALID
# over WORKERS=NODES*GPUS; per-shard .done makes every (re)start skip finished work.
WORKERS=$(( NODES * GPUS ))
sbatch --array=0-$(( NODES - 1 )) -N1 --gres=gpu:$GPUS --ntasks-per-node=$GPUS --exclusive --mem=0 \
  -t "$TIME" --requeue -p "$PARTITION" -J preprocess_track \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/preprocess_track_%A_%a.out" -e "$WORK/logs/preprocess_track_%A_%a.out" \
  --wrap "srun --chdir=$WORK/FastVideo bash -lc '
    set -uo pipefail
    source .venv/bin/activate
    export HOME=$WORK TRITON_CACHE_DIR=$WORK/.triton XDG_CACHE_HOME=$WORK/.cache \
      HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=\$(( SLURM_LOCALID % $GPUS ))
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
    export MASTER_PORT=\$(( 29500 + SLURM_LOCALID ))
    W=\$(( SLURM_ARRAY_TASK_ID * $GPUS + SLURM_LOCALID )); NW=$WORKERS
    echo \"[worker \$W/\$NW arr=\$SLURM_ARRAY_TASK_ID] host=\$(hostname) CVD=\$CUDA_VISIBLE_DEVICES\"
    for IDX in \$(seq \$W \$NW $(( NUM_SHARDS - 1 ))); do
      SHARD=\$(printf shard_%05d \$IDX); SDIR=$SHARDS_DIR/\$SHARD; ODIR=$COMBINED/\$SHARD
      [ -f \"\$ODIR/.done\" ] && continue
      rm -rf \"\$ODIR\"
      if python fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL --preprocess_task i2v_track \
        --data_merge_path \$SDIR/merge.txt --output_dir \$ODIR \
        --max_height $MAX_H --max_width $MAX_W --num_frames $NUM_FRAMES \
        --train_fps $TRAIN_FPS --num_latent_t $NUM_LATENT_T --vae_precision $VAE_PREC \
        --preprocess_video_batch_size $BATCH --dataloader_num_workers $NW; then
        touch \"\$ODIR/.done\"; echo \"[worker \$W] \$SHARD done\"
      else
        echo \"[worker \$W] \$SHARD FAILED (rc=\$?), skipping\"
      fi
    done
    echo \"[worker \$W] all shards done\"'"
