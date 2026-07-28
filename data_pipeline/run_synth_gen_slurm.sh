#!/usr/bin/env bash
# Multi-node data-parallel Wan2.2-T2V-A14B synthetic video generation.
#
# One VideoGenerator per GPU (num_gpus=1, NO SP/TP -- data-parallel is faster at scale).
# Node layout: NODES x 4 GPU x 1 proc/GPU = 4*NODES independent single-GPU workers.
# Worker W in [0, 4*NODES) owns prompts[W :: 4*NODES] (stride slice of a fixed shuffle).
#
# CORDON-PROOF: k8s cordons worker pods mid-run; a gang -N$NODES job dies if ANY node is
# cordoned. So we submit a JOB ARRAY of single-node tasks (--array=0..NODES-1, each -N1):
# a cordon kills+requeues only ONE array task; the others keep running. Per-video mp4
# existence makes every (re)start skip finished work -> fully resumable.
#
# Usage:
#   NODES=12 MAX_VIDEOS=100000 bash data_pipeline/run_synth_gen_slurm.sh
#   NODES=1  MAX_VIDEOS=8  SMOKE=1 bash data_pipeline/run_synth_gen_slurm.sh   # smoke test
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
FV=$WORK/FastVideo
PROMPTS=${PROMPTS:-$FV/examples/dataset/vidprom/prompts/vidprom_filtered_extended.txt}
OUTDIR=${OUTDIR:-$WORK/data/wan22_synth_720p}
NODES=${NODES:-12}
GPUS=${GPUS:-4}
WORKERS=$(( NODES * GPUS ))
MAX_VIDEOS=${MAX_VIDEOS:-100000}
HEIGHT=${HEIGHT:-720}; WIDTH=${WIDTH:-1280}
NUM_FRAMES=${NUM_FRAMES:-121}; DROP=${DROP:-8}; FPS=${FPS:-16}
STEPS=${STEPS:-40}; GS=${GS:-4.0}; GS2=${GS2:-3.0}
SEED_BASE=${SEED_BASE:-1024}; SHUFFLE_SEED=${SHUFFLE_SEED:-1234}
PARTITION=${PARTITION:-all}
TIME=${TIME:-24:00:00}
SMOKE=${SMOKE:-0}
CPUS_PER_TASK=$(( 128 / GPUS )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
mkdir -p "$WORK/logs" "$OUTDIR"

[ -f "$PROMPTS" ] || { echo "PROMPTS not found: $PROMPTS"; exit 1; }
echo "NODES=$NODES GPUS=$GPUS -> $WORKERS workers | target=$MAX_VIDEOS videos | out=$OUTDIR"
echo "gen $((NUM_FRAMES+DROP))f -> keep ${NUM_FRAMES} (drop ${DROP}) @ ${WIDTH}x${HEIGHT} ${FPS}fps, ${STEPS} steps, CFG ${GS}/${GS2}"

sbatch --array=0-$(( NODES - 1 )) -N1 --gres=gpu:$GPUS --ntasks-per-node=$GPUS --exclusive --mem=0 \
  --cpus-per-task=$CPUS_PER_TASK -t "$TIME" --requeue -p "$PARTITION" -J synth_gen \
  --chdir="$FV" -o "$WORK/logs/synth_gen_%A_%a.out" -e "$WORK/logs/synth_gen_%A_%a.out" \
  --wrap "srun --chdir=$FV bash -lc '
    set -uo pipefail
    source .venv/bin/activate
    export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
      XDG_CACHE_HOME=$WORK/.cache TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 \
      PYTHONPATH=$FV
    export CUDA_VISIBLE_DEVICES=\$(( SLURM_LOCALID % $GPUS ))
    export TRITON_CACHE_DIR=/tmp/triton_synth_\${SLURM_LOCALID}
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=\$(( 29700 + SLURM_LOCALID ))
    mkdir -p \$TRITON_CACHE_DIR
    W=\$(( SLURM_ARRAY_TASK_ID * $GPUS + SLURM_LOCALID ))
    echo \"[launch] worker \$W/$WORKERS host=\$(hostname) CVD=\$CUDA_VISIBLE_DEVICES\"
    python data_pipeline/gen_synth_worker.py \
      --prompts $PROMPTS --output-dir $OUTDIR \
      --worker-id \$W --num-workers $WORKERS --max-videos $MAX_VIDEOS \
      --height $HEIGHT --width $WIDTH --num-frames $NUM_FRAMES --drop $DROP --fps $FPS \
      --steps $STEPS --guidance-scale $GS --guidance-scale-2 $GS2 \
      --seed-base $SEED_BASE --shuffle-seed $SHUFFLE_SEED
  '"
echo "submitted. monitor: squeue -u \$USER -n synth_gen ; tail -f $WORK/logs/synth_gen_*.out"
echo "progress:  python data_pipeline/merge_synth_manifests.py --output-dir $OUTDIR"
