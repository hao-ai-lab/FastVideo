# Data Pipeline — Runbook

All heavy work runs on the `shao_wm` allocation via `srun --overlap`. Do NOT run model
code on the login node.

## 0. Env + node handles
```bash
PY=/mnt/weka/home/hao.zhang/shao/FastVideo/.venv/bin/python
JOBID=$(squeue -u "$USER" -n shao_wm -h -o %i | head -1)   # shao_wm allocation id
echo "jobid=$JOBID"
# live GPU usage on the node (lightweight):
srun --jobid="$JOBID" --overlap nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

## 1. One-time setup (login node — has internet)
```bash
cd /mnt/weka/home/hao.zhang/shao/FastVideo
# prompts (self-forcing / vidprom):
[ -f examples/dataset/vidprom/prompts/vidprom_filtered_extended.txt ] || \
  ( cd examples/dataset/vidprom && ./download_dataset.sh )
# prefetch CoTracker v3 into the shared torch.hub cache so compute nodes work offline:
$PY -c "import torch; torch.hub.load('facebookresearch/co-tracker','cotracker3_offline'); print('cotracker ok')"
```

**Pick free GPUs yourself** from step 0 and pass them via `CUDA_VISIBLE_DEVICES` — the node
is shared and the scripts do NOT auto-select. Pinning to busy GPUs OOMs.

## 2. Generate videos (Wan2.2-T2V-A14B, 720p, 16fps, 81f)
```bash
GPUS=2,3   # <- two currently-free GPUs from step 0
srun --jobid="$JOBID" --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=$GPUS \
  $PY data_pipeline/generate_videos.py \
  --prompts examples/dataset/vidprom/prompts/vidprom_filtered_extended.txt \
  --output-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p \
  --num-videos 50 --num-gpus 2          # num-gpus must match the device count
```
- Smoke first: `--num-videos 2`.
- Idempotent: re-run resumes from `manifest.jsonl`.

## 3. Extract point tracks (CoTracker v3, 50×50 grid)
```bash
srun --jobid="$JOBID" --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=$GPUS \
  $PY data_pipeline/extract_tracks.py \
  --data-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p \
  --grid-size 50
```
- Idempotent: skips videos whose `tracks/<stem>.npz` exists.
- If 720p OOMs CoTracker, add `--downscale 0.5` (coords are rescaled back to original px).

## 4. One-shot wrapper (derives num_gpus from CUDA_VISIBLE_DEVICES)
```bash
CUDA_VISIBLE_DEVICES=2,3 bash data_pipeline/run.sh            # full run (defaults)
CUDA_VISIBLE_DEVICES=2,3 bash data_pipeline/run.sh --smoke    # 2 videos
```

## Outputs
```
<output-dir>/
  videos/vid_000000.mp4 ...
  tracks/vid_000000.npz ...        # keys: tracks (T,N,2 px), visibility (T,N), grid_size, height, width, fps, num_frames
  videos2caption.json              # FastVideo manifest (+ points_path after step 3)
  merge.txt
  manifest.jsonl                   # incremental resume log (generation)
```

## Gotchas
- Check GPU availability before launching (step 0) — the node may not be fully free.
- `generate_video(...)` logs a Deprecation warning (use of legacy API) — harmless.
- First Wan2.2 run loads ~14B MoE weights; expect a few min of startup before frame 1.
