# Data Pipeline

All commands run from `/home/hal-kevin/FastVideo`.

---

## Stage 1 — Generate Videos

```bash
python data_pipeline/generate_videos.py \
  --prompts examples/dataset/motion-test/prompts.txt \
  --output-dir /home/hal-kevin/data/motion-stream-test \
  --num-videos 100 \
  --num-gpus 4 \
  --num-inference-steps 40
```

Output: `motion-physics/videos/vid_000000.mp4 ...`

---

## Stage 2 — VAE Round-Trip Videos

Encode then decode each video through the FastVideo WanVAE so CoTracker runs on
the same frames that training will see (Stage 5 re-encodes these, so tracks must
be extracted from the decoded version, not the raw source).

```bash
python data_pipeline/decode_roundtrip_videos.py \
  --data-dir /home/hal-kevin/data/motion-stream-test \
  --vae-path /home/hal-kevin/models/trackwan_1.3b_i2v_control_init/vae
```

Output: `motion-physics/roundtrip_videos/vid_000000.mp4 ...`

---

## Stage 3 — Extract Tracks

Run CoTracker on the round-trip videos, parallelized across 4 GPUs:

```bash
bash data_pipeline/run_extract_tracks.sh
```

Pass extra args (e.g. `--force`, `--limit 5`) directly — they are forwarded to each worker.
The script no longer hardcodes `--force`; existing `.npz` are skipped unless you pass it:

```bash
bash data_pipeline/run_extract_tracks.sh --force
```

Speed knobs: `--sam-batch 16` (frames per batched FastSAM forward, default 16) and `--amp`
(bf16 autocast for CoTracker, ~1.5-2x faster but slightly different coords — validate before
adopting). Entry events now share one extra CoTracker pass (queries filtered to the new-object
regions) instead of a full 2500-point pass per mask.

**Fused mode:** `--segment` (with `--vis-override-every 3`) runs Stage 4 inside this pass —
object IDs, vis override, and track weights — reusing the decoded video and the entry-detection
FastSAM masks, so Stage 4 does not need to run at all. Same results as the standalone stage
(shared implementation). `--viz`/`--viz-dir` render the same overlay mp4s as standalone Stage 4
(slow — skip for large-scale runs); `--min-area-frac`/`--max-masks` remain standalone-only.
Benchmark it with `FUSED=1 bash data_pipeline/benchmark_tracks.sh` (add `VIZ=1` for renders).

Single-GPU alternative:

```bash
python data_pipeline/extract_tracks.py \
  --data-dir /home/hal-kevin/data/motion-stream-test \
  --videos-subdir roundtrip_videos \
  --grid-size 50 \
  --device cuda \
  --detect-entries \
  --sam-conf 0.75 \
  --sam-iou 0.9 \
  --sam-imgsz 1024
```

Output: `motion-stream-test/tracks/vid_000000.npz ...`

---

## Stage 4 — Segment Tracks

Assign object IDs and compute motion weights, parallelized across 4 GPUs:

```bash
bash data_pipeline/run_segment_tracks.sh
```

The script no longer hardcodes `--force` (pass it to re-process npz that already have
`object_ids`). Each video is now decoded once and FastSAM runs in batched forwards
(`--sam-batch 16`), shared between object-ID assignment and the vis override sweep.

Single-GPU alternative:

```bash
python data_pipeline/segment_tracks.py \
  --data-dir /home/hal-kevin/data/motion-stream-test \
  --videos-subdir roundtrip_videos \
  --conf 0.75 --iou 0.9 --imgsz 1024 \
  --vis-override-every 3 \
  --force \
  --viz
```

Adds `object_ids`, `n_objects`, `track_weights` to each `.npz`.

---

## Stage 5 — Preprocess to Parquet

Reads raw `videos/` for VAE encoding and `tracks/` for track data.

```bash
torchrun --nproc_per_node=1 -m fastvideo.pipelines.preprocess.v1_preprocess \
  --model_path /home/hal-kevin/models/trackwan_1.3b_i2v_control_init \
  --data_merge_path /home/hal-kevin/data/motion-stream-test/data_merge.txt \
  --output_dir /home/hal-kevin/data/motion-stream-test/preprocessed_i2v_track \
  --preprocess_task i2v_track \
  --num_frames 121 \
  --num_latent_t 31 \
  --train_fps 24 \
  --max_height 480 \
  --max_width 832 \
  --preprocess_video_batch_size 1 \
  --samples_per_file 64
```

`--train_fps` must match the source video fps (24 here). If omitted it defaults to 30,
and `FrameSamplingStage` resamples with interval `fps/train_fps = 0.8` — duplicating
every 5th frame and covering only the first ~97 of 121 frames. The stored latents then
encode slowed, stuttering motion that no longer aligns with the tracks (extracted at
native fps), which shows up as drifting motion in validation reference videos.

Output: `motion-physics/preprocessed_i2v_track/combined_parquet_dataset/`

---

## Stage 6 — Training

```bash
python -m fastvideo.train.train \
  --config examples/train/scenario/worldmodel/finetune_wantrack_golf_overfit.yaml
```

Update the yaml to point at your data and checkpoint directories.
