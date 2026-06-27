# 01 — Stage-0 Data Pipeline

Everything *upstream* of FastVideo's `preprocess` stage: generate videos, extract
point tracks. Lives in `data_pipeline/` (plain scripts, not pipeline stages —
matches repo convention that everything before preprocess is standalone).

## Flow

```
vidprom prompts ─> generate_videos.py ─> videos/*.mp4 + videos2caption.json + merge.txt
                                            │
                  extract_tracks.py  ─> tracks/*.npz  (+ patch points_path into the json)
                                            │
                  v1_preprocess.py --preprocess_task i2v_track ─> parquet (latents+text+tracks)
                                            │
                  WanTrack bidir trainer
```

## Scripts

- `data_pipeline/generate_videos.py` — Wan2.2-T2V-A14B → mp4 + FastVideo manifest.
  Generates at train fps/length so source frames == sampled frames (tracks align 1:1,
  no temporal re-indexing).
- `data_pipeline/extract_tracks.py` — CoTracker3 (`cotracker3_offline` via torch.hub),
  50×50 uniform grid → `tracks/<stem>.npz` with keys `tracks [T,N,2]` (pixel coords),
  `visibility [T,N]`, `grid_size, height, width, num_frames`. Patches `points_path`
  (absolute) into the manifest. Idempotent.
- `data_pipeline/run.sh` — one-shot wrapper (derives num_gpus from `CUDA_VISIBLE_DEVICES`).
- `data_pipeline/convert_trackwan_init.py` — builds the 52-ch init model (see [02](02-model-architecture.md)).
- `data_pipeline/visualize_tracks.py` — overlay tracks on a video (dots + motion tails).

## Current dataset (smoke / overfit set)

```
/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_a14b_720p_24fps/
  videos/vid_0000{00..09}.mp4         # 10 Wan2.2-A14B T2V clips, 720p→preprocessed 480×832
  tracks/vid_0000{00..09}.npz         # CoTracker3 50×50 = 2500 points, 121 frames
  videos2caption.json, merge.txt
  preprocessed_i2v_track/combined_parquet_dataset/   # 10 rows (4+4+2)
```

## Preprocess task `i2v_track`

`fastvideo/pipelines/preprocess/track/track_i2v_preprocess_pipeline.py`
(`--preprocess_task i2v_track`, registered in `v1_preprocess.py`). Required modules:
`text_encoder, tokenizer, vae` (NO image_encoder — no CLIP). It:
- T5-encodes the prompt (text kept),
- VAE-encodes the first frame → `first_frame_latent` (**normalized** at encode time,
  `(latent - mean)/std`, like MatrixGame2),
- loads tracks from `points_path`, normalizes coords to [0,1] by width/height,
  truncates to `num_frames`,
- writes `pyarrow_schema_i2v_track`: `vae_latent`, `text_embedding`,
  `first_frame_latent`, `track_points [T,N,2]`, `track_visibility [T,N]` (+ metadata).

Stored shapes (480×832, 121 frames): `vae_latent [16,31,60,104]`,
`first_frame_latent [16,31,60,104]`, `track_points [121,2500,2]`,
`track_visibility [121,2500]`.

### Reproduce

```bash
OUT=/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_a14b_720p_24fps
CLEAN=/mnt/weka/home/hao.zhang/shao/data/models/hf_cache_clean
CUDA_VISIBLE_DEVICES=1 srun --jobid=1788946 --overlap --ntasks=1 \
  /usr/bin/env HF_HUB_CACHE="$CLEAN" CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/torchrun --nproc_per_node=1 --master_port=29577 \
  fastvideo/pipelines/preprocess/v1_preprocess.py \
  --model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers --data_merge_path "$OUT/merge.txt" \
  --preprocess_task i2v_track --max_height 480 --max_width 832 \
  --num_frames 121 --train_fps 24 --output_dir "$OUT/preprocessed_i2v_track" \
  --preprocess_video_batch_size 1 --samples_per_file 4 --flush_frequency 4
```

## Bugs fixed along the way

- **torchvision 0.26 removed `torchvision.io.read_video`** → switched to `decord`
  (with imageio/ffmpeg fallback) in `extract_tracks.py`, `visualize_tracks.py`,
  `preprocessing_datasets.py`.
- **FA3 `flash_attn_func` returns a `(out, lse)` tuple** → unwrapped in
  `fastvideo/attention/backends/flash_attn.py`.
- **Preprocess dropped 2/10 rows**: `preprocess_pipeline_base.py` only flushed
  complete `samples_per_file` chunks; trailing rows were lost. Fixed with a final
  `dataset_writer.flush(write_remainder=True)`. Now 10/10 (chunks 4+4+2).

## Notes

- Base for the smoke/overfit work is single-expert **FastWan2.1-T2V-1.3B** (avoids
  Wan2.2-A14B MoE dual-expert training; same VAE/UMT5 as the A14B-preprocessed latents).
- Wan2.2-A14B default HF cache is polluted with a causal SF model — use the clean
  cache (`hf_cache_clean`) for bidirectional gen/preprocess.
- CoTracker3 hub cache lives on shared weka so compute nodes work offline.
