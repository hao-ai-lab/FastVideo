# Data Pipeline — Decisions & Rationale

## Why generation + tracking live OUTSIDE `fastvideo/pipelines/preprocess/`
FastVideo's preprocess pipeline is structurally a *consumer of existing .mp4 files*
(`VideoCaptionMergedDataset` reads a `merge.txt`→`annotation.json` over a folder and
asserts each video path exists). There is no seam to *produce* a video inside it. So
generation (Wan2.2) and tracking (CoTracker) are standalone Stage-0 scripts that
produce the files preprocess later ingests — matching the repo convention (everything
upstream of preprocess is plain scripts, not pipeline stages).

## Two scripts, not one fused process
- The `.mp4` on disk is the checkpoint; generation is the expensive part. Tracking
  config will be tweaked many times — must not re-run Wan each time.
- Wan2.2-14B is huge (multi-GPU + offload); CoTracker is tiny. "Generate all → free
  model → track all" is the natural ordering and is exactly two phases.
- Independent restart/parallelism. Both scripts are idempotent (skip done work).

## Generate at training fps/length (no temporal alignment headache)
CoTracker tracks per *video* frame, but the VAE compresses time (~4× for Wan). If we
generated at arbitrary fps we'd have to re-index tracks by the preprocess frame-sampler
(`sample_frame_index`) and then fold onto latent time. Since we *control* generation,
we emit videos already at the train fps/length (default 16 fps, 81 frames ≈ 5 s) so
source frames == sampled frames, 1:1. Tracks then align to frames trivially; the only
remaining fold (frames→latent-frames) happens in the model/trainer.

## Generation is T2V even though the model is I2V+points
Synthetic videos come from text→video (Wan2.2-T2V-A14B). At *training* time the model
uses frame 0 as the I2V conditioning image + the point tracks as motion control. So we
never need an input image during generation.

## Manifest format (FastVideo-compatible)
`generate_videos.py` emits the same shape the existing loader expects:
- `videos/vid_NNNNNN.mp4`
- `videos2caption.json`: list of `{path(basename), cap:[...], fps, duration, num_frames, resolution}`
- `merge.txt`: one line `<videos_dir>,<videos2caption.json>`
Loader actually consumes only `path, cap, fps, duration` (+ optional conditioning path);
`resolution`/`num_frames` are informational. We know all of these at generation time, so
we write the manifest directly and SKIP `scripts/dataset_preparation/prepare_json_file.py`
(it exists only to *recover* fps/duration by re-probing videos of unknown provenance).

## points_path stored as ABSOLUTE path
The future preprocess loader joins `folder + relative_path`. Tracks live in a sibling
`tracks/` dir, not under `videos/`. Storing `points_path` absolute makes `os.path.join`
return it unchanged, so it resolves regardless of the manifest folder. Mirrors how
MatrixGame2 references `action_path`, which is the pattern the points preprocess task
will copy.

## CoTracker v3 via torch.hub (`cotracker3_offline`)
The repo only vendors CoTracker **v2** under `third_party/eval/vbench` (eval-only). For
v3 we pull `facebookresearch/co-tracker` `cotracker3_offline` via torch.hub. Prefetch on
the login node (has internet); the hub cache lives on shared weka so compute nodes reuse
it offline (`extract_tracks.py` falls back to `source="local"` from the cache dir).

## Output location OUTSIDE the git repo
Generated videos are large (≈2–5 MB each → ~25 GB at 5k). Default output dir is
`/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/...` (outside `FastVideo/`) to avoid
bloating the working tree / accidental commits.

## Cluster
- `shao_wm` = SLURM job **1788946** on node **fs-mbz-gpu-538**: 8×GPU (~140 GB each),
  128 CPU, 768 GB RAM, 5-day limit. As of setup all 8 GPUs were idle.
- Attach with `srun --jobid=1788946 --overlap ...`. Never run heavy work on the login
  node (`fs-mbz-login-big-001`) — it lags the shared connection.
- `sqs` is a `.bashrc` alias (`squeue -u hao.zhang`); not available in non-interactive
  shells — use `squeue -u $USER` directly.
