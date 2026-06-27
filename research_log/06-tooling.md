# 06 — Tooling (eval + inference + interactive)

Everything built for testing/driving the track-conditioned model. All reuse the same
inference backend for train/inference parity.

## Synthetic track authoring — `data_pipeline/synthetic_tracks.py`

Author control tracks directly (CoTracker is only for *extracting* tracks from real
video; for *testing* you author them). Outputs `tracks [T,N,2]` normalized [0,1] +
`visibility [T,N]`, from a 50×50 frame-0 grid.

- Motion fields: `pan, zoom, rotate, drag, swirl, static`.
- Presets: `pan_{right,left,up,down}, zoom_{in,out}, rotate_{cw,ccw}, swirl,
  drag_center_right` via `preset(name, num_frames, grid_size, strength=...)`.
- **Sparsify** (the drag-a-few-points control): `select_radius` (handle only),
  `select_stride` (coarse grid), `select_random(k)`.
- Motion transfer: `from_npz(track_npz)` loads + normalizes an extracted-track file.
- CLI: `python data_pipeline/synthetic_tracks.py --frame f.png --preset pan_right --out o.png`
  overlays a preset on a first frame (CPU only, sanity check).

## CoTracker EPE metric — `fastvideo/eval/metrics/motion/cotracker_epe/metric.py`

Registered in the FastVideo eval system as **`motion.cotracker_epe`** (auto-discovered,
`BaseMetric` contract, `requires_reference=False`, `higher_is_better=False`,
`needs_gpu=True`). Reads from the sample:

```
sample["video"]                (T,C,H,W) float [0,1]   generated
sample["input_tracks"]         (T,N,2)   control tracks (normalized by default)
sample["input_visibility"]     (T,N)
sample["input_tracks_normalized"]  bool (default True; scaled by W,H)
```

Re-tracks the generated video with CoTracker3 at the input points' query-frame
positions and returns mean EPE (px) + per-frame + coverage. Standalone helper
`compute_epe(frames_thwc, input_tracks_px, input_vis, model, device, max_points=600)`
is exposed for the test script and Gradio app.

## Standalone inference — `data_pipeline/trackwan_infer.py`

Loads a `dcp_to_diffusers`-exported checkpoint into the real `WanTrackModel` wrapper
(exact train/inference parity), on 1 GPU:

- `load_trackwan(export_dir, yaml_path) -> (model, tc)`
- `load_conditioning_from_parquet(data_path, indices, text_len)` — first_frame_latent,
  text embed/mask, vae_latent, GT tracks, caption per clip
- `generate(model, *, first_frame_latent, text_embedding, text_attention_mask,
  track_points, track_visibility, num_steps=30, seed=0)` → normalized latents
  (FlowMatchEuler denoise; `track_points=None` ⇒ no-track)
- `decode_to_pixels(model, latents)` / `decode_reference(model, vae_latent)` → uint8 [T,H,W,3]

## Controllability test — `data_pipeline/test_controllability.py`

Drives the full disentangling protocol from [04](04-controllability-eval-and-findings.md):
generates each control per clip, overlays the control tracks, saves mp4s, prints an EPE
table.

```bash
EXPORT=/mnt/weka/home/hao.zhang/shao/data/models/trackwan_1.3b_overfit4k
CUDA_VISIBLE_DEVICES=1 srun --jobid=1788946 --overlap --ntasks=1 \
  /usr/bin/env HF_HUB_CACHE=/mnt/weka/home/hao.zhang/shao/data/models/hf_cache_clean \
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
  .venv/bin/python data_pipeline/test_controllability.py \
  --export "$EXPORT" --out .../controllability_4k --clips 0 1 --steps 30
```

## Training-time validation callback — `fastvideo/train/callbacks/track_validation.py`

`TrackValidationCallback` (YAML `callbacks.track_validation`): in-process (no checkpoint
load), pulls N fixed clips from the parquet, runs the track-conditioned denoise loop,
decodes, overlays GT tracks, logs `track_val/generated` + one-time `track_val/reference_gt`
to wandb every `every_steps` (+ baseline at start). Knobs: `every_steps, num_val_samples,
num_inference_steps, guidance_scale, grid_stride, validate_at_start, seed`.

## First-frame / trace diagnostic — `data_pipeline/diag_first_frame.py`

Decodes the stored `first_frame_latent` under both space interpretations and compares
to the GT first frame (localizes any conditioning-space bug), shows the current model's
generated first frame, and writes synthetic-track preview overlays on the real first
frame. Outputs PNGs to `research_log/figures/`:
`firstframe_clip0_{1_gt,2_cond_as_normalized_MODELVIEW,3_cond_as_raw,4_generated}.png`,
`firstframe_clip0_compare.png`, `trackpreview_{pan_right,zoom_in,rotate_cw,drag_center_right,swirl}.png`.
Used to refute the first-frame normalization-bug hypothesis (see
[04](04-controllability-eval-and-findings.md)).

## Interactive Gradio demo — `examples/inference/gradio/trackwan/app.py`

MotionStream-style control surface (offline, bidir):
- pick a clip (first frame + text + optional GT tracks),
- modes: `preset` (pan/zoom/rotate/swirl), `drag` (click the first frame to set the
  handle center + dx/dy/radius), `transfer` (another clip's tracks), `gt`, `none`,
- sparsity: `full | radius (handle only) | coarse grid`,
- **Preview tracks** (overlay on frame 0) and **Generate** (→ video + EPE).

```bash
CUDA_VISIBLE_DEVICES=1 srun --jobid=1788946 --overlap --ntasks=1 \
  /usr/bin/env HF_HUB_CACHE=/mnt/weka/home/hao.zhang/shao/data/models/hf_cache_clean \
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
  .venv/bin/python examples/inference/gradio/trackwan/app.py \
  --export $EXPORT --num-clips 10 --port 7860
# then SSH-forward 7860 and open http://localhost:7860
```

> Reminder: on the `WANTRACK_AUG=0` checkpoint, sparse controls are OOD — use dense
> presets to evaluate it. Sparse needs an augments-on model.
