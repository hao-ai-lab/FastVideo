# Waypoint: Compare official vs FastVideo output

Use this to track down why FastVideo can produce green/garbled frames while the official model gives a good image.

## 1. Generate the two images (same prompt and seed)

**Official model (Overworld code + same VAE/text encoder):**

```bash
python test_official_model.py
```

→ Saves `waypoint_official_frame.png` (seed 42, prompt: "first person view of a grassy field with blue sky").

**FastVideo pipeline (same prompt and seed):**

```bash
python run_fastvideo_single_frame.py
# optional: --seed 42 --prompt "first person view of a grassy field with blue sky"
```

→ Saves `waypoint_single_image.png`.

## 2. Compare the images

- **If FastVideo frame is still green/garbled:** The bug is in the FastVideo port (transformer or pipeline). Focus on:
  - RoPE / position IDs in `waypoint_transformer.py`
  - Gated attention and cross-attention
  - Sigma schedule and step integration in `waypoint_pipeline.py`
- **If both look similar (e.g. both bad or both okay):** The issue may be shared (e.g. sigma schedule, VAE scaling, or prompt encoding).

## 3. Optional: match sigma schedule

The official RunPod run used 5 sigmas (4 steps). FastVideo default is 4 sigmas (3 steps). To use 4 steps in FastVideo, the pipeline config already supports `num_inference_steps`; ensure `scheduler_sigmas` in config has 5 values if you want exact step alignment with the official run.

## 4. Next debugging steps (if FastVideo is still wrong)

- Add a one-step numeric comparison: same initial latent `x` and sigma, compare `v_pred` from official transformer vs FastVideo transformer (mean/std/min/max).
- Diff the official `attn.py` / `model.py` (RoPE, block mask, gate) against `waypoint_transformer.py`.
- Confirm conditioning (noise embed, prompt_emb, mouse/button/scroll) is identical in shape, dtype, and value when calling the transformer.

---

## 5. Video generation: how to find the issue

Single-frame works; when you run **videos**, use this to narrow down problems.

### 5.1 Run a short video (on SSH)

```bash
cd /root/FastVideo

# Short run: 10 steps × 1 frame = 10 frames (~0.17 s at 60 fps)
python examples/inference/basic/basic_waypoint_runpod.py \
  --num_steps 10 \
  --seed 42 \
  --output waypoint_video_test.mp4
```

Output: `video_samples_waypoint_runpod/waypoint_video_test.mp4`. Download it and watch.

### 5.2 What to look for

| What you see | Likely cause | Where to look |
|--------------|--------------|---------------|
| **First frame OK, later frames green/bad** | KV cache or frame indexing wrong for frame &gt; 0 | `waypoint_pipeline.py`: `ctx.frame_index`, `ctrl_step`, cache pass (sigma=0); `waypoint_transformer.py`: how `frame_timestamp` / pos_ids use frame index |
| **All frames green/garbled** | Same as single-frame bug (transformer/VAE) | Already fixed if single frame works; if video-only, check per-step shapes (e.g. latent [B,1,C,H,W] vs [B,C,H,W]) |
| **Flicker / random frame to frame** | Cache not updated or wrong frame conditioning | KV cache update (sigma=0 pass), `mouse`/`button`/`scroll` indexed by `ctrl_step` |
| **First frame bad, rest OK** | Unlikely; would point to init vs step path difference | Compare code paths for frame_index==0 vs &gt;0 |

### 5.3 Extract frames from the MP4 (compare frame 0 vs frame 5)

On the server or locally (if you have ffmpeg):

```bash
# Frame 0 (first frame)
ffmpeg -i video_samples_waypoint_runpod/waypoint_video_test.mp4 -vf "select=eq(n\,0)" -vframes 1 frame_000.png

# Frame 5
ffmpeg -i video_samples_waypoint_runpod/waypoint_video_test.mp4 -vf "select=eq(n\,5)" -vframes 1 frame_005.png
```

Compare `frame_000.png` vs `frame_005.png`: if 0 is good and 5 is green, the bug is in multi-frame path (cache or indexing).

### 5.4 Code spots that affect video (not single frame)

- **`waypoint_pipeline.py`**
  - `ctx.frame_index` — incremented each step; used for `frame_ts`, `ctrl_step`.
  - `ctrl_step = min(ctx.frame_index, mouse.shape[1]-1)` — so each step gets the right control slice.
  - The **cache pass** after denoising: `transformer(..., update_cache=True)` with sigma=0; if this is wrong or skipped, later frames get no history.
- **`waypoint_transformer.py`**
  - Position IDs / RoPE for **t** (time): must use frame index so later frames get correct temporal positions.
  - KV cache shape and `update_cache`: must match what the blocks expect.

### 5.5 Quick test: 2-step video

```bash
python examples/inference/basic/basic_waypoint_runpod.py --num_steps 2 --seed 42 --output waypoint_2frame.mp4
```

If frame 0 and frame 1 both look good, the multi-frame path is likely OK. If frame 1 is bad, focus on cache and frame indexing.
