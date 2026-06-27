# 07 — Open Questions & Next Steps

## The blocker

Overfitting a few clips can't teach track-following: content (text + first frame)
fully determines the output, so tracks are ignored ([04](04-controllability-eval-and-findings.md)).
Everything else (pipeline, training, reconstruction, eval harness) works.

## Prioritized next steps

### A. Force track-reliance with a "same first frame, N motions" set  ← highest value
Build a tiny dataset where one (or a few) first frame(s) map to **multiple different
motions**, each with its own tracks. Then content can't determine the output and the
model must read tracks to cut the loss. Candidates:
- **Synthetic / sim (cleanest, exact tracks):** Kubric or moving-shapes — render one
  scene with N authored motions; tracks are exact, counterfactuals trivial.
- **Real, controllable:** robot pushing (Bridge / Something-Something "push") — same
  scene, different push directions.
Success = counterfactual EPE drops toward gt EPE.

### B. Quantify how much augments alone help
Re-run the overfit with `WANTRACK_AUG=1` (point subsampling + p_mask + motion-CFG
dropout) and re-measure EPE. Expectation: helps sparse control / robustness but does
**not** fix content-overfit on few clips. Also required before sparse drag works in the
Gradio app. (Resume into a NEW output_dir or fresh — don't overwrite checkpoint-4000.)

### C. Scale data (the real route)
Move toward MotionStream-scale data (diverse clips) so motion isn't pinned by the first
frame on average. This is the path to a model that genuinely controls, en route to the
14B / 720p / 24fps target.

### D. Productionize the harness
- Add EPE + counterfactual modes to the **training-time** validation (log control-EPE to
  wandb each eval), so controllability is tracked during training, not just post-hoc.
- Add a `none`-control "does it still move?" check (content-overfit detector) to wandb.
- Camera-control tracks via monocular depth (MotionStream mode 3) — currently only
  parametric pan/zoom/rotate.

## Deferred (later stages)

- Scale to **A14B MoE** (dual-expert boundary training).
- **Causal** finetune (Stage 2), **step distillation** (Stage 3), **self-forcing**
  (Stage 4) → real-time interactive (StreamDiffusion-v2-style inference).
- True freehand trajectory painting in the Gradio app (custom JS / ImageEditor stroke
  capture) — current app uses click-to-set-handle + parametric drag.

## Resolved questions

- *Is the per-step loss a good overfit signal?* No — flow-matching per-step loss is
  dominated by the random timestep; use windowed median + reconstruction PSNR + EPE.
- *Does more steps help?* Yes for reconstruction (600→4000 big gain), but not for
  control (orthogonal — needs data design).
- *Is first_frame_latent normalized consistently?* Yes — normalized at preprocess
  (std~1.3), same space as the (normalized) main latents. No bug.
- *Synthetic tracks hard to author?* No — pan/zoom/drag are a few lines of numpy; only
  *extraction* from real video needs CoTracker.

## Watch-outs

- `WANTRACK_AUG=0` checkpoints are dense-only → sparse control is OOD.
- `checkpoints_total_limit=2` rotates old checkpoints — copy any you want to keep before
  resuming into the same `output_dir`.
- Wan2.2-A14B default HF cache is polluted (causal SF model) — use `hf_cache_clean`.
- Validation video MSE/PSNR is single-seed noisy; trust the trend, not one checkpoint.
