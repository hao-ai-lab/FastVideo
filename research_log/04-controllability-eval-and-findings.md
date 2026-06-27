# 04 — Controllability Evaluation & The Key Finding

> **TL;DR:** The overfit model **reconstructs the clips from text + first frame and
> ignores the tracks.** It follows its own GT tracks (because that *is* the clip it
> memorized) but does not follow counterfactual controls. This is structural, not a
> bug — overfitting a few clips gives the model no reason to use tracks.

## The problem with the naive overfit test

Reconstruction (gen-with-GT-tracks ≈ GT clip) **conflates** two things:
1. did the model memorize the video content, and
2. did it learn to follow the tracks.

Because each of the 10 clips is uniquely identified by its (text, first-frame), the
model can minimize loss by a pure content lookup — making the tracks redundant.

## The test (MotionStream's disentangling protocol)

Hold the content conditioning fixed (same first frame + text), vary **only** the
tracks, and measure whether the generated motion follows them via **CoTracker EPE**
(L2 between input control tracks and tracks re-extracted from the generated video).

Controls per clip (`data_pipeline/test_controllability.py`):

| control | what it feeds |
|---|---|
| `gt` | the clip's own tracks (reconstruction baseline) |
| `none` | no tracks |
| `pan_right` | authored global pan (dense) |
| `zoom_in` | authored dolly-in (dense) |
| `drag_dense` | authored local drag, all points present |
| `drag_sparse` | same drag, only handle points active (sparse) |
| `swap` | another clip's tracks (motion transfer) |

## Results — checkpoint-4000, EPE in pixels (480×832), lower = follows control

| control | clip 0 | clip 1 |
|---|---|---|
| **gt** | **9.77** | **29.49** |
| none | n/a | n/a |
| pan_right | 99.09 | 169.17 |
| zoom_in | 33.73 | 146.60 |
| drag_dense | 20.94 | 155.89 |
| drag_sparse | 43.53 | 74.20 |
| swap | 62.97 | 151.26 |

Output videos: `/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/controllability_4k/`
(`clip{0,1}_{control}.mp4`, tracks overlaid, + `clip{0,1}_reference.mp4`).

## Interpretation

- **gt low (9.8 / 29.5)** but every **counterfactual high** (pan 99 / 169). If the model
  followed tracks, counterfactual EPE would be near the gt EPE. Instead it reproduces the
  same content-driven motion no matter the track input. The low gt-EPE is *reconstruction*,
  not control.
- EPE magnitude tracks the control's displacement (pan moves points ~200px → EPE ~100–170;
  zoom/drag smaller → smaller EPE), consistent with "ignores the control, reproduces the clip."
- The model **can** produce coherent motion (gt works), so the high counterfactual EPE is
  *ignoring the control*, not generation failure.
- `drag_sparse` worst on clip 0 (OOD: the model was trained `WANTRACK_AUG=0`, dense-only),
  but even **dense** counterfactuals are ignored, so the sparse caveat is secondary.

**Verdict: content overfit, tracks ≈ dead weight.**

## Why this happens (root cause)

The training loss is reconstruction of 10 fixed clips. Each clip is identifiable from
its (text, first-frame), so the optimal solution is a content lookup; the tracks add no
information needed to cut the loss ⇒ ~no gradient reaches the (zero-init) track encoder
⇒ it stays ≈ zero ⇒ tracks ignored. `WANTRACK_AUG=0` compounds this (tracks are always
present and consistent, never the sole signal), but augments alone won't fix the root
cause: if content determines the output, dropping tracks doesn't raise the loss enough
to force track-reliance.

## The fix — data design, not just more steps/augments

Track-following requires data where **first frame + text under-determine the motion**,
so the model *must* read tracks to reduce loss:

- **Same/similar first frame, multiple different motions** (each with its own tracks).
  This is the key principle that makes tracks *necessary*. Easiest with synthetic / sim
  data (Kubric, moving shapes, a controllable scene): one scene → N authored motions →
  N (video, tracks) pairs sharing frame 0.
- Or MotionStream's route: **scale to diverse data** (they use ~0.6M clips) where motion
  is not pinned by the first frame on average.

This sharpens the earlier "specialized data" idea: not merely "simple single-object
clips," but a set with **motion ambiguity given the content**.

## Related symptom: generated first frame doesn't match GT (same root cause)

Observed: validation/generated videos don't reproduce the GT first frame, which for
I2V "shouldn't happen." Investigated (`data_pipeline/diag_first_frame.py`, figures in
`research_log/figures/firstframe_clip0_*.png`):

- **Not a normalization bug.** First hypothesis was that the preprocessing stored the
  first-frame latent in the wrong space (it uses `vae.scaling_factor/shift_factor`,
  which Wan lacks, vs the per-channel `latents_mean/std` used everywhere else).
  **Refuted empirically:** `decode_latents(first_frame_latent)` reconstructs the GT
  first frame to **MSE 0.1** — the FastVideo Wan VAE's `encode()` already returns
  normalized latents, so the conditioning is in the correct (normalized) space.
  (The fix attempt was reverted; it would have double-normalized.)
- The conditioning is also built identically in training and inference.
- Yet the **generated** frame 0 is the right scene but globally shifted/zoomed
  (MSE ~2889 vs GT) — i.e. the model regenerates the clip rather than copying frame 0.

**Root cause = same as track-ignoring.** (a) The base is a **T2V** model
(FastWan2.1-T2V-1.3B) with the 20 I2V conditioning channels zero-padded in at init, so
I2V frame-0 pinning is learned *from scratch*; and (b) on 10 text-identifiable clips
there is little gradient pressure to use the first-frame conditioning (text already
determines the clip), so it stays under-used — exactly like the tracks. Fixes: data
where conditioning is *necessary* (same as below), and/or init from a real I2V base
(e.g. Wan2.2-I2V), and/or hard frame-0 latent replacement at inference to force pinning.

## What worked

The **eval harness itself is validated** — it correctly caught a model that doesn't
control. That's exactly the instrument we needed before trusting any "it works" claim.
EPE metric: `motion.cotracker_epe` in the FastVideo eval system (see [06](06-tooling.md)).
