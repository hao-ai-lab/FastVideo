# First-frame saturation in Wan2.2-T2V-A14B synthetic generation

**Date:** 2026-07-12 · **Scope:** base Wan2.2-T2V-A14B (14B MoE), T2V, 720p/121f, 40 steps.

## TL;DR
The over-bright / blown-highlight **first frame** is **real** but **content-dependent**
(high-key scenes: sky / sun / water). Root cause is **NOT the VAE decode** — the causal
VAE renders frame 0 faithfully. The artifact lives in the **diffusion model's latent for
temporal position 0**, which is architecturally special in the Wan causal-VAE latent
layout (position 0 → 1 output frame; positions ≥1 → 4 frames each). It is not fixable in
the decoder. **Fix = over-generate and drop the first frames** (generate 129, drop 8, keep
121 → frame-0 blown-pixel% falls 20.8% → 0.9%).

## Evidence

### 1. Reproduction (real generations, 720p/121f, seed 1024)
Per-frame % of near-white (≥250) "blown" pixels; baseline = settled median (frames ≥10):

| scene | f0 mean dev | f0 clip% | f4 clip% | f8 clip% | baseline clip% |
|-------|-------------|----------|----------|----------|----------------|
| cottage (sunset/ocean, high-key) | **+8.3%** | **20.8%** | 7.5% | 0.9% | 0.1% |
| raccoon (sunflowers, warm/mid-key) | −3.1% | 7.8% | 3.7% | ~1.6% | 1.6% |

- Cottage frame 0: the sun+sky is a blown white blob; resolves to a crisp sun disk by
  frame ~8. See `synth_gen_scratch/report_strip_cottage.png`.
- Raccoon: no *visible* artifact (frame 0 ≈ rest); the warm scene has no large bright
  region to clip. See `report_strip_raccoon.png`. → **content-dependent.**
- The effect peaks at frame 0 and decays over the first ~1–2 VAE temporal chunks
  (≈ frames 0–8), i.e. exactly the causal temporal-warmup window.

### 2. The VAE decoder is NOT the source (controlled isolation)
`synth_gen_scratch/vae_roundtrip.py`: build a **constant** clip (all 121 frames identical =
a clean settled frame), encode→decode with the Wan2.2 VAE. Since every input frame is
identical, any per-frame output variation is 100% the VAE.
- Result: frame 0 mean within **−0.2%** of baseline, clip% **identical** (0.1% / 1.5%).
  The causal VAE reconstructs frame 0 faithfully. → decoder exonerated.
- `fastvideo/tests/vaes/test_wan_vae.py` asserts FastVideo's `AutoencoderKLWan` decode is
  byte-identical to diffusers' (`assert_close` atol=1e-5), so the round-trip (run with
  diffusers) faithfully represents the generation decode path. → no FastVideo decode bug.

### 3. Conclusion
Frame 0 comes out bright because the **DiT places more energy in the position-0 latent**,
not because the decoder mishandles it. In the causal-VAE latent space, position 0 encodes a
single frame (vs 4 for later positions) and carries different training statistics; for
high-key scenes the model's position-0 latent decodes to blown highlights. This is a known
property of Wan's causal-VAE latent structure (the user's "causal VAE" intuition is
directionally right — it's the causal-VAE *latent layout*, not the decoder). The default
pipeline already CFG-negatives against 过曝/色调艳丽 (overexposed/gaudy), which is why mid-key
scenes look fine.

## Fix (script-side, lossless)
Over-generate `121 + drop` frames (must be 4k+1) and drop the first `drop` decoded frames:

| drop | generate | kept f0 clip% | note |
|------|----------|---------------|------|
| 0 | 121 | 20.8% | artifact present |
| 4 | 125 | 7.5% | good, +3.3% compute |
| **8** | **129** | **0.9%** | clean for all content, +6.6% compute (recommended) |

Implemented in `synth_gen_scratch/gen_wan22_clean.py` (`--num-frames 121 --drop 8`).
Especially important for the track-conditioned I2V dataset, where frame 0 becomes the
conditioning image — a blown frame 0 would poison conditioning.

## UPDATE: the dominant driver is OUT-OF-DISTRIBUTION CLIP LENGTH (121f on an 81f model)
Community (ComfyUI issue #9091): A14B is natively **16 fps / 81 frames**; the 5B TI2V is the
**24 fps / 121-frame** model. "When using 121 frames the first few frames always have the
burn-in look; when using 81 frames it is fine." We run A14B at 121f — ~50% past native.

Confirmed empirically (cottage, same seed 1024):
| config | raw frame0 blown% | gen time (1×GB200) |
|--------|-------------------|--------------------|
| A14B **121f** raw | **22.3%** | 915 s |
| A14B **81f native** raw | **1.6%** (no drop needed) | **497 s** |
| A14B 121f + drop8 | 1.6% | 1006 s |
See `synth_gen_scratch/report_length_compare.png`. So the frame-0 burn is mostly the OOD
121-frame regime, not the VAE. Note 81f@16fps and 121f@24fps are the SAME ~5.05 s clip —
the difference is temporal density (fps), not duration.

**Why I2V is immune:** in I2V frame 0 is the real input image, VAE-*encoded* and pinned into
latent position 0 — the DiT never generates it, so it can't overshoot. This corroborates the
root cause (generated position-0 latent), and is a known community observation.

### Mitigation options (drop-frames is NOT the only one)
1. **Drop first 8** (gen 129) — keeps 121f/24fps, content-agnostic, lossless. Used for Phase 2.
2. **Generate native 81f/16fps** — clean with NO drop, ~2× faster; but 16 fps (lower temporal
   density) and off the 121f/24fps training spec.
3. **81f native → interpolate to 121f** (RIFE/FILM) — 2× faster gen; but interpolated motion is
   smoothed → questionable as training data for a motion/track model.
4. **Latent surgery**: overwrite position-0 latent with position-1 before decode — keeps 121f,
   no extra gen; frame 0 becomes ~duplicate of frame 1.
5. **Color-match frame0→frame1** (post) — cheap but CANNOT recover clipped highlights (info lost).
6. **5B TI2V** — native 121f/24fps; different/smaller model.

## Note for scale-out (Phase 3)
Base A14B at 40 steps = **~915 s (~15 min) per 720p/121f video on one GB200**. 100k videos ≈
27k GPU-hours. The prior scale generation almost certainly used the few-step **self-forcing
causal** distilled model (8 steps) for speed — which is *more* prone to this artifact. Decide
model/step budget explicitly before scale-out.
