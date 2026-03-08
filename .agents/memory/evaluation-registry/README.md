# Evaluation Metrics Registry

Living catalog of all evaluation metrics for FastVideo-WorldModel video quality
assessment. Each metric includes a detailed explanation, implementation status,
usage instructions, and interpretation guide.

_Last updated: 2026-03-02_

---

## Metric Summary

| Metric | Category | Status | Location | Trust |
|--------|----------|--------|----------|-------|
| **FVD** | Distribution | ✅ Implemented | `benchmarks/fvd/` | High |
| **SSIM** | Reference | ✅ Implemented | `fastvideo/tests/ssim/` | High |
| **LPIPS** | Perceptual | ✅ Implemented | `scripts/lora_extraction/` | Medium |
| **Loss trajectory** | Training signal | ✅ Implemented | W&B `train_loss` | Medium |
| **Grad norm stability** | Training signal | ✅ Implemented | W&B `grad_norm` | Medium |
| **GameWorld Score** | Multi-dim benchmark | 🟡 External | Matrix-Game repo | Low |
| **Human preference** | Gold standard | 🔴 Manual | N/A | Highest |

---

## Implemented Metrics

### FVD — Fréchet Video Distance

**Category**: Distribution-level quality metric
**Status**: ✅ Fully implemented in `benchmarks/fvd/`
**Trust**: High — standard protocol, I3D feature extractor

#### What It Measures
FVD measures the distance between the **distribution** of generated videos and
a distribution of real/reference videos. It works by:
1. Extracting spatiotemporal features from both real and generated video sets
   using a pretrained **I3D** (Inflated 3D ConvNet) model.
2. Modeling each set of features as a multivariate Gaussian (mean + covariance).
3. Computing the **Fréchet distance** between the two Gaussians.

Lower FVD = generated videos are more statistically similar to real videos.

#### Why It Matters
- FVD is the **de facto standard** for benchmarking video generation models.
- It captures both **visual quality** (are individual frames realistic?) and
  **temporal coherence** (do frames flow naturally?).
- Matrix-Game 2.0, Open-Sora, and most video generation papers report FVD.

#### Limitations
- Requires a **large sample set** (standard protocol uses 2048 videos) to
  produce stable statistics. Small sample sizes yield noisy results.
- Measures **distributional similarity**, not per-video quality. A model could
  have low FVD by generating a diverse set of "roughly okay" videos.
- The I3D model was trained on Kinetics-400 (human actions). It may be less
  sensitive to domain-specific artifacts in non-human-action videos (e.g.,
  driving, game environments).
- Does not directly measure text-video alignment or action controllability.

#### How to Use

```python
# Programmatic
from benchmarks.fvd import compute_fvd_with_config, FVDConfig

config = FVDConfig.fvd2048_16f()  # Standard: 2048 videos, 16 frames
results = compute_fvd_with_config('data/real/', 'outputs/gen/', config)
print(f"FVD: {results['fvd']:.2f}")
```

```bash
# CLI
python -m benchmarks.fvd.cli \
    --real-path data/real/ \
    --gen-path outputs/gen/ \
    --protocol fvd2048_16f
```

**Preset protocols**:
| Protocol | Videos | Frames | Use Case |
|----------|--------|--------|----------|
| `fvd2048_16f` | 2048 | 16 | Standard benchmark (papers) |
| `fvd2048_128f` | 2048 | 128 | Long video evaluation |
| `quick_test` | 100 | 16 | Fast dev iteration |

**Feature extractors**: `i3d` (default, standard), `clip`, `videomae`

#### Interpretation
| FVD Range | Interpretation |
|-----------|---------------|
| < 100 | Excellent — near-real quality |
| 100–300 | Good — competitive with SOTA |
| 300–600 | Fair — noticeable gap from real |
| > 600 | Poor — significant quality issues |

> FVD values are dataset-dependent. Always compare against baselines evaluated
> on the same real video distribution.

---

### SSIM — Structural Similarity Index

**Category**: Per-frame reference comparison
**Status**: ✅ Implemented in `fastvideo/tests/ssim/`
**Trust**: High — used in CI regression tests

#### What It Measures
SSIM compares two images (or video frames) based on three components:
1. **Luminance**: brightness similarity
2. **Contrast**: dynamic range similarity
3. **Structure**: spatial pattern similarity

The final score is a value in [0, 1] where 1.0 = identical.

#### Why It Matters
- Used as a **regression guard** in CI: ensures model updates don't degrade
  visual output below a threshold.
- More perceptually meaningful than raw pixel MSE.
- Fast to compute — suitable for automated testing.

#### Limitations
- Requires a **pixel-aligned reference** video. Cannot compare videos with
  different seeds, prompts, or angles.
- Operates **per-frame** — does not capture temporal coherence.
- Insensitive to some perceptual artifacts (color shifts, high-frequency noise).

#### How to Use

```bash
pytest fastvideo/tests/ssim/ -vs
```

#### Interpretation
| SSIM Range | Quality |
|------------|---------|
| > 0.90 | Excellent — very close to reference |
| 0.80–0.90 | Good — acceptable for most uses |
| 0.70–0.80 | Fair — noticeable differences |
| < 0.70 | Poor — significant divergence |

---

### LPIPS — Learned Perceptual Image Patch Similarity

**Category**: Per-frame perceptual distance
**Status**: ✅ Implemented in `scripts/lora_extraction/lora_inference_comparison.py`
**Trust**: Medium — available but only used for LoRA comparison currently

#### What It Measures
LPIPS uses a pretrained neural network (AlexNet by default) to extract
deep features from two images and computes the distance between them in
feature space. Unlike SSIM, LPIPS correlates much more strongly with
**human perceptual judgments**.

Lower LPIPS = more perceptually similar.

#### Why It Matters
- Best available automated proxy for **human visual judgments** at the frame
  level.
- Captures semantic and structural differences that SSIM misses (e.g., texture
  changes, minor recoloring).
- Used for validating LoRA merge quality.

#### Limitations
- Per-frame metric — no temporal awareness.
- Requires reference video (paired comparison only).
- Slightly slower than SSIM due to neural network forward pass.

#### How to Use

```bash
python scripts/lora_extraction/lora_inference_comparison.py \
  --base merged_model \
  --ft path/to/finetuned \
  --adapter NONE \
  --output-dir results \
  --prompt "A cat" \
  --compute-lpips
```

#### Interpretation
| LPIPS Range | Quality |
|-------------|---------|
| < 0.10 | Excellent — nearly indistinguishable |
| 0.10–0.20 | Good — minor perceptual differences |
| 0.20–0.40 | Fair — noticeable differences |
| > 0.40 | Poor — clearly different |

---

### Loss Trajectory

**Category**: Training signal proxy
**Status**: ✅ Active (from W&B `train_loss`)
**Trust**: Medium — proxy, not direct quality measure

#### What It Measures
Tracks the training loss over time. A healthy training run shows:
- **Decreasing loss** over the first hundreds of steps.
- **Stable gradient norms** (no wild spikes).
- **Consistent step times** (no infrastructure issues).

#### Why It Matters
- Cheapest evaluation signal — available in real-time from W&B.
- Critical for the **30-minute quality check** workflow.
- At later training stages (when loss becomes meaningful), trajectory shape
  can predict final model quality.

#### Context: How This Evolves
The team's experience shows evaluation signals change during a project:
- **Early stage**: Loss may be flat or meaningless → focus on SSIM & visual
  inspection instead.
- **Mid stage**: Loss starts decreasing → trajectory shape becomes useful.
- **Late stage**: Loss is meaningful → can compare trajectories across runs.

This dynamic is a key insight from the team's workflow: don't over-rely on
loss early; don't ignore it late.

---

### Grad Norm Stability

**Category**: Training health diagnostic
**Status**: ✅ Active (from W&B `grad_norm`)
**Trust**: Medium — diagnostic, not quality metric

#### What It Measures
The magnitude of gradients during training. Stable grad norms indicate
healthy optimization. Spikes or NaN values indicate training instability.

#### Alert Thresholds
| Condition | Meaning |
|-----------|---------|
| Stable ~0.3–0.5 | Normal training |
| Single spike > 3× average | Possible bad batch, monitor |
| NaN or Inf | 🔴 Training has diverged — stop run |
| Increasing trend | Learning rate may be too high |

---

## External Benchmarks

### GameWorld Score Benchmark (Matrix-Game)

**Category**: Multi-dimensional evaluation framework for interactive world models
**Status**: 🟡 External — not implemented in-repo
**Source**: [Matrix-Game 1.0 benchmark](https://github.com/SkyworkAI/Matrix-Game), used in [Matrix-Game 2.0 paper](https://arxiv.org/abs/2508.13009)

#### What It Measures
A comprehensive benchmark examining **four critical capabilities**:

| Dimension | What It Evaluates | Example Signals |
|-----------|-------------------|-----------------|
| **Visual quality** | Frame-level realism, absence of artifacts | Color fidelity, sharpness, coherence |
| **Temporal quality** | Smoothness across frames, motion consistency | Jitter, flickering, temporal aliasing |
| **Action controllability** | Response to input actions (keyboard/mouse) | Action delay, correctness, smoothness |
| **Physical rule understanding** | Adherence to physics (gravity, collision) | Object persistence, plausible motion |

#### Context from Matrix-Game 2.0
- Evaluation uses **597-frame composite action sequences** over 32 Minecraft
  scenes and 16 wild scenes.
- Action controllability assessment is **Minecraft-specific** — cannot be
  directly applied to wild/general scenes.
- The paper notes that models that "collapse" to static frames can
  paradoxically score higher on consistency metrics — beware of this confound.

#### Relevance to FastVideo
- Matrix-Game 2.0 is built on SkyReels-V2/Wan2.1 architecture — **same model
  family as FastVideo**.
- Their distillation uses DMD-based Self-Forcing — **same technique** as our
  `self_forcing_distillation_pipeline.py`.
- GameWorld Score dimensions are a useful framework for thinking about world
  model quality even outside gaming contexts.

---

## Human Preference Evaluation

**Category**: Gold-standard quality assessment
**Status**: 🔴 Manual process — no automated implementation
**Priority**: **Highest** — this is the most important evaluation signal
**Trust**: Highest — but expensive

#### What It Measures
Human evaluators compare generated videos and rate them on dimensions like:
- Overall quality and realism
- Temporal coherence and smoothness
- Prompt adherence / action correctness
- Absence of artifacts

#### Why It's the Most Important Metric
All automated metrics are **proxies** for human judgment. They can be gamed
or may miss artifacts that humans easily notice. Human preference is the
ultimate ground truth for video generation quality.

#### Cost & Practicality
| Approach | Cost | Scale | When to Use |
|----------|------|-------|-------------|
| Internal team review | Low | ~10–50 videos | Every major checkpoint |
| Crowdsource (MTurk, Scale) | Medium | 100+ videos | Pre-release validation |
| A/B preference test | Medium | Pairs | Comparing two model versions |

#### Recommended Protocol
1. Sample 10–20 videos from the model at a checkpoint.
2. Include diverse prompts (easy + hard, short + long).
3. Have 2–3 evaluators score each video 1–5 on: quality, coherence, fidelity.
4. Record scores in the experiment journal.

---

## Metrics NOT Used

| Metric | Reason |
|--------|--------|
| ~~CLIP-Score~~ | Not used by the team. Measures text-image alignment using CLIP embeddings, but not well-suited for video temporal quality. |
| Inception Score (IS) | Less informative than FVD for video; primarily an image metric. |
| PSNR | Pixel-level metric; less perceptually meaningful than SSIM/LPIPS. |

---

## Adding a New Metric

Follow the SOP: `.agents/workflows/evaluation-development.md`

1. Prototype in `.agents/exploration/`
2. Validate on known-good and known-bad samples
3. Add to this registry
4. Update the `evaluate-video-quality` skill
