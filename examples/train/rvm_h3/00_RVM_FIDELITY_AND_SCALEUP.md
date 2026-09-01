# RVM fidelity and FastH3 scale-up contract

This file records the method-level corrections required after the first
successful 4×H100 runtime pilots. New scientific runs must use the committed
paper-faithful path, not the legacy per-group-normalized pilot implementation.

## Corrections

### Reward normalization

For prompt group `g` and candidate `i`:

```text
C[g,i] = R[g,i] - mean_i R[g,i]
s_global = std over every raw aggregate reward R[g,i] in the rollout collection
A[g,i] = 0.1 * clip(C[g,i] / (s_global + 1e-4), -5, 5)
```

The mean is prompt-relative; the standard deviation is batch-global across DP
replicas. Only SP leaders contribute sufficient statistics, so each sample is
counted once.

### Training time

The published RVM endpoint regression samples:

```text
t ~ Uniform(0,1)
```

FastH3 then maps the shared base time through its video/audio scheduler shifts.
The four deployment timesteps remain the behavior-rollout schedule; they are not
the default distribution for the analytic training state. A deployment-grid
training-time mode remains available only as a named ablation.

### Motion diagnostics

The optimized Dynamic Tracking score stays clipped to `[0,1]`. The same RAFT
forward also records:

```text
dynamic_tracking_raw
dynamic_tracking_saturation
```

These diagnostics do not affect reward weighting. They reveal when the motion
term has become a saturated guardrail rather than a useful ranking signal.

### Reproducibility

Numbered launch scripts execute `verify_clean_source.py` before training and
record Git `HEAD` and `HEAD^{tree}`. Reported runs must never use dirty tracked
source or uncommitted runtime patches.

## Production defaults

| Setting | Value |
|---|---|
| Behavior sampler | four-step FastH3 VSA |
| Base steps | `1000,750,500,250` |
| CFG | off; guidance 1.0, dropout 0.0 |
| Geometry | `480x832x124`, 24 FPS |
| VSA | 90% sparsity |
| LoRA | rank 128, alpha 64, Q/K/V/out only |
| Advantage | scale 0.1, clip 5 |
| Training time | continuous Uniform(0,1) |
| Default paper reference | no video/audio anchor |
| Reward | TA 1.5, MQ 1.0, HPS general 0.1, HPS percentile 0.1, DT 0.7 |

Audio-only and full anchors are matched H3 safety ablations. They must not be
silently mixed into the paper-reference run.

## Required sequence

### 1. Preflight and compact smoke

```bash
bash examples/train/rvm_h3/03_preflight_1gpu.sh
bash examples/train/rvm_h3/04_run_1gpu_smoke.sh
```

### 2. Eight-GPU topology, resume, and export

```bash
bash examples/train/rvm_h3/05_run_8gpu_topology_smoke.sh
```

This uses SP4×DP2, K=8, four global prompt groups, checkpoints after one update,
resumes to update two, then exports the LoRA.

### 3. Learning-rate sweep

```bash
bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
```

Defaults: `5e-6,1e-5,2e-5`, K=8, four global prompt groups, eight optimizer
updates, 32 fixed validation prompts, and evaluation every 5% of optimizer
progress. Add `5e-5` explicitly only after lower rates are stable.

### 4. Anchor sweep

```bash
RVM_SELECTED_LR=<winner> \
  bash examples/train/rvm_h3/06_run_8gpu_anchor_sweep.sh
```

Compare exact, audio-only `1e-3`, and full `1e-3` anchors under matched prompts,
seeds, reward stack, LR, and sample budget. Evaluation remains every 5%.

### 5. Medium trend run

```bash
RVM_SELECTED_LR=<winner> \
RVM_SCALEUP_CONFIG=<winning-config> \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Defaults: 50 optimizer steps, 8 prompts per collection, K=8, 1,600 rewarded
endpoints, at least 4,096 encoded training prompts, and 32 fixed validation
prompts every 5%.

### 6. Full run

```bash
RVM_FULL_APPROVED=1 \
RVM_SELECTED_LR=<winner> \
RVM_VIDEO_ANCHOR_BETA=<winner> \
RVM_AUDIO_ANCHOR_BETA=<winner> \
  bash examples/train/rvm_h3/07_run_8gpu_full.sh
```

The full campaign uses 90 collections, 32 prompts per collection, K=8, 180
optimizer updates, 23,040 rewarded endpoints, and 100-prompt evaluation every
nine steps. By default the launcher requires at least 48,000 encoded training
prompts, corresponding to the complete pinned corpus after reserving evaluation
prompts.

## Go/no-go rule

Do not launch the full campaign until:

1. SP4×DP2/K8 checkpoint, resume, and export pass from one immutable clean SHA.
2. The selected LR improves paired held-out aggregate reward.
3. VideoAlign TA and MQ do not show a persistent meaningful decline.
4. Gradient clipping is uncommon rather than the dominant update regime.
5. Batch-global reward variance is finite and not collapsed.
6. DT saturation is measured and understood.
7. Full-video inspection shows no static, flicker, repetition, camera-motion, or
   oversaturation exploit.
8. Trained-checkpoint audio remains present and coherent.

The previous 34-step 4×H100 run remains useful evidence that the model/reward/
optimizer/checkpoint stack works, but it used the legacy normalization and
training-time choices. It is not evidence that the corrected RVM recipe has
already improved quality.
