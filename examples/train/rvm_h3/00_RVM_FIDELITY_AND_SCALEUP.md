# FastH3 RVM fidelity and 8-GPU scale-up runbook

This document supersedes any earlier description that said FastH3 RVM divides
reward residuals by a separate standard deviation for each prompt group or
trains only at the four deployment timesteps. The production implementation now
matches the published RVM video update on both points.

## 1. Exact training update

For prompt group `g` and candidate `i`, let `R[g,i]` be the weighted reward:

```text
1.5 * VideoAlign TA
+ 1.0 * VideoAlign MQ
+ 0.1 * HPSv3 general
+ 0.1 * HPSv3 percentile
+ 0.7 * Dynamic Tracking
```

RVM centers each candidate against the other samples for the same prompt but
uses one standard deviation over every reward in the rollout collection:

```text
centered[g,i] = R[g,i] - mean_i R[g,i]
global_std = std_{g,i}(R[g,i])
advantage[g,i] = 0.1 * clip(centered[g,i] / (global_std + 1e-4), -5, 5)
```

This distinction matters. A prompt whose candidates differ only by scorer noise
must not receive the same update magnitude as a prompt with a large, meaningful
reward spread. The implementation computes global sufficient statistics only
on sequence-parallel leaders and all-reduces them across data-parallel replicas,
so every generated sample is counted exactly once.

For one generated endpoint `x0`, sample a continuous base time:

```text
t ~ Uniform(0, 1)
```

H3 maps this shared base time through its video and audio scheduler shifts. The
endpoint is analytically noised and receives the original flow-matching target:

```text
x_t = (1 - sigma(t)) * x0 + sigma(t) * epsilon
velocity_target = epsilon - x0
```

Only the video slice receives the signed reward coefficient. Audio receives no
visual reward; an optional reference-field anchor is tested separately. The
signed update remains implemented through a detached nonnegative surrogate, so
its gradient equals RVM without exposing AMP to an unbounded negative scalar
loss.

Behavior rollouts remain the exact released four-forward FastH3 VSA sampler at
base steps `1000, 750, 500, 250`. Continuous time applies only to the analytic
post-training regression state. `training_timestep.mode: deployment_grid`
remains available as an explicit H3 ablation, not the production default.

## 2. Fixed scientific contract

Do not silently change:

| Setting | Required value |
|---|---|
| FastH3 checkpoint | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` |
| Rollout steps | `1000,750,500,250` plus terminal zero |
| CFG | guidance `1.0`, conditioning dropout `0.0` |
| Attention | `VIDEO_SPARSE_ATTN_H3` |
| VSA sparsity | `0.90` |
| Geometry | `480x832`, 124 frames, 24 FPS |
| LoRA | rank 128, alpha 64, `to_q/to_k/to_v/to_out` |
| VSA compression gate | frozen |
| Advantage scale / clip | `0.1` / `[-5,5]` |
| Training-time distribution | continuous `Uniform(0,1)` |
| Reward centering / scale | per-prompt mean / batch-global std |

All numbered training scripts fail when tracked source differs from the checked
out Git commit. The source manifest records both `HEAD` and `HEAD^{tree}`.
`RVM_ALLOW_DIRTY_SOURCE=1` exists only for deliberate debugging and must never
be used for a reported experiment.

## 3. New diagnostics

The optimized Dynamic Tracking reward is clipped to `[0,1]`, matching RVM. The
same RAFT pass now also logs two non-optimized diagnostics:

- `dynamic_tracking_raw`: mean unclipped `flow_ratio`;
- `dynamic_tracking_saturation`: fraction of sampled frame pairs at or above
  the clipping threshold.

The training loop also logs:

- global aggregate-reward mean and standard deviation;
- global standard deviation for every reward component;
- mean prompt-group reward standard deviation;
- zero-variance prompt-group ratio;
- mean absolute advantage and clipping ratio;
- gradient norm and whether clipping was applied;
- continuous training-time mean, minimum, maximum, and nearest deployment bin.

Do not interpret a clipped DT value of `1.0` as continued progress when its
saturation fraction is already near one. In that regime DT is a collapse
barrier, not a useful ranking signal.

## 4. Required experiment order

### A. One-GPU correctness gate

```bash
bash examples/train/rvm_h3/03_preflight_1gpu.sh
bash examples/train/rvm_h3/04_run_1gpu_smoke.sh
```

This verifies imports, reward checkpoints, continuous-time config, global-std
unit tests, VSA forward/backward, Adam, checkpointing, and validation. It is not
a quality result.

### B. Exact 8-GPU topology/resume/export gate

```bash
bash examples/train/rvm_h3/05_run_8gpu_topology_smoke.sh
```

This uses `SP4 x DP2`, `K=8`, four global prompt groups, one update, resumes to
a second update, and exports the LoRA. Each DP replica processes two prompts
and 16 videos locally—the same local rollout load as the successful four-GPU
`K=4` pilot.

Accept only when:

- both reward leaders return finite, non-identical candidate rewards;
- global reward statistics agree on all ranks;
- DP replicas receive different prompts while SP ranks agree;
- checkpoint resume advances from step 1 to step 2;
- the exported adapter contains all expected layers;
- validation videos and audio remain valid.

### C. Learning-rate bracket

```bash
bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
```

Defaults:

```text
LRs: 5e-6, 1e-5, 2e-5
K: 8
four global prompt groups
8 optimizer updates
32 fixed validation prompts
validation at baseline and final only
```

The short sweep deliberately does not run 100-video evaluation after every
step. Add the Wan paper's `5e-5` LR through `RVM_LR_SWEEP` only when the lower
bracket is stable; FastH3 is a much larger, already distilled model.

Select the largest LR satisfying all of:

- held-out aggregate reward improves;
- VideoAlign TA and MQ do not materially regress;
- fewer than roughly 10-20% of steps are clipped at norm 1;
- no static/repeated-frame or oversaturation collapse;
- reward global std remains finite and useful;
- audio remains coherent.

### D. Anchor sweep

```bash
RVM_SELECTED_LR=<winner> \
  bash examples/train/rvm_h3/06_run_8gpu_anchor_sweep.sh
```

Compare exact RVM, audio-only anchor `1e-3`, and full video/audio anchor `1e-3`.
The exact unanchored run is the published reference. Choose the audio anchor
only when it measurably protects audio without erasing video reward gains.

### E. Medium scale-up pilot

Prepare at least 4,096 prompts, then run:

```bash
RVM_SELECTED_LR=<winner> \
RVM_SCALEUP_CONFIG=<winning-anchor-config> \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Defaults: 50 optimizer updates, 8 prompt groups per collection, `K=8`, and 32
fixed validation prompts every 5% of steps. This produces 1,600 fresh rewarded
endpoints and is the first run intended to establish a learning trend.

### F. Published-scale run

Only after the previous gates produce a held-out and qualitative win:

```bash
RVM_FULL_APPROVED=1 \
RVM_SELECTED_LR=<winner> \
RVM_VIDEO_ANCHOR_BETA=<winner> \
RVM_AUDIO_ANCHOR_BETA=<winner> \
  bash examples/train/rvm_h3/07_run_8gpu_full.sh
```

The full run uses 90 rollout collections, 32 prompt groups, `K=8`, two attached
updates per collection, 180 optimizer updates, and 23,040 rewarded endpoints.
It evaluates 100 fixed prompts every nine steps, exactly 5% of the optimizer
horizon. Use the complete 48,998-prompt RVM/VidProM bank when storage permits;
the launcher refuses fewer than 10,000 rows by default.

## 5. Checkpoint selection

Never select the final checkpoint automatically. For each fixed validation
prompt, retain the same prompt index and seed across baseline/checkpoints. Track
at minimum:

- aggregate and every component reward;
- VideoAlign TA and MQ separately;
- DT raw and saturation diagnostics;
- group/global reward variance;
- gradient clipping fraction;
- repeated/static/flicker failures;
- independent quality evaluation;
- full-video blinded preference;
- audio presence, quality, and synchronization.

A larger rollout reward on different sampled prompts is not an apples-to-apples
quality claim. Scale only when paired held-out deltas and inspected videos agree.

## 6. References

- **Scaling Reinforcement Learning for Diffusion Models via Velocity Matching**
  (`arXiv:2608.23664`): endpoint RVM objective, batch-global reward standard
  deviation, continuous training time, video reward mixture, and published
  sample budget.
- **Flow-Factory AdvantageProcessor**: independent implementation whose
  `global_std=True` path computes a global standard deviation over aggregated
  rewards while subtracting each prompt-group mean.
