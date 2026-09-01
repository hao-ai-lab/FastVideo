# FastH3 RVM GPU-agent runbook

This is the operating contract for the agent running and debugging FastH3 RVM.
The goal is a reproducible quality improvement over the released four-forward
FastH3 checkpoint—not a novel algorithm and not a merely successful process
exit.

## 1. Method

RVM is endpoint-only reward-weighted self-distillation. The behavior policy
samples complete FastH3 outputs, black-box rewards rank those outputs, and each
endpoint is analytically noised once for the model's native velocity-matching
update.

For prompt group `g` and candidate `i`:

```text
R[g,i] = weighted reward
C[g,i] = R[g,i] - mean_i R[g,i]
s = population_std over every R[g,i] in the rollout collection
A[g,i] = 0.1 * clip(C[g,i] / (s + 1e-4), -5, 5)
```

The numerator is prompt-relative; the denominator is batch-global. Do not
replace this with a separate unit-variance normalization for every K-sample
group. That would amplify low-spread scorer noise to the same magnitude as
large reward differences.

For each endpoint `x0`, sample:

```text
t ~ Uniform(0, 1)
epsilon ~ Normal(0, I)
x_t = (1 - sigma(t)) * x0 + sigma(t) * epsilon
v_target = epsilon - x0
```

H3 maps the shared base time through video shift 12 and audio shift 3. The video
output gradient is:

```text
A * (v_video - v_target_video)
+ beta_video * (v_video - v_reference_video)
```

Audio receives no visual reward coefficient:

```text
beta_audio * (v_audio - v_reference_audio)
```

The implementation builds a detached target `v - gradient` and minimizes a
nonnegative MSE against it. Its derivative is exactly the signed RVM update.
Do not replace it with a literal negative-weighted MSE.

The behavior rollout is always the released FastH3 four-step VSA policy. The
continuous time is used only for the forward-noised RVM regression example.
For VSA at a continuous state, the nearest released deployment-step mask is
used. Treat deployment-grid training as an ablation, not the default.

## 2. Reward stack

Production weights:

```text
videoalign_ta       1.5
videoalign_mq       1.0
hpsv3_general       0.1
hpsv3_percentile    0.1
dynamic_tracking    0.7
```

Meaning:

- **VideoAlign TA:** visual prompt adherence.
- **VideoAlign MQ:** motion quality, evaluated on grayscale video.
- **HPSv3 general:** mean frame preference under `A high-quality image`.
- **HPSv3 percentile:** prompt-conditioned mean of the top 30% frames.
- **Dynamic Tracking:** clipped RAFT top-5%-pixel flow reward.

VideoAlign runs temporary MP4s at 8 FPS, matching the public GenRL reward path.
HPSv3 evaluates up to 53 evenly sampled H3 frames and preserves the published
mean/top-30% aggregation while chunking inference to avoid OOM.

Dynamic Tracking additionally reports:

```text
dynamic_tracking_raw
dynamic_tracking_saturation
```

The raw metric is the unclipped flow ratio; saturation is the fraction of frame
pairs at or above the clipped reward ceiling. These are diagnostics only and do
not alter the weighted reward.

A DT reward near 1.0 is not necessarily continued progress. When saturation is
high, DT acts mainly as a static-collapse guardrail. Inspect global/camera motion
qualitatively before increasing its weight.

## 3. Non-negotiable model contract

| Variable | Value |
|---|---|
| Checkpoint | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` |
| Base rollout steps | `1000,750,500,250` |
| DiT calls | 4 |
| CFG | guidance 1.0; conditioning dropout 0.0 |
| Attention | `VIDEO_SPARSE_ATTN_H3` |
| VSA sparsity | 0.90 |
| Video shift | 12 |
| Audio shift | 3 |
| Geometry | 480x832, 124 frames, 24 FPS |
| LoRA target | `to_q,to_k,to_v,to_out` |
| Full LoRA | rank 128, alpha 64 |
| VSA compression gate | frozen |
| LoRA master dtype | FP32 |
| Reward text | visual description only |
| Training time | continuous Uniform(0,1) |
| Reward center/std | per-prompt / global collection |

Do not silently switch to dense attention, add CFG, change the four rollout
steps, shorten the production video, include audio text in visual rewards, or
train the VSA gate just to clear a runtime error.

## 4. Data

The primary prompt source is the pinned DanceGRPO/VidProM list. Preparation:

1. downloads or reads the exact source;
2. normalizes and deduplicates prompts;
3. shuffles with a fixed seed;
4. reserves the fixed evaluation split;
5. writes H3 documents;
6. precomputes Qwen3-VL layer-50 conditioning embeddings;
7. records the source SHA-256 and row counts.

H3 documents use:

```text
integrated_multimodal_description: <visual prompt>
overall_soundscape: N/A
non_diegetic_music: N/A
```

The visual reward receives only `integrated_multimodal_description`.

Prompt embeddings are large. Use 256 prompts for smoke tests, at least 4,096 for
the medium 8-GPU pilot, and the complete roughly 48.9K training split for the
full campaign. Check available disk before full preprocessing.

## 5. Required gates

### Gate A: public inference parity

Run:

```bash
bash examples/train/rvm_h3/03_public_inference_smoke.sh
```

Archive prompt, seed, Git SHA/tree, model digest, scheduler, VSA configuration,
stage timings, video stream, and audio stream. Fixed repeats must be identical
under the strict profile.

### Gate B: preflight

Run:

```bash
bash examples/train/rvm_h3/03_preflight_1gpu.sh
```

It must pass compilation, focused unit tests, YAML invariants, all reward-model
loads, synthetic reward checks, and a real H3 dry-run config build.

### Gate C: one-GPU optimizer smoke

Run:

```bash
bash examples/train/rvm_h3/04_run_1gpu_smoke.sh
```

This compact test must show:

- finite rollout rewards and global reward std;
- finite continuous training times spanning more than one value;
- nonzero signed velocity gradients;
- FP32 LoRA masters and BF16 compute;
- finite Adam update;
- checkpoint save;
- validation media;
- no source-tree mutation.

This is not a quality result.

### Gate D: exact 8-GPU topology/resume/export

Run:

```bash
bash examples/train/rvm_h3/05_run_8gpu_topology_smoke.sh
```

The gate uses:

```text
8 H100s
SP4 x DP2
K=8
4 global prompt groups
2 prompts per DP replica
16 local videos per replica
one update, checkpoint, resume to update two, export LoRA
```

Verify:

- SP ranks share prompt/noise and reward scalars;
- DP replicas receive different prompts;
- only SP leaders contribute to global reward sufficient statistics;
- the globally logged mean/std agree on every rank;
- checkpoint 1 resumes to checkpoint 2;
- optimizer, scheduler, and RNG state advance correctly;
- exported LoRA contains every expected A/B tensor and scaling metadata;
- fixed-seed checkpoint and exported-adapter inference agree numerically.

### Gate E: LR sweep

Run:

```bash
bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
```

Default bracket:

```text
5e-6
1e-5
2e-5
```

The Wan paper's 5e-5 can be added explicitly after lower rates are stable. Use
the same prompt bank, K, prompt groups, seeds, reward stack, anchor setting, and
validation prompts for every LR.

Select the largest LR satisfying:

- held-out aggregate improves;
- TA and MQ do not materially regress;
- gradient clipping is uncommon rather than persistent;
- reward global std stays finite and nontrivial;
- no static, repeated-frame, flicker, oversaturation, or camera-motion exploit;
- audio remains present and coherent.

### Gate F: anchor sweep

Run:

```bash
RVM_SELECTED_LR=<winner> \
  bash examples/train/rvm_h3/06_run_8gpu_anchor_sweep.sh
```

Compare only:

```text
exact:        beta_video=0,    beta_audio=0
 audio-safe:  beta_video=0,    beta_audio=1e-3
full-anchor:  beta_video=1e-3, beta_audio=1e-3
```

Exact RVM is the paper reference. Choose an anchor only when measured
preservation gains outweigh reward-learning loss.

### Gate G: medium scale-up

Prepare at least 4,096 prompts and run:

```bash
RVM_SELECTED_LR=<winner> \
RVM_SCALEUP_CONFIG=<winning-config> \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Default budget:

```text
50 optimizer updates
25 fresh rollout collections
8 prompts per collection
K=8
1,600 rewarded endpoints
32 fixed validation prompts every 5% of optimizer progress
```

This is the first run intended to establish a learning trend.

### Gate H: full campaign

Only after Gate G wins on held-out and qualitative evaluation:

```bash
RVM_FULL_APPROVED=1 \
RVM_SELECTED_LR=<winner> \
RVM_VIDEO_ANCHOR_BETA=<winner> \
RVM_AUDIO_ANCHOR_BETA=<winner> \
  bash examples/train/rvm_h3/07_run_8gpu_full.sh
```

Full budget:

```text
90 rollout collections
32 prompts per collection
K=8
2 updates per collection
180 optimizer updates
23,040 rewarded endpoints
100 fixed validation prompts every 9 steps
```

## 6. Evaluation policy

Evaluation interval is:

```text
ceil(0.05 * max_train_steps)
```

Use the same prompt indices and seeds at baseline and every checkpoint. Report
training rollout reward separately from held-out reward.

At minimum inspect:

```text
validation/reward/avg
validation/reward/videoalign_ta
validation/reward/videoalign_mq
validation/reward/hpsv3_general
validation/reward/hpsv3_percentile
validation/reward/dynamic_tracking
validation/reward/dynamic_tracking_raw
validation/reward/dynamic_tracking_saturation
rvm/reward_global_std
rvm/group_reward_std_mean
rvm/zero_std_group_ratio
rvm/advantage_abs_mean
rvm/advantage_clip_ratio
rvm/grad_norm
rvm/grad_clipped
rvm/training_timestep_mean/min/max
```

Do not automatically choose the last checkpoint. Use paired held-out deltas,
confidence intervals when enough prompts exist, full-video inspection, and an
independent benchmark or human comparison.

Stop when:

- training reward rises but fixed held-out reward falls persistently;
- TA/MQ decline beyond ordinary prompt-sample noise;
- clipping occurs on most updates;
- reward global std collapses or becomes dominated by outliers;
- DT saturates broadly while subject motion does not improve;
- static/repeated/flicker failure rises;
- audio disappears, corrupts, or desynchronizes;
- any loss, reward, gradient, or parameter becomes non-finite.

## 7. Runtime recovery

### CUDA OOM during rollout/reward

In order:

1. confirm only SP leaders hold reward models;
2. set `FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE=1`;
3. ensure VAE/reward tensors are released before NCCL collectives;
4. reduce prompt groups per collection while preserving K=8;
5. use SP8 x DP1 on 8x80GB only as a documented topology fallback;
6. reduce LoRA rank 128 to 64 only after the above;
7. do not change rollout steps, CFG, VSA, or production geometry.

### OOM during backward

Confirm only LoRA parameters require gradients, full activation checkpointing
is active, VSA tile buffer is disabled, no decoded/reward tensors remain, and
LoRA masters—not the frozen backbone—are FP32.

### Reward model failure

- VideoAlign: verify pinned source/checkpoint, Transformers-5 compatibility
  bridge, BF16 reward model, 8-FPS MP4s, grayscale MQ and color TA.
- HPSv3: verify `hpsv3==1.0.0`, correct trained reward head, frame chunking, and
  mean/top-30% aggregation.
- RAFT: verify pretrained weights and moving-over-static synthetic preflight.
  A random or unavailable RAFT model invalidates the run.

### Zero or tiny global reward std

Check that seeds differ, each scorer returns one value per candidate, rewards
are not cached across candidates, and the prompt bank is not duplicated. Do not
artificially re-normalize each group to unit variance.

### High clipping rate

First verify reward normalization and gradient sign. Then reduce LR one bracket.
Do not hide a broken update by only lowering the clip threshold.

### Static/camera-motion exploit

Inspect full videos and compare clipped DT, raw DT, and saturation. A high raw
flow score can come from camera movement or flicker. Keep DT as a guardrail; do
not blindly raise its weight.

### Audio regression

Confirm audio's reward coefficient is zero, audio has an independent reduction,
and the reference prediction is evaluated at the same noisy audio state and
continuous base time. Then test the `1e-3` audio anchor. Do not train audio from
VideoAlign/HPS/RAFT.

### Resume/export mismatch

Resume with the same Git tree, config, topology, prompt digest, reward versions,
and output path. Regenerate the rollout buffer if interruption occurred between
attached updates. Compare fixed-seed inference from the DCP checkpoint and the
exported LoRA before continuing.

## 8. Reproducibility and reporting

Every reported run must record:

- Git `HEAD` and `HEAD^{tree}`;
- clean tracked source;
- model and prompt-source identifiers/digests;
- actual GPU names and memory;
- full command and config overrides;
- package versions;
- reward checkpoint/source versions;
- prompt counts and seeds;
- topology, K, rollout collections, endpoints, optimizer updates;
- all checkpoints, validation metrics, and media;
- every runtime fix as a committed change followed by a fresh smoke test.

Never describe a run as improved because its final on-policy training batch had a
larger reward than baseline validation. Only paired fixed-prompt evaluation and
inspected outputs support a quality claim.
