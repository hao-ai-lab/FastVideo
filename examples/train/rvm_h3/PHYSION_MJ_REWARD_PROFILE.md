# FastH3 RVM with the Physion/MJ-VIDEO reward profile

This guide adds a second reward profile to the existing paper-faithful FastH3
RVM implementation. It does **not** introduce another RL loss.

## Reward switch

The published RVM reproduction remains:

```text
1.5 VideoAlign TA
1.0 VideoAlign MQ
0.1 HPSv3 General
0.1 HPSv3 Percentile
0.7 RAFT Dynamic Tracking
```

The alternate Physion-aligned profile is:

```text
0.30 z(VideoAlign TA)
0.40 z(MJ-VIDEO Coherence & Consistency)
0.25 z(MJ-VIDEO Fineness)
0.05 z(RAFT Dynamic Tracking)
```

`z` is a fixed robust calibration derived from released FastH3 outputs:

```text
center = median(baseline scores)
scale = 1.4826 * median(abs(score - center))
z(score) = (score - center) / scale
```

Population standard deviation is used only when MAD is degenerate. A constant
component fails calibration unless the operator explicitly supplies and records
a fallback scale.

This calibration is upstream of the unchanged RVM advantage equation:

```text
calibrated weighted reward
  -> subtract each K-sample prompt-group mean
  -> divide by one rollout-global population standard deviation
  -> multiply by 0.1 and clip
  -> signed velocity-matching update
```

## Source alignment

The MJ adapter is pinned to:

```text
paper:               arXiv:2502.01719
source repository:   aiming-lab/MJ-Video
source commit:       cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a
reward checkpoint:   MJ-Bench/MJ-VIDEO-2B
checkpoint revision: 5d32c2416bf5ffb9331a175890744e73defb54c4
base model:          OpenGVLab/InternVL2-2B
base revision:       e4f6747
```

The adapter follows the official implementation:

- 28 criteria and five aspects;
- Fineness is aspect index `2`;
- Coherence & Consistency is aspect index `3`;
- eight endpoint-exclusive uniform frames;
- 448x448 bicubic inputs;
- one image tile per frame;
- ImageNet normalization;
- BF16 model inference;
- gating temperature `1.0`, hidden dimension `1024`, three hidden layers;
- strict checkpoint loading.

C&C and Fineness share one MJ-VIDEO model and one forward result for the same
video batch. There is no fallback reward if the pinned source or checkpoint is
incompatible.

## Setup

First complete the normal FastH3 RVM setup:

```bash
bash examples/train/rvm_h3/00_create_conda_env.sh
bash examples/train/rvm_h3/01_download_models.sh
RVM_MAX_TRAIN_PROMPTS=4096 \
  bash examples/train/rvm_h3/02_prepare_dataset.sh
```

Then fetch the alternate reward assets:

```bash
bash examples/train/rvm_h3/01_download_mj_video.sh
```

Run the real-checkpoint preflight:

```bash
bash examples/train/rvm_h3/03_preflight_mj_video.sh
```

This must strictly load the pinned 2B reward model and produce finite, distinct
C&C and Fineness scores from one shared forward per input video. The official
MJ-VIDEO code predates FastVideo's Transformers 5 runtime; treat any
compatibility error as a committed implementation bug, not a reason to silently
change the reward.

## Build the fixed calibration artifact

Generate or reuse up to 100 deterministic released-FastH3 videos sampled from
the **training prompt split**, then score all four raw components:

```bash
export NUM_GPUS=4
export RVM_SP_SIZE=4
bash examples/train/rvm_h3/04_calibrate_physion_mj_rewards.sh
```

Outputs:

```text
artifacts/rvm_h3/rewards/physion_mj_calibration.json
artifacts/rvm_h3/rewards/physion_mj_calibration.scores.jsonl
outputs/rvm_h3/calibration_bank/validation/step-000000/*.mp4
```

The JSON artifact records source/model revisions, Git head, prompt and video
hashes, sample count, quantiles, median, MAD, standard deviation, and the chosen
scale for every component.

Do not calibrate from a partially trained policy. The calibration distribution
is the released FastH3 baseline and remains fixed for every compared run. The
calibration bank uses `train_h3.txt` and `data/train`; `eval_h3.txt` and
`data/eval` remain untouched for model selection and final reporting.

## Run either profile

Original published RVM reward profile:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Physion/MJ reward profile:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

The second config changes only:

- reward names and weights;
- fixed calibration artifact/options;
- method class used to synchronize scorer-declared diagnostics;
- output directory and run name.

Tests deep-compare the remaining parsed YAML structure against exact RVM.

## Matched reward-profile experiment

On the custom 8-H100 node:

```bash
export NUM_GPUS=8
export RVM_SP_SIZE=4
export RVM_SELECTED_LR=1e-5

bash examples/train/rvm_h3/06_run_8gpu_reward_profile_sweep.sh
```

Defaults:

```text
SP4 x DP2
K = 8
8 prompt groups per rollout collection
20 optimizer updates
32 fixed held-out prompts
validation at step 0, midpoint, and final
```

Both arms use the same initialization, prompt order, seeds, learning rate,
rollout geometry, VSA policy, continuous RVM time, LoRA, optimizer budget, and
held-out prompts.

Select using:

- paired per-prompt held-out changes;
- raw and calibrated reward components;
- full-video inspection;
- prompt adherence and motion checks;
- audio preservation;
- an independent evaluator or blinded human comparison.

Do not select a profile from its own final training-rollout aggregate alone.

## Logged Physion-profile values

The scorer reports calibrated values used in the weighted scalar and raw values
for interpretation:

```text
reward/videoalign_ta
reward/videoalign_ta_unnormalized
reward/mjvideo_cc
reward/mjvideo_cc_unnormalized
reward/mjvideo_fineness
reward/mjvideo_fineness_unnormalized
reward/dynamic_tracking
reward/dynamic_tracking_unnormalized
reward/dynamic_tracking_raw
reward/dynamic_tracking_saturation
reward/avg
```

RVM also continues to log rollout-global reward standard deviation, prompt-group
spread, zero-spread groups, advantage magnitude/clipping, gradient clipping, and
continuous training-time statistics.

## Memory and topology

Only sequence-parallel leaders instantiate reward models. MJ-VIDEO-2B is shared
between C&C and Fineness. Reward models run sequentially after VAE decode, while
H3 rollout activations are absent.

If a reward leader OOMs:

1. verify only SP leaders loaded reward models;
2. keep MJ `batch_size: 1`;
3. keep `FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE=1`;
4. reduce prompt groups per collection, not `K` first;
5. do not change the FastH3 rollout, CFG, VSA, or RVM loss to clear reward OOM;
6. use a separate reward-service process only as a later measured optimization.

Production 8/16-H100 runs use the ordinary custom-node scripts. Modal remains a
thin one-/four-GPU integration-test transport and is not part of this reward
implementation.

## Validation boundary

Static code and fake-runtime tests do not prove that the old official InternVL
implementation loads under the current FastVideo environment. The sequence is:

```text
real MJ preflight
-> fixed training-split baseline calibration
-> one/four-GPU profile smoke
-> 8-GPU matched profile sweep
-> only then a longer run
```

Record failures and fixes in `MJ_VIDEO_REWARD_PROGRESS.md` and commit each fix
before rerunning.
