# MJ-VIDEO reward implementation progress

This report records the incremental implementation of a selectable
Physion-aligned reward profile for FastH3 RVM. The alternate profile changes the
reward scalar only; the existing paper-faithful RVM rollout, advantage, velocity
loss, optimizer, LoRA, and FastH3 inference contract remain unchanged.

## Fixed objective

```text
R_physion =
    0.30 * z(VideoAlign text alignment)
  + 0.40 * z(MJ-VIDEO Coherence & Consistency)
  + 0.25 * z(MJ-VIDEO Fineness)
  + 0.05 * z(RAFT Dynamic Tracking)
```

For component `j`, `z` is fixed from released-FastH3 calibration videos drawn
from the **training prompt split**:

```text
center_j = median(baseline reward_j)
scale_j  = 1.4826 * MAD(baseline reward_j)
z_j(r)   = (r - center_j) / scale_j
```

Population standard deviation is used only when MAD is degenerate. A component
that is constant under both estimators fails calibration unless the operator
provides an explicit audited fallback scale.

The existing RVM update remains:

```text
calibrated weighted reward
  -> subtract the K-sample prompt-group mean
  -> divide by one rollout-global population standard deviation
  -> multiply by 0.1 and clip
  -> continuous-time velocity matching on epsilon - x0
```

The calibration set and model-selection set are disjoint:

```text
calibration: train_h3.txt / data/train
held-out validation: eval_h3.txt / data/eval
```

## Pinned references

### RVM

- Paper: `Scaling Reinforcement Learning for Diffusion Models via Velocity
  Matching`, arXiv:2608.23664.
- FastVideo source of truth: `fastvideo/train/methods/rl/rvm_faithful.py`.
- Preserved behavior:
  - four-step FastH3 VSA endpoint sampling;
  - prompt-relative centering and rollout-global reward scale;
  - `t ~ Uniform(0,1)` analytic RVM regression state;
  - signed coefficient scale `0.1` and clipping;
  - detached-target surrogate with native target `epsilon - x0`.

### MJ-VIDEO

- Paper: `MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences
  in Video Generation`, arXiv:2502.01719, NeurIPS 2025 Spotlight.
- Official source: `aiming-lab/MJ-Video`.
- Source revision: `cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a`.
- Reward checkpoint: `MJ-Bench/MJ-VIDEO-2B`.
- Checkpoint revision: `5d32c2416bf5ffb9331a175890744e73defb54c4`.
- Base model: `OpenGVLab/InternVL2-2B` revision `e4f6747`.
- Official references:
  - `scripts/model/moe_reward.py`;
  - `scripts/model/internvl2/`;
  - `scripts/data_processor/data.py`;
  - `scripts/eval/eval_genai_mjvideo.py`.

Source-derived settings preserved:

```text
28 criteria / 5 aspects
Fineness aspect index: 2
Coherence & Consistency aspect index: 3
8 endpoint-exclusive uniformly sampled frames
448x448 bicubic inputs
1 image tile per frame
ImageNet normalization
BF16 inference
MoE gating temperature 1.0
MoE hidden dimension 1024
3 hidden gating layers
strict safetensors loading
```

## Phase 0 — plan and source audit

**Commit:** `55d60aec720be847daea4966bc0e98e57eda7abf`

Added the implementation plan and this progress report before any runtime code.
The plan records exact source/checkpoint revisions, non-goals, task ordering,
acceptance criteria, and the GPU validation boundary.

## Phase 1 — fixed calibration and reward-output contracts

**Implementation commit:** `9848405f27400781fa9ba8b8478ec975034fa97b`

Implemented:

- versioned calibration schema in
  `fastvideo/train/methods/rl/rewards/calibration.py`;
- strict finite-center and positive-scale checks;
- required component coverage and optional symmetric z clipping;
- `CalibratedRewardScorer`, preserving raw values as diagnostics;
- deterministic `MultiRewardScorer.output_keys` and exact diagnostic contract;
- `RVMRewardProfileMethod`, inheriting the existing RVM method and adding only
  SP synchronization of scorer-declared output keys;
- unchanged aggregate behavior when calibration is absent.

Tests cover affine calibration, raw diagnostics, invalid scales, missing
components, JSON metadata, deterministic output ordering, and unchanged
uncalibrated weighted sums.

## Phase 2 — source-aligned MJ-VIDEO adapter

**Implementation commit:** `c3445e3211764dfe3281d87d661c2c4a0eada2a8`

Implemented:

- `fastvideo/train/methods/rl/rewards/mj_video.py`;
- strict source/model/base revision verification;
- dynamic import of the pinned official model code;
- exact 28-criterion/five-aspect configuration;
- exact eight-frame, 448-pixel, one-tile preprocessing;
- `mjvideo_fineness` from aspect index `2`;
- `mjvideo_cc` from aspect index `3`;
- one process-local model/runtime shared by both scorers;
- one forward result reused for both aspects on the same media/prompt batch;
- bounded inference batches, finite checks, strict checkpoint loading, and no
  fallback reward.

A narrow compatibility module restores the removed, unused
`LLAMA_INPUTS_DOCSTRING` imported by the pinned official source under
Transformers 5. It changes no weights or forward math. Functional
incompatibilities remain hard failures in the real-checkpoint preflight.

Tests cover exact aspect/criterion mapping, official frame indices, shared
forward caching, cache invalidation, and rejection of source-drift settings.

## Phase 3 — assets and fixed baseline calibration

**Implementation commit:** `1d2a3077eedc216c5a790b1f94757f3d895bd671`

Implemented:

- provider-independent MJ paths in `common.sh`;
- immutable source/model/base revisions in `01_download_mj_video.sh`;
- `h3_rvm_calibration_bank.yaml` for deterministic released-FastH3 videos at
  full `480x832x124` geometry with the exact four-step VSA policy;
- `calibrate_reward_profile.py` for raw component scoring, median/MAD
  statistics, standard-deviation fallback, JSON/JSONL output, and provenance;
- `04_calibrate_physion_mj_rewards.sh` for calibration generation and artifact
  validation;
- `preflight_mj_video.py` and `03_preflight_mj_video.sh` for strict real-model
  load/forward validation.

Robustness and split-isolation follow-ups:

- `babd41ae10587e7cafb2c34a15537808475d57db` — handle a missing calibration
  directory safely under `set -euo pipefail`;
- `809552c972003e93bde3b44481c2ec2ebe936043` — make MJ revisions immutable and
  verify revision markers after download;
- `0c1e4729f2b8b63e331f7f4a0c865526844f51be` — calibrate against
  `train_h3.txt`, not `eval_h3.txt`;
- `6ae0e70c9775220f44453c0d357148059f15c491` — generate the baseline bank from
  `data/train`, keeping `data/eval` held out;
- `a7e7dc882ddade002ace89d16962a7765f510acd` — enforce calibration/evaluation
  split separation in config tests;
- `54016212373bbb7cc51bd8470e32788894c20557` — document the split contract.

The original RVM setup remains independent. MJ assets are downloaded only when
the alternate profile is requested.

## Phase 4 — selectable profile and matched comparison

**Implementation commit:** `95fe9106706782802ef0e5d24ba170add6126e91`

Implemented:

- `rvm_h3_8gpu_physion_mj.yaml` with exactly the requested calibrated
  `0.30/0.40/0.25/0.05` weights;
- required fixed calibration artifact;
- unchanged exact unanchored RVM, four-step behavior policy, continuous
  regression time, rank-128 attention LoRA, optimizer, geometry, and VSA policy;
- `06_run_8gpu_reward_profile_sweep.sh` for a matched original-versus-Physion
  comparison;
- config deep-comparison tests proving that the profiles differ only in reward
  configuration, the diagnostic-synchronization subclass, and run/output names.

Profile switch:

```bash
# Published RVM reward recipe.
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh

# Physion/MJ reward recipe.
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh

# Matched profile comparison.
bash examples/train/rvm_h3/06_run_8gpu_reward_profile_sweep.sh
```

## Phase 5 — documentation and custom-node scale-up

Key commits:

- `a9ea0b2f90b00b6ed7f66a853bd2a94fdaf9f156` — profile guide, expanded
  preflight, and Transformers-5 import compatibility;
- `c22a503a808fd2370ff469a828aea4e7571ea9be` — 8/16-H100 custom-node topology,
  pilot, and full-campaign scripts;
- `c79e74b8b115f20d058b825540ab64ddd2ed2abc` — distinguish compact one-H100
  smoke geometry from production geometry;
- `7bb6dc9c3e92a0931c3f872ab4cc23c8738dde3d` — audit exact pinned MJ source
  import under the current dependency family.

The custom-node scripts support:

```text
8 H100s:  SP4 x DP2
16 H100s: SP4 x DP4
```

Modal remains a thin one-/four-GPU test transport and contains no dataset
processing, reward/model setup, hyperparameters, or training implementation.

## Final static audit

Temporary GitHub Actions audit:

```text
Git SHA: 54016212373bbb7cc51bd8470e32788894c20557
Run ID: 33619034023
Job ID: 100211511291
Result: success
```

Passed:

- compilation of every new reward/calibration/profile entry point;
- import of the exact pinned MJ source commit under Transformers `5.16.1` with
  the narrow compatibility shim;
- shell syntax for asset download, real-model preflight, calibration, 8/16-GPU
  topology, profile sweep, pilot, and full-run scripts;
- YAML parsing for calibration-bank and Physion-profile configs;
- **26 focused CPU tests** covering calibration, CLI statistics, MJ
  preprocessing/aspects/cache, diagnostics, config equivalence, and
  calibration/eval split separation;
- `git diff --check`.

This validates source import, static structure, equations, configuration
contracts, and provider-independent orchestration. It does **not** load the
4.43-GB MJ reward checkpoint or execute H3/MJ on a GPU.

## Required execution order

```bash
# Existing FastH3/RVM setup.
bash examples/train/rvm_h3/00_create_conda_env.sh
bash examples/train/rvm_h3/01_download_models.sh
RVM_MAX_TRAIN_PROMPTS=4096 \
  bash examples/train/rvm_h3/02_prepare_dataset.sh

# Alternate reward setup and exact-model gate.
bash examples/train/rvm_h3/01_download_mj_video.sh
bash examples/train/rvm_h3/03_preflight_mj_video.sh

# Fixed released-FastH3 calibration from training prompts.
export NUM_GPUS=4
export RVM_SP_SIZE=4
bash examples/train/rvm_h3/04_calibrate_physion_mj_rewards.sh

# Matched custom-node comparison.
export NUM_GPUS=8
export RVM_SP_SIZE=4
bash examples/train/rvm_h3/06_run_8gpu_reward_profile_sweep.sh
```

For 16 H100s, set `NUM_GPUS=16` and first run
`05_run_8gpu_topology_smoke.sh`.

## Honest validation boundary

No new H3 or MJ-VIDEO GPU execution has occurred for this extension in the
current environment. Existing one-/four-H100 results validate the pre-existing
H3 RVM runtime only. Before any quality claim, the new path must pass:

1. exact pinned MJ checkpoint load and real forward;
2. fixed released-FastH3 training-split calibration-bank generation;
3. one-/four-GPU profile integration smoke;
4. 8-H100 SP4xDP2 topology/resume/export gate;
5. matched original-RVM-versus-Physion profile sweep;
6. paired held-out metrics plus full-video and audio inspection.

Any runtime compatibility fix must be committed and re-tested. Do not patch the
Modal wrapper or custom node in place, and do not replace MJ-VIDEO with another
reward silently.
