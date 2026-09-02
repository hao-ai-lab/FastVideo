# Physion-aligned MJ-VIDEO reward implementation plan

## Objective

Add a second, selectable reward profile on top of the existing paper-faithful
FastH3 RVM implementation:

```text
R_physion =
    0.30 * z(VideoAlign text alignment)
  + 0.40 * z(MJ-VIDEO Coherence & Consistency)
  + 0.25 * z(MJ-VIDEO Fineness)
  + 0.05 * z(RAFT Dynamic Tracking)
```

The RVM optimization algorithm must remain unchanged:

1. Generate `K` endpoints with the released four-forward FastH3 VSA policy.
2. Combine reward components into one scalar for each endpoint.
3. Center the scalar reward inside each prompt group.
4. Divide by one population standard deviation over the whole rollout
   collection, including all DP replicas.
5. Apply scale `0.1` and the existing signed clipping.
6. Sample the analytic RVM regression time with `t ~ Uniform(0, 1)`.
7. Regress the native velocity target `epsilon - x0` through the existing
   detached-target surrogate.

This change is therefore a **reward-profile addition**, not a new RL loss.
The original RVM reward profile remains available and must be switchable by
choosing a YAML config.

## Non-goals

- Do not modify the four-step FastH3 rollout, CFG policy, VSA sparsity, model
  geometry, LoRA targets, velocity target, advantage equation, or optimizer
  semantics.
- Do not add PhyJudge, VideoScore2, VisionReward, or a new model-generated
  reward in this change. They remain checkpoint-evaluation candidates.
- Do not normalize each prompt group to unit variance.
- Do not estimate calibration statistics from the live policy during training.
  That would make reward semantics drift over time.
- Do not hide MJ-VIDEO load or compatibility failures by silently falling back
  to another reward model.

## Source inventory and pinned references

### 1. Reward-based Velocity Matching

- Paper: `Scaling Reinforcement Learning for Diffusion Models via Velocity
  Matching`, arXiv:2608.23664.
- Existing implementation source of truth:
  `fastvideo/train/methods/rl/rvm_faithful.py`.
- Required invariants:
  - endpoint-only on-policy sampling;
  - prompt-relative reward centering;
  - rollout-global reward standard deviation;
  - signed scale `0.1`;
  - continuous analytic regression time;
  - native flow target `epsilon - x0`.

No RVM loss code should be forked for the new reward profile.

### 2. MJ-VIDEO paper and official code

- Paper: `MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences
  in Video Generation`, arXiv:2502.01719, NeurIPS 2025 Spotlight.
- Official repository: `aiming-lab/MJ-Video`.
- Pinned source commit:
  `cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a`.
- Official checkpoint: `MJ-Bench/MJ-VIDEO-2B`.
- Pinned checkpoint revision:
  `5d32c2416bf5ffb9331a175890744e73defb54c4`.
- Primary implementation references:
  - `scripts/model/moe_reward.py`;
  - `scripts/model/internvl2/`;
  - `scripts/data_processor/data.py`;
  - `scripts/eval/eval_genai_mjvideo.py`.

Exact official inference settings to preserve:

```text
base model: OpenGVLab/InternVL2-2B
video frames: 8 uniformly spaced segments
input size: 448
max_num image tiles: 1
num_objectives: 28
num_aspects: 5
aspect2criteria:
  0: [0, 1, 2, 3, 4]
  1: [5, 6, 7, 8, 9, 10]
  2: [11, 12, 13, 14, 15]
  3: [16, 17, 18, 19, 20, 21, 22]
  4: [23, 24, 25, 26, 27]
gating_temperature: 1.0
gating_hidden_dim: 1024
gating_n_hidden: 3
inference dtype: BF16
```

Aspect mapping from the paper/code:

```text
0 Alignment
1 Safety
2 Fineness
3 Coherence & Consistency
4 Bias & Fairness
```

The requested reward components are therefore:

```text
mjvideo_fineness = output.aspect_scores[:, 2]
mjvideo_cc       = output.aspect_scores[:, 3]
```

### 3. VideoAlign and RAFT

Reuse the already-pinned implementations and checkpoints in this branch:

- VideoAlign TA: unchanged preprocessing and reward semantics.
- Dynamic Tracking: unchanged clipped RVM reward; retain raw-flow and
  saturation diagnostics.

The Physion profile lowers DT to `0.05`; it remains an anti-static guardrail,
not the primary quality signal.

### 4. Fixed robust reward calibration

The four reward models have unrelated numeric scales. Implement a fixed
baseline calibration rather than using arbitrary raw units:

```text
z_j(r) = (r - center_j) / max(scale_j, eps)
center_j = median_j
scale_j = 1.4826 * MAD_j
```

If MAD is degenerate, use the baseline population standard deviation. Fail if
both scales are degenerate unless the user explicitly overrides the component.
Optionally clip calibrated component values using a fixed config value.

Calibration must be computed once from a fixed bank of released FastH3 outputs
and saved as a versioned JSON artifact. It is separate from RVM advantage
normalization:

```text
raw component -> fixed z calibration -> weighted reward scalar
             -> per-prompt center / rollout-global std -> RVM coefficient
```

## Implementation tasks and commit sequence

### Phase 0 — plan and provenance

- [x] Inspect the current RVM reward builder, aggregation path, validation
  artifacts, data scripts, and tests.
- [x] Inspect the MJ-VIDEO paper, official checkpoint, model architecture,
  aspect mapping, frame preprocessing, and evaluation script.
- [x] Record exact pinned source/checkpoint revisions and implementation plan.
- [ ] Create an implementation progress report updated after every phase.

Acceptance: this plan and the initial progress report are committed before code
changes.

### Phase 1 — fixed calibration infrastructure

Implement:

- `RewardCalibrationEntry` and calibration-artifact parser;
- `CalibratedRewardScorer`, preserving raw values as diagnostics;
- top-level `reward_fn.calibration` support in the reward builder;
- deterministic reward-output-key discovery so distributed RVM broadcasts all
  scorer diagnostics without hard-coded reward names.

Tests:

- exact z-score application;
- MAD fallback and invalid-scale failure;
- required/missing calibration behavior;
- diagnostics do not enter the weighted aggregate;
- output-key discovery is identical on all ranks.

Acceptance: existing reward profiles produce identical aggregate values when no
calibration block is configured.

### Phase 2 — MJ-VIDEO runtime adapter

Implement `fastvideo/train/methods/rl/rewards/mj_video.py`:

- load the pinned official source tree dynamically from a configured/local path;
- verify its Git commit unless an explicit development override is enabled;
- load the pinned `MJ-Bench/MJ-VIDEO-2B` checkpoint strictly;
- reproduce official 8-frame, 448-pixel, ImageNet-normalized preprocessing in
  memory;
- reproduce the official InternVL prompt construction;
- expose `mjvideo_fineness` and `mjvideo_cc` scorers;
- share one model/runtime and one forward result between both aspects;
- support bounded inference chunks and SP-leader-only loading;
- fail loudly on Transformers/runtime incompatibility.

Tests use a fake runtime/model and verify:

- exact frame indices;
- aspect indices 2 and 3;
- shared forward cache;
- output shapes and finite checks;
- source/checkpoint revision validation.

Acceptance: a dedicated GPU preflight loads the real checkpoint and returns two
finite, non-identical aspect tensors for deterministic videos.

### Phase 3 — assets and calibration workflow

Extend provider-independent scripts:

- add MJ-VIDEO runtime/model/calibration paths to `common.sh`;
- pin and download the official code/checkpoint in `01_download_models.sh`;
- save a complete validation JSONL manifest containing prompt, video path, and
  raw reward components;
- add a calibration CLI that scores a fixed released-FastH3 video bank and
  writes robust median/MAD statistics plus provenance;
- add `04_calibrate_physion_mj_rewards.sh`.

Acceptance:

- calibration generation is deterministic for fixed videos/prompts;
- artifact records model/source revisions, sample count, component statistics,
  and input-manifest digest;
- training refuses to use the Physion profile when the required calibration is
  absent or incomplete.

### Phase 4 — selectable reward profile

Add a config that changes only the reward profile:

```text
0.30 VideoAlign TA (calibrated)
0.40 MJ-VIDEO C&C (calibrated)
0.25 MJ-VIDEO Fineness (calibrated)
0.05 Dynamic Tracking (calibrated)
```

Keep identical:

- `RVMFaithfulMethod` / `RVMWithLocalMetricsMethod`;
- four-step behavior rollout;
- continuous RVM regression time;
- LoRA rank/targets;
- geometry and VSA settings;
- optimizer and advantage semantics.

Add a matched reward-profile sweep script comparing:

1. the original published RVM reward profile;
2. the Physion/MJ-VIDEO profile.

Acceptance: tests prove the two configs differ only in reward configuration,
artifact paths, and run/output names.

### Phase 5 — preflight, docs, and final audit

- extend static preflight to cover the MJ adapter and calibration code;
- add a real-MJ reward preflight mode;
- document installation, calibration, profile switching, memory expectations,
  failure modes, and custom-node launch commands;
- update this progress report with exact commands and honest validation status;
- run formatting, focused tests, config parsing, shell syntax, and diff checks;
- run GPU preflight separately before any long training claim.

Acceptance: all CPU/static tests pass; GPU status is reported explicitly rather
than inferred from code completion.

## Experiment switch

Original RVM reward profile:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Physion/MJ-VIDEO reward profile after calibration:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

The matched profile-sweep script will set the corresponding output directories
and run names while preserving the same prompt split, seeds, LR, topology, and
RVM loss.

## Go/no-go criteria

Do not run the long Physion profile campaign until:

1. the real MJ-VIDEO checkpoint loads under the FastVideo environment;
2. exact aspect mapping and fixed calibration are verified;
3. baseline and trained validation use the same prompts and seeds;
4. no reward component is constant or non-finite;
5. MJ C&C/Fineness correlate in the expected direction on a small manually
   inspected pair bank;
6. the 1/4-GPU integration test and 8-GPU topology test complete;
7. full-video quality, prompt adherence, motion, and audio are inspected rather
   than selecting solely by the optimized scalar.
