# MJ-VIDEO reward implementation progress

This report tracks the incremental implementation of a selectable
Physion-aligned reward profile for FastH3 RVM. Every phase is committed before
the next begins so partial work is not lost.

## Fixed scope

The alternate scalar reward is:

```text
0.30 * z(VideoAlign TA)
+ 0.40 * z(MJ-VIDEO Coherence & Consistency)
+ 0.25 * z(MJ-VIDEO Fineness)
+ 0.05 * z(RAFT Dynamic Tracking)
```

The RVM loss, endpoint sampling, advantage construction, continuous regression
time, FastH3 rollout, LoRA target modules, and optimizer semantics are not being
replaced.

## Phase 0 — source audit and implementation plan

**Status:** complete.

**Commit:** `55d60aec720be847daea4966bc0e98e57eda7abf`

### Current FastVideo implementation inspected

- `fastvideo/train/methods/rl/rvm_faithful.py`
  - current paper-faithful endpoint RVM;
  - per-prompt reward centering;
  - rollout-global standard deviation;
  - continuous `t ~ Uniform(0,1)` analytic regression state;
  - signed detached-target velocity update.
- `fastvideo/train/methods/rl/rewards/__init__.py`
  - reward construction and per-reward options.
- `fastvideo/train/methods/rl/rewards/media.py`
  - weighted scalar aggregation and diagnostic propagation.
- `fastvideo/train/methods/rl/rewards/{videoalign,dynamic_tracking}.py`
  - existing source-aligned VideoAlign and RAFT implementations.
- H3 reward configs, model download scripts, portable smoke runner, validation
  persistence, and focused RVM tests.

### MJ-VIDEO sources inspected

- Paper: arXiv:2502.01719.
- Official repository: `aiming-lab/MJ-Video` at
  `cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a`.
- Official checkpoint: `MJ-Bench/MJ-VIDEO-2B` at
  `5d32c2416bf5ffb9331a175890744e73defb54c4`.
- `scripts/model/moe_reward.py` for criteria/aspect/overall outputs and MoE
  gating.
- `scripts/model/internvl2/` for the exact InternVL input/model path.
- `scripts/data_processor/data.py` for uniform eight-frame sampling, 448 input,
  one tile per frame, and ImageNet normalization.
- `scripts/eval/eval_genai_mjvideo.py` for the exact 28-criterion, five-aspect
  mapping and inference defaults.

### Exact source-derived decisions

- Use aspect index `2` for Fineness.
- Use aspect index `3` for Coherence & Consistency.
- Sample eight evenly spaced frames with the upstream endpoint-exclusive index
  rule.
- Use BF16 and the official InternVL2-2B reward checkpoint.
- Share one MJ-VIDEO forward between both requested aspect scorers.
- Pin and verify both source code and model revisions.
- Use fixed robust baseline calibration rather than online scale adaptation.
- Fail loudly on runtime/checkpoint incompatibility.

### Risks identified before coding

1. MJ-VIDEO's official implementation targets an older Transformers stack,
   while FastVideo currently uses Transformers 5. A real GPU load/forward test
   is mandatory; the adapter must not silently substitute another model.
2. The official repository does not currently expose the requirements files
   referenced by its README. The integration should rely on FastVideo's existing
   dependencies plus explicit minimal additions, not install an unknown legacy
   environment wholesale.
3. Two separately constructed MJ aspect scorers would otherwise load/execute the
   same 2B model twice. A shared runtime and result cache are required.
4. Raw reward units are not comparable. Fixed calibration artifacts must be
   generated from baseline FastH3 outputs and versioned with source provenance.
5. Distributed RVM must know every raw/diagnostic output key before tensor
   broadcasts. The scorer now declares this contract explicitly.

## Phase 1 — fixed calibration and reward-output contracts

**Status:** implementation complete; GPU-independent tests authored.

**Implementation commit:** `9848405f27400781fa9ba8b8478ec975034fa97b`

### Implemented

- Added `fastvideo/train/methods/rl/rewards/calibration.py`.
- Added versioned calibration schema `1` with strict validation of finite
  centers, positive scales, optional sample counts, required component coverage,
  and optional symmetric clipping.
- Added `CalibratedRewardScorer`:
  - applies only a fixed affine transform `(raw - center) / scale`;
  - preserves each raw score as `<reward>_unnormalized`;
  - preserves diagnostics from the wrapped scorer;
  - does not modify the wrapped reward implementation.
- Extended `build_multi_reward_scorer` with an optional
  `reward_fn.calibration` block using either a versioned JSON artifact or inline
  entries for tests.
- Preserved existing behavior when no calibration is configured: the same raw
  component scores and weighted sum are used.
- Added an explicit `MultiRewardScorer.output_keys` contract. It validates that
  runtime diagnostics exactly match names declared by each scorer.
- Added `RVMRewardProfileMethod`, a thin subclass of the existing
  `RVMWithLocalMetricsMethod`. It synchronizes the scorer-declared key tuple over
  each SP group's CPU process group. It does not override the RVM rollout,
  advantage, velocity loss, optimizer, or validation equations.
- Declared the existing Dynamic Tracking `raw` and `saturation` diagnostics in
  the reward builder so the general output contract includes them.

### Tests added

`fastvideo/tests/train/methods/test_reward_calibration.py` covers:

- exact affine calibration and clipping;
- raw/nested diagnostic propagation;
- required-component failures;
- JSON metadata parsing;
- invalid zero, negative, NaN, and infinite scales;
- deterministic output-key ordering;
- unchanged uncalibrated weighted sums;
- calibrated weighted sums and raw-score logging.

### Why this is separate from RVM normalization

The fixed component calibration makes unrelated reward units comparable:

```text
raw reward -> fixed baseline z-score -> configured weighted sum
```

The existing RVM code then computes the policy update:

```text
weighted sum -> per-prompt centering -> rollout-global std -> signed coefficient
```

The second stage is unchanged. Calibration never estimates live-policy moments,
so the reward target does not drift during training.

## Phase 2 — source-aligned MJ-VIDEO aspect adapter

**Status:** implementation complete; real-checkpoint GPU preflight still pending.

**Implementation commit:** `c3445e3211764dfe3281d87d661c2c4a0eada2a8`

### Implemented

- Added `fastvideo/train/methods/rl/rewards/mj_video.py`.
- Pinned the official source revision and checkpoint revision in code.
- Added strict source/model/base-model revision verification. Model downloads
  must contain `.fastvideo_revision` markers written by the setup script; local
  overrides require an explicit `verify_revision=false` choice.
- Dynamically import the official `moe_reward.py` and InternVL2 implementation
  from a configured MJ-VIDEO checkout rather than copying or silently changing
  upstream model semantics.
- Reproduce the source inference configuration:
  - `OpenGVLab/InternVL2-2B` base architecture;
  - 28 criteria and five aspects;
  - exact aspect-to-criterion mapping;
  - gating temperature `1.0`;
  - hidden dimension `1024`;
  - three hidden gating layers;
  - strict safetensors load;
  - BF16 evaluation.
- Reproduce source video preprocessing in memory:
  - eight uniformly sampled, endpoint-exclusive frames;
  - 448x448 bicubic resize;
  - one tile per frame;
  - ImageNet mean/std;
  - official `FrameN: <image>` prompt construction.
- Added `mjvideo_fineness` from aspect index `2` and `mjvideo_cc` from aspect
  index `3` to the existing reward builder.
- Added one shared process-local runtime cache. When both aspect scorers receive
  the same media object and prompt tuple, only one MJ-VIDEO forward is executed.
- Added configurable bounded batch size, with source-faithful batch size one as
  the default.
- Added an isolated one-process Gloo initialization only when a standalone
  preflight has no distributed process group; the official InternVL code calls
  `torch.distributed.get_rank()` during forward.
- All load/import/shape/non-finite failures are explicit. There is no fallback to
  HPSv3, VideoAlign, or another reward.

### Tests added

`fastvideo/tests/train/methods/test_mj_video_reward.py` covers:

- exact five-aspect mapping;
- exact criteria groups;
- official frame indices for 124-frame and short videos;
- aspect indices `2` and `3`;
- one shared forward for C&C and Fineness;
- cache invalidation for a new media object;
- rejection of non-source values for frame count, input size, tiling, and dtype.

### Important unresolved compatibility gate

The official MJ-VIDEO code was authored against an older Transformers API,
whereas FastVideo currently pins Transformers 5. The adapter intentionally
raises a detailed error if that import or strict model load fails. Any required
compatibility change must be committed and tested; it must not be applied as an
untracked cloud-launcher patch.

## Phase 3 — pinned assets and baseline calibration workflow

**Status:** implementation complete; real asset/preflight/calibration execution
pending.

**Implementation commit:** `1d2a3077eedc216c5a790b1f94757f3d895bd671`

### Implemented

- Extended `common.sh` with provider-independent paths for the official
  MJ-VIDEO source, MJ-VIDEO-2B checkpoint, InternVL2-2B base model, and fixed
  calibration artifact.
- Added `01_download_mj_video.sh`, pinned to the exact source/model/base
  revisions and writing revision markers consumed by the runtime adapter.
- Added `h3_rvm_calibration_bank.yaml` to generate up to 100 fixed released
  FastH3 videos at full geometry and the exact four-step VSA sampler.
- Added `calibrate_reward_profile.py`:
  - decodes fixed baseline MP4s with PyAV;
  - scores raw TA, MJ C&C, MJ Fineness, and DT;
  - uses median and `1.4826 * MAD`, with population-std fallback;
  - refuses constant components unless explicitly audited and overridden;
  - writes a versioned JSON artifact and per-video JSONL;
  - records Git/model/source/input provenance.
- Added `04_calibrate_physion_mj_rewards.sh` to generate or reuse the baseline
  bank and build/validate the artifact.
- Added `preflight_mj_video.py` and `03_preflight_mj_video.sh` for a strict real
  checkpoint load/forward and the focused tests.
- Added `test_reward_calibration_cli.py`.

### Deliberate separation from the original RVM setup

The original `01_download_models.sh` remains unchanged. MJ-VIDEO is downloaded
only when the alternate profile is requested. All calibration and setup logic
lives in ordinary repository scripts; no Modal-only code or runtime patching is
involved.

## Phase 4 — selectable Physion/MJ reward profile

**Status:** implementation complete; matched GPU comparison pending.

**Implementation commit:** `95fe9106706782802ef0e5d24ba170add6126e91`

### Implemented

- Added `rvm_h3_8gpu_physion_mj.yaml` with exactly:

  ```text
  0.30 * z(videoalign_ta)
  0.40 * z(mjvideo_cc)
  0.25 * z(mjvideo_fineness)
  0.05 * z(dynamic_tracking)
  ```

- The profile requires the versioned fixed calibration artifact and clips only
  extreme calibrated component values at absolute z-score five.
- It uses `RVMRewardProfileMethod`, which inherits the existing
  `RVMWithLocalMetricsMethod`; the only added behavior is synchronization of the
  scorer-declared diagnostic key tuple.
- It keeps exact unanchored RVM, the four-step behavior policy, continuous
  analytic regression time, rank-128 attention LoRA, full geometry, VSA 90%,
  optimizer, advantage scale, clipping, validation, and custom-node topology.
- Added `06_run_8gpu_reward_profile_sweep.sh`:
  - compares the original published RVM reward recipe against the Physion/MJ
    profile;
  - holds initialization, prompt order, seeds, LR, K, prompt groups, optimizer
    budget, topology, and held-out evaluation fixed;
  - runs the real MJ preflight first unless explicitly skipped;
  - requires the fixed calibration artifact;
  - evaluates at baseline, midpoint, and final by default.
- Extended config tests to enumerate the new config, verify exact weights and MJ
  settings, and deep-compare it with the original exact-RVM config after removing
  only the reward profile, method-name shim, and run/output names.
- Updated the reward-diagnostic test to declare its deterministic output
  contract explicitly.

### How to switch

Original published RVM rewards:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Physion/MJ rewards:

```bash
RVM_SCALEUP_CONFIG=examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Matched comparison:

```bash
bash examples/train/rvm_h3/06_run_8gpu_reward_profile_sweep.sh
```

### Next phase

Run a clean-clone static audit, fix any import/format/config defects, update the
README/runbook/preflight and PR description, and document the exact GPU commands
and unresolved real-checkpoint boundary.

## Commit log

| Phase | Commit | Summary | Validation |
|---|---|---|---|
| 0 | `55d60aec` | Plan, source inventory, and implementation contract | Source audit |
| 1 | `9848405f` | Fixed calibration, output contracts, distributed profile method | AST parse; tests authored |
| 2 | `c3445e32` | Official MJ-VIDEO adapter, exact preprocessing, shared aspect cache | AST parse; fake-runtime tests authored |
| 3 | `1d2a3077` | Pinned assets, baseline bank, robust calibration, real-model preflight | AST/shell parsing pending final audit |
| 4 | `95fe9106` | Selectable calibrated profile and matched profile sweep | Config invariants authored |

## GPU validation boundary

No new GPU execution has occurred for this MJ-VIDEO extension. Existing H3 RVM
GPU results validate the pre-existing model/rollout/reward path only. The new
MJ-VIDEO adapter and calibrated profile must pass the committed real-checkpoint
preflight, baseline calibration, and matched short training sweep before any
quality claim.
