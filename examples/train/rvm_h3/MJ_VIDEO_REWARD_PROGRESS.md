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

The new Python sources and tests were parsed with Python's AST before commit.
A complete repository pytest/pre-commit run has not yet been executed in this
environment and remains part of the final audit.

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

The adapter and test file were parsed with Python's AST before commit. The tests
use a fake InternVL runtime to validate batching, prompt construction, aspect
selection, and caching without downloading the 4.43 GB checkpoint.

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

- Extended `common.sh` with provider-independent paths for:
  - official MJ-VIDEO source;
  - MJ-VIDEO-2B checkpoint;
  - InternVL2-2B base model;
  - fixed Physion reward calibration artifact.
- Added `01_download_mj_video.sh`:
  - clones and checks out the exact official source commit;
  - downloads the exact MJ-VIDEO model revision;
  - downloads the pinned InternVL2 base revision;
  - writes revision markers consumed by the runtime adapter;
  - validates all required source/model files.
- Added `h3_rvm_calibration_bank.yaml`:
  - generates up to 100 fixed released-FastH3 videos at the exact production
    geometry and four-step VSA sampler;
  - uses deterministic held-out prompt indices and seeds;
  - runs calibration on the step-zero model before the near-zero-LR launcher
    compatibility update;
  - avoids loading the expensive learned reward stack during generation.
- Added `calibrate_reward_profile.py`:
  - decodes the fixed baseline MP4s with PyAV;
  - scores raw VideoAlign TA, MJ C&C, MJ Fineness, and Dynamic Tracking;
  - uses fixed `median` and `1.4826 * MAD` statistics;
  - falls back to population standard deviation only when MAD is degenerate;
  - refuses a constant component unless the operator gives an explicit audited
    fallback scale;
  - writes a versioned calibration JSON and per-video score JSONL;
  - records Git head, model/source revisions, prompt/video hashes, sample count,
    and reward preprocessing settings.
- Added `04_calibrate_physion_mj_rewards.sh` to generate/reuse the deterministic
  baseline bank and build/validate the calibration artifact.
- Added `preflight_mj_video.py` and `03_preflight_mj_video.sh`:
  - compile and run the focused unit tests;
  - strictly load the real pinned MJ checkpoint;
  - score deterministic videos;
  - verify finite distinct C&C/Fineness outputs;
  - verify one shared forward per video rather than one per aspect.
- Added `test_reward_calibration_cli.py` for median/MAD behavior, constant-scale
  failures/overrides, and prompt-index/video discovery.

### Deliberate separation from the original RVM setup

The original `01_download_models.sh` remains unchanged. MJ-VIDEO is downloaded
only when the alternate profile is requested, so existing RVM users do not pay
for the additional source checkout, base model, or 4.43 GB reward checkpoint.
All calibration and profile setup lives in ordinary repository scripts; no
Modal-only code or runtime patching is involved.

### Next phase

Add the selectable `rvm_h3_8gpu_physion_mj.yaml` configuration and a matched
original-RVM-versus-Physion reward-profile sweep. Add config tests proving the
loss/model/topology differ only where intended.

## Commit log

| Phase | Commit | Summary | Validation |
|---|---|---|---|
| 0 | `55d60aec` | Plan, source inventory, and implementation contract | Source audit |
| 1 | `9848405f` | Fixed calibration, output contracts, distributed profile method | AST parse; tests authored |
| 2 | `c3445e32` | Official MJ-VIDEO adapter, exact preprocessing, shared aspect cache | AST parse; fake-runtime tests authored |
| 3 | `1d2a3077` | Pinned assets, baseline bank, robust calibration, real-model preflight | AST/shell parsing pending final audit |

## GPU validation boundary

No new GPU execution has occurred for this MJ-VIDEO extension. Existing H3 RVM
GPU results validate the pre-existing model/rollout/reward path only. The new
MJ-VIDEO adapter and calibrated profile must pass the committed real-checkpoint
preflight and calibration workflow before any training-quality claim.
