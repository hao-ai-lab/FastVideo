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

### Next phase

Implement the pinned MJ-VIDEO runtime adapter, exact eight-frame preprocessing,
aspect extraction, one-model/two-scorer shared cache, and fake-runtime tests.
Then add a real-checkpoint GPU preflight; no compatibility claim will be made
until that forward succeeds.

## Commit log

| Phase | Commit | Summary | Validation |
|---|---|---|---|
| 0 | `55d60aec` | Plan, source inventory, and implementation contract | Source audit |
| 1 | `9848405f` | Fixed calibration, output contracts, distributed profile method | AST parse; tests authored |

## GPU validation boundary

No new GPU execution has occurred for this MJ-VIDEO extension. Existing H3 RVM
GPU results validate the pre-existing model/rollout/reward path only. The new
MJ-VIDEO adapter and calibrated profile must pass a fresh real-checkpoint
preflight before any training-quality claim.
