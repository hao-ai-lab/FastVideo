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

**Status:** complete; awaiting commit.

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
5. RVM currently knows some diagnostic keys through a Dynamic Tracking-specific
   branch. General reward-output-key discovery is needed before calibrated/MJ
   diagnostics can be broadcast safely across SP ranks.

### Next phase

Implement and test the fixed reward-calibration layer and general reward-output
key discovery. This phase must preserve byte-for-byte aggregate behavior for the
existing reward profile when calibration is not configured.

## Commit log

This table is updated after every pushed phase.

| Phase | Commit | Summary | Validation |
|---|---|---|---|
| 0 | pending | Plan, source inventory, and implementation contract | Source audit only |

## GPU validation boundary

No new GPU execution has occurred for this MJ-VIDEO extension. Existing H3 RVM
GPU results validate the pre-existing model/rollout/reward path only. The new
MJ-VIDEO adapter and calibrated profile must pass a fresh real-checkpoint
preflight before any training-quality claim.
