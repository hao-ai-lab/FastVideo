# PR Review: `refactor/train` vs `upstream/main`

**144 files changed, 24042 insertions, 451 deletions.**

This PR mixes three categories of changes. This document categorizes
every file so you can quickly decide what belongs in this PR and what
should be split out or dropped.

---

## Category A — WanGame Model & Infrastructure (60 new + 17 modified files)

Pure WanGame/WanLingBot model, pipeline, data, and example additions.
None of these touch `fastvideo/train/`.

### New model code

| File | Summary |
|------|---------|
| `fastvideo/models/dits/wangame/__init__.py` | Package init |
| `fastvideo/models/dits/wangame/model.py` | WanGame transformer (action-conditioned Wan) |
| `fastvideo/models/dits/wangame/causal_model.py` | Causal WanGame transformer (streaming KV cache) |
| `fastvideo/models/dits/wangame/hyworld_action_module.py` | Keyboard/mouse action embedding module |
| `fastvideo/models/dits/wangame_lingbot/__init__.py` | Package init |
| `fastvideo/models/dits/wangame_lingbot/model.py` | WanLingBot transformer variant |
| `fastvideo/models/dits/wangame_lingbot/cam_utils.py` | Camera utility for LingBot |
| `fastvideo/configs/models/dits/wangamevideo.py` | WanGame/WanLingBot model configs |

### New pipeline code

| File | Summary |
|------|---------|
| `fastvideo/pipelines/basic/wan/wangame_i2v_pipeline.py` | WanGame I2V inference pipeline |
| `fastvideo/pipelines/basic/wan/wangame_causal_dmd_pipeline.py` | WanGame causal DMD pipeline |
| `fastvideo/pipelines/preprocess/wangame/wangame_preprocess_pipeline.py` | WanGame data preprocessing |
| `fastvideo/pipelines/preprocess/wangame/wangame_preprocess_pipeline_ode_trajectory.py` | ODE trajectory preprocessing |
| `fastvideo/pipelines/samplers/__init__.py` | New sampler package init |
| `fastvideo/pipelines/samplers/base.py` | Base sampler class |
| `fastvideo/pipelines/samplers/wan.py` | Wan-specific sampler utilities |

### New legacy training pipelines (old-style, not `fastvideo/train/`)

| File | Summary |
|------|---------|
| `fastvideo/training/wangame_training_pipeline.py` | WanGame SFT training (old pipeline style) |
| `fastvideo/training/wangame_distillation_pipeline.py` | WanGame DMD distillation (old style) |
| `fastvideo/training/wangame_self_forcing_distillation_pipeline.py` | WanGame self-forcing (old style, 952 lines) |
| `fastvideo/training/wangame_ar_diffusion_pipeline.py` | WanGame AR diffusion (old style) |
| `fastvideo/training/wangame_ode_causal_pipeline.py` | WanGame ODE causal (old style) |
| `fastvideo/training/wangame_lingbot_training_pipeline.py` | WanLingBot training (old style) |

> **Question:** Are these legacy `fastvideo/training/wangame_*.py` files
> still needed now that `fastvideo/train/` exists? If they're
> superseded, consider removing them to avoid confusion.

### New examples & scripts (WanGame-specific)

| File | Summary |
|------|---------|
| `examples/inference/basic/basic_wangame.py` | WanGame inference example |
| `examples/inference/basic/basic_causal_wangame.py` | Causal WanGame inference example |
| `examples/inference/basic/basic_wangame_lingbot.py` | WanLingBot inference example |
| `examples/distill/WanGame2.1/distill_dmd.slurm` | WanGame DMD slurm job |
| `examples/distill/WanGame2.1/validation.json` | Validation prompts |
| `examples/distill/SFWanGame2.1/distill_dmd.sh` | Self-forcing distill script |
| `examples/distill/SFWanGame2.1/distill_dmd.slurm` | Self-forcing slurm job |
| `examples/distill/SFWanGame2.1/validation.json` | Validation prompts |
| `examples/training/finetune/WanGame2.1_1.3b_i2v/` | WanGame finetune scripts, slurm jobs, validation JSONs, helper scripts (~15 files) |
| `examples/training/finetune/WanGame2.1_1.3b_i2v_LingBot/` | LingBot finetune scripts (~8 files) |
| `examples/training/consistency_finetune/causal_wangame_ode_init/` | ODE-init consistency finetune scripts (~8 files) |
| `docs/wangame/zero_init_fixes.md` | Zero-init debugging notes |
| `visualize_trajectory.py` | Trajectory visualization tool (224 lines) |

### Modified files (WanGame support)

| File | What changed |
|------|-------------|
| `fastvideo/configs/models/dits/__init__.py` | Imports/exports `WanGameVideoConfig`, `WanLingBotVideoConfig` |
| `fastvideo/configs/pipelines/__init__.py` | Imports/exports WanGame/LingBot pipeline configs |
| `fastvideo/configs/pipelines/wan.py` | Adds WanGame/LingBot/SelfForcing pipeline config dataclasses |
| `fastvideo/dataset/dataloader/record_schema.py` | Adds `wangame_ode_record_creator()` |
| `fastvideo/dataset/dataloader/schema.py` | Adds `pyarrow_schema_wangame`, `_lingbot`, `_ode_trajectory_wangame` |
| `fastvideo/dataset/validation_dataset.py` | Adds action (keyboard/mouse) loading support |
| `fastvideo/models/dits/hyworld/pose.py` | Adds `reformat_keyboard_and_mouse_tensors()`, `process_custom_actions()` |
| `fastvideo/models/loader/fsdp_load.py` | Extends `ALLOWED_NEW_PARAM_PATTERNS` for WanGame modules; adds kaiming init |
| `fastvideo/models/registry.py` | Registers WanGame/LingBot transformer classes |
| `fastvideo/registry.py` | Registers `Wan2.1-Game-Fun-1.3B-InP-Diffusers` HF path |
| `fastvideo/pipelines/preprocess/v1_preprocess.py` | Adds `wangame` and `wangame_ode_trajectory` preprocess tasks |
| `fastvideo/pipelines/stages/denoising.py` | Adds action conditioning via `process_custom_actions()` |
| `fastvideo/pipelines/stages/matrixgame_denoising.py` | Adds action/camera kwargs for causal WanGame inference (+486 lines) |

---

## Category B — New `fastvideo/train/` Architecture (40 new + 15 modified files)

The core of this PR: the new pluggable, YAML-driven training framework.

### New files (all under `fastvideo/train/`)

| File | Summary |
|------|---------|
| `fastvideo/train/__init__.py` | Package init |
| `fastvideo/train/.style.yapf` | YAPF config (80-col) |
| `fastvideo/train/trainer.py` | Training loop: gradient accumulation, callbacks, checkpointing |
| `fastvideo/train/models/base.py` | `ModelBase`, `CausalModelBase` ABCs |
| `fastvideo/train/models/wan/wan.py` | Wan 2.1 T2V model plugin |
| `fastvideo/train/models/wangame/wangame.py` | WanGame I2V model plugin |
| `fastvideo/train/models/wangame/wangame_causal.py` | WanGame causal (streaming) model plugin |
| `fastvideo/train/methods/base.py` | `TrainingMethod` ABC |
| `fastvideo/train/methods/distribution_matching/dmd2.py` | DMD2 distillation method |
| `fastvideo/train/methods/distribution_matching/self_forcing.py` | Self-Forcing method |
| `fastvideo/train/methods/fine_tuning/finetune.py` | SFT method |
| `fastvideo/train/methods/fine_tuning/dfsft.py` | Diffusion-forcing SFT method |
| `fastvideo/train/callbacks/callback.py` | `CallbackDict` registry |
| `fastvideo/train/callbacks/grad_clip.py` | Gradient clipping callback |
| `fastvideo/train/callbacks/validation.py` | Validation callback |
| `fastvideo/train/callbacks/ema.py` | EMA callback |
| `fastvideo/train/entrypoint/train.py` | CLI entrypoint (`torchrun -m fastvideo.train.entrypoint.train`) |
| `fastvideo/train/entrypoint/dcp_to_diffusers.py` | Checkpoint conversion |
| `fastvideo/train/utils/config.py` | YAML parser -> `RunConfig` |
| `fastvideo/train/utils/builder.py` | `build_from_config`: instantiate models + method |
| `fastvideo/train/utils/instantiate.py` | `_target_`-based class instantiation |
| `fastvideo/train/utils/training_config.py` | `TrainingConfig` dataclass |
| `fastvideo/train/utils/dataloader.py` | Dataset/dataloader construction |
| `fastvideo/train/utils/optimizer.py` | Optimizer/scheduler construction |
| `fastvideo/train/utils/checkpoint.py` | DCP save/resume |
| `fastvideo/train/utils/tracking.py` | W&B tracker |
| `fastvideo/train/utils/module_state.py` | `apply_trainable()` |
| `fastvideo/train/utils/moduleloader.py` | Dynamic module loading |
| `fastvideo/train/utils/validation.py` | Validation helpers |
| `fastvideo/train/methods/consistency_model/__init__.py` | Placeholder |
| `fastvideo/train/methods/knowledge_distillation/__init__.py` | Placeholder |
| Various `__init__.py` files | Package inits |

### New example configs (for the new training architecture)

| File | Summary |
|------|---------|
| `examples/train/rfc.md` | Training architecture RFC document |
| `examples/train/issue.md` | Public issue for community discussion |
| `examples/train/run.sh` | Example launch script |
| `examples/train/example.yaml` | Generic example config |
| `examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml` | DMD2 distillation on Wan 2.1 |
| `examples/train/finetune_wan2.1_t2v_1.3B_vsa_phase3.4_0.9sparsity.yaml` | VSA finetune on Wan 2.1 |
| `examples/train/finetune_wangame2.1_i2v_1.3B.yaml` | Finetune WanGame (new arch) |
| `examples/train/dfsft_wangame_causal_v3.yaml` | DFSFT on causal WanGame (new arch) |
| `examples/train/self_forcing_wangame_causal_v3.yaml` | Self-forcing on causal WanGame (new arch) |

### Modified files outside `fastvideo/train/` (needed by the new architecture)

| File | What changed | Necessary? |
|------|-------------|------------|
| `fastvideo/configs/pipelines/base.py` | Adds `sampler_kind` (ode/sde) and `ode_solver` config fields | Yes — pluggable sampler strategy |
| `fastvideo/dataset/parquet_dataset_map_style.py` | Multi-path with repeat counts (`/dir:2`), epoch reshuffling, hash-based caching | Yes — flexible dataset composition |
| `fastvideo/fastvideo_args.py` | Adds `reshuffle_each_epoch`, `validation_num_samples`, action training flags | Partially — some WanGame-specific |
| `fastvideo/models/loader/component_loader.py` | Removes unused imports; fixes FSDP exclusions | Partially — some cleanup, some WanGame |
| `fastvideo/pipelines/basic/wan/wan_pipeline.py` | Refactors to use `build_wan_scheduler()`, pluggable ODE/SDE sampler | Yes — sampler abstraction |
| `fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py` | Thins to compatibility wrapper (sets `sampler_kind=sde`, delegates to WanPipeline) | Yes — aligns with sampler refactor |
| `fastvideo/pipelines/basic/wan/wan_i2v_dmd_pipeline.py` | Removes `TimestepPreparationStage` (not needed for SDE) | Yes — aligns with sampler refactor |
| `fastvideo/pipelines/pipeline_batch_info.py` | Adds `sampling_timesteps` field to `ForwardBatch` | Yes — SDE denoising needs explicit timesteps |
| `fastvideo/pipelines/stages/__init__.py` | Exports `SdeDenoisingStage`, `MatrixGameCausalOdeDenoisingStage` | Yes — new stages |
| `fastvideo/training/checkpointing_utils.py` | Fixes activation-checkpoint wrapper key renaming in `ModelWrapper.state_dict()` | Yes — bugfix for grad checkpointing + DCP |
| `fastvideo/training/distillation_pipeline.py` | Adds `num_samples` to validation, hasattr checks, video saving | Partially — some robustness, some WanGame |
| `fastvideo/training/training_pipeline.py` | Epoch reshuffling, deterministic seeds, trainable param counting | Yes — training improvements |
| `fastvideo/training/training_utils.py` | Adds `count_trainable_total()` for distributed param counting | Yes — FSDP-aware param logging |

---

## Category C — Standalone Bugfixes / Improvements (6 files)

Small fixes and improvements unrelated to either WanGame or the new
training architecture. These could be merged independently.

| File | What changed |
|------|-------------|
| `.gitignore` | Adds `*.npy`, `slurm_outputs/` |
| `fastvideo/configs/sample/wan.py` | Updates `Wan2_1_Fun_1_3B_InP_SamplingParam` defaults (resolution 352x640, 77 frames, 25fps, guidance 1.0, 40 steps) |
| `fastvideo/configs/pipelines/wan.py` | Same sampling param updates |
| `fastvideo/training/trackers.py` | Adds `log_file()` to `BaseTracker` / `WandbTracker` / `SequentialTracker` |
| `fastvideo/utils.py` | Adds `.cpu()` before `.numpy()` on GPU tensor — **bugfix** |
| `fastvideo/models/dits/matrixgame/utils.py` | Code reformatting of already-commented drawing functions — **no functional change** |

---

## Summary Table

| Category | New files | Modified files | Lines added |
|----------|-----------|----------------|-------------|
| **A. WanGame** | ~60 | 13 | ~12,600 |
| **B. `fastvideo/train/`** | ~40 | 13 | ~8,000 |
| **C. Bugfixes** | 0 | 6 | ~100 |
| **Overlap (A+B)** | — | ~4 | — |

---

## Recommended Review Order

1. **Category C first** — 6 small, independent changes. Quick to review
   and merge separately if desired.

2. **Category B (`fastvideo/train/`)** — The core architecture. Start
   with `base.py` (model + method ABCs), then `trainer.py`, then the
   four method implementations, then utils.

3. **Category A (WanGame)** — Larger but mostly additive. The key
   question is whether the legacy `fastvideo/training/wangame_*.py`
   pipelines (~3,500 lines) are still needed alongside the new
   `fastvideo/train/` architecture.

4. **Overlap files** — Files modified for both WanGame and the training
   architecture (e.g., `fastvideo_args.py`, `component_loader.py`,
   `distillation_pipeline.py`, `denoising.py`). Review these last
   since they require understanding both contexts.

---

## Open Questions

1. **Legacy training pipelines**: 6 new files under
   `fastvideo/training/wangame_*.py` (~3,500 lines) use the old
   training pipeline pattern. Are these still needed, or are they
   superseded by `fastvideo/train/` configs?

2. **`matrixgame/utils.py`**: 351-line diff that appears to be
   formatting-only on commented-out code. Drop?

3. **`visualize_trajectory.py`**: Top-level script (224 lines). Should
   this live under `examples/` or `scripts/` instead?

4. **Sampling param changes** (`configs/sample/wan.py`,
   `configs/pipelines/wan.py`): Changed resolution from 480x832 to
   352x640, frames 81->77, guidance 6.0->1.0. Is this intentional for
   all users or WanGame-specific?

5. **`fastvideo_args.py` additions**: Several new flags
   (`train_action_only`, `action_train_target`,
   `action_warmup_steps`, `best_checkpoint_start_step`) appear
   WanGame-specific. Should these live in a WanGame-specific config
   instead of the shared args?
