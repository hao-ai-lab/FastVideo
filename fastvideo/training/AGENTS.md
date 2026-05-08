# `fastvideo/training/` — Legacy Monolithic Training Pipelines

**Generated:** 2026-05-02

> **Status:** maintenance mode. New training work should go in `fastvideo/train/`.
> Existing shipped recipes (Wan, LTX-2, MatrixGame distillation) still live here.

## What's Here

```
training/
├── training_pipeline.py                  # Base TrainingPipeline ABC
├── distillation_pipeline.py              # Base DistillationPipeline (~80 KB)
├── self_forcing_distillation_pipeline.py # Self-forcing causal distill (~55 KB)
├── wan_training_pipeline.py              # Wan T2V finetune
├── wan_i2v_training_pipeline.py          # Wan I2V finetune
├── wan_distillation_pipeline.py          # Wan T2V distill
├── wan_i2v_distillation_pipeline.py      # Wan I2V distill
├── wan_self_forcing_distillation_pipeline.py
├── ltx2_training_pipeline.py             # LTX-2 training
├── matrixgame_training_pipeline.py       # MatrixGame training
├── ode_causal_pipeline.py                # ODE-causal pipeline
├── checkpointing_utils.py                # save/load helpers
├── activation_checkpoint.py              # AC wrapping helpers
├── trackers.py                           # WandbTracker (legacy)
└── training_utils.py                     # Grad clip, FSDP state-dicts (~75 KB)
```

## Style of This Stack

- One file per (model × method). Subclass an existing pipeline if your config is
  a near-clone; otherwise fork and rename.
- Pipelines are torch-distributed-launched directly (no shared trainer). Each
  pipeline calls `dist.init_process_group` lifecycle through
  `fastvideo.distributed`.
- Configs are flat argparse flags wired in `fastvideo_args.py` and per-pipeline
  argparse groups. No YAML.

## When to Touch a File Here

- Bugfix or behavior tweak to a shipped recipe (Wan, LTX-2, MatrixGame).
- Performance / memory regression in `training_utils.py` (used by both stacks).
- New attention backend that needs distillation-time wiring.

## When to NOT Touch a File Here

- Adding a new model's training. Build it in `fastvideo/train/` instead.
- Adding a new training method (DMD2 variant, new self-forcing variant) for a
  model that already has a `train/` plugin. Add a method class in
  `train/methods/`.
- Refactoring shared utilities. The new stack owns the modular utilities now.

## Cross-Stack Rule

Imports from `fastvideo.train.*` into `fastvideo.training.*` (or vice versa)
are **forbidden**. They are independent stacks; mixing them creates cyclic
dependency hazards. Share via `fastvideo.training.training_utils` or
`fastvideo.distributed` only when both stacks already use the helper.
