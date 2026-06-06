# Training reviewer — references

## New framework (`fastvideo/train/`, preferred)

- `fastvideo/train/trainer.py` — main training loop coordinator.
- `fastvideo/train/entrypoint/train.py` — `torchrun` entry point.
- `fastvideo/train/methods/base.py` — `TrainingMethod` ABC.
- `fastvideo/train/methods/fine_tuning/` — `FineTuneMethod`, `DiffusionForcingSFTMethod`.
- `fastvideo/train/methods/distribution_matching/` — `DMD2Method`, `SelfForcingMethod`.
- `fastvideo/train/models/` — per-model wrappers (e.g. `wan/WanModel`, `wan/WanCausalModel`).
- `fastvideo/train/callbacks/` — composable hooks (grad_clip, ema, validation).
- `fastvideo/train/utils/builder.py` — method / model / optimizer registration.
- `fastvideo/train/utils/checkpoint.py` — DCP save / load.

YAML config examples: `examples/train/*.yaml`.

## Legacy framework (`fastvideo/training/`, being phased out)

- `fastvideo/training/training_pipeline.py` — base.
- `fastvideo/training/wan_training_pipeline.py` — Wan T2V.
- `fastvideo/training/wan_i2v_training_pipeline.py` — Wan I2V.
- `fastvideo/training/wan_distillation_pipeline.py` — Wan distill.
- `fastvideo/training/wan_self_forcing_distillation_pipeline.py` — self-forcing distill.
- `fastvideo/training/ltx2_training_pipeline.py` — LTX-2.
- `fastvideo/training/matrixgame_training_pipeline.py` — MatrixGame.
- `fastvideo/training/trackers.py` — W&B tracker.
- `fastvideo/training/training_utils.py` — checkpointing, grad clipping.

Launch scripts: `scripts/distill/*.sh`, `examples/training/**/*.sh`.

## Distributed

- `fastvideo/distributed/__init__.py`
- `fastvideo/distributed/communication_op.py` — SP collectives.
- `fastvideo/distributed/parallel_state.py` — group management.

## Dataset

- `fastvideo/dataset/` — dataset + loader.
- `fastvideo/pipelines/preprocess/` — preprocessing stages.
- `examples/preprocessing/*.sh` — preprocessing scripts.

## Recent relevant fixes

- #1178 — SP deadlock in negative prompt encoding during training.
- #1217 — CPU-GPU sync elimination in training pipeline.
- #1192 — LoRA distillation distributed checkpointing bug.
- #1173 — self-forcing train/validation step mismatch.
- #1181 — VAE temporal tiling blend corruption.
- #1184 — I2V preprocessing crash for models without CLIP.
- #1152 — preprocessing pipeline tests.

## Docs

- `docs/training/finetune.md` — training args table.
- `docs/training/data_preprocess.md` — preprocessing.
- `docs/contributing/testing.md` — test expectations.

## Related skills

- `.agents/skills/launch-experiment/SKILL.md`
- `.agents/skills/monitor-experiment/SKILL.md`
- `.agents/skills/summarize-run/SKILL.md`
- `.agents/skills/log-experiment/SKILL.md`
- `.agents/workflows/experiment-lifecycle.md`
