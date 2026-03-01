# Phase 2 examples

This folder contains **YAML-only** distillation examples for the Phase 2
entrypoint:

- `fastvideo/training/distillation.py --config path/to/distill.yaml`

Important:

- Phase 2 does not rewrite config paths automatically. Pass the explicit YAML
  path (we keep runnable YAMLs under `fastvideo/distillation/outside/`).

Start from:

- `distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`

Recommended:

- Edit the runnable YAML in:
  `fastvideo/distillation/outside/fastvideo/configs/distillation/`
- `temp.sh` (runs the config above; same dataset + validation defaults as Phase0/Phase1).

Resume:

- Checkpoints are saved under `${output_dir}/checkpoint-<step>/dcp/` when
  `training.training_state_checkpointing_steps > 0` in YAML.
- To resume, pass a checkpoint directory (or an output_dir to auto-pick latest):
  - `fastvideo/training/distillation.py --config <yaml> --resume-from-checkpoint <checkpoint-or-output-dir>`
