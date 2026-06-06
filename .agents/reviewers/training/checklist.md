# Training reviewer checklist

## Framework choice

- [ ] Net-new training code lives in `fastvideo/train/` (not `fastvideo/training/`).
- [ ] If legacy is touched, it's a bugfix or a documented deprecation, not new features.

## Training method (`fastvideo/train/methods/`)

- [ ] New method subclasses `TrainingMethod` in `fastvideo/train/methods/base.py`.
- [ ] `training_step`, `validation_step` have consistent signatures with
      other methods.
- [ ] Method-specific state (EMA / student / teacher) is initialized,
      forward-compatible, and serialized in checkpoints.
- [ ] Method is registered with the builder
      (`fastvideo/train/utils/builder.py`).

## Distributed

- [ ] SP collectives use the correct group (`get_sequence_model_parallel_group`).
- [ ] No conditional `return` inside SP regions that could leave ranks
      stranded (ref #1178).
- [ ] FSDP wrap policy updated if a new top-level module is added.
- [ ] Multi-node env not hard-coded in new scripts.

## Checkpointing

- [ ] DCP save/load path covered — state dict round-trips.
- [ ] EMA / teacher / student state dicts included in checkpoint.
- [ ] `resume_from_checkpoint` works (author confirms in PR body).

## Dataset / preprocessing

- [ ] New preprocessing has a test under `fastvideo/tests/training/` or a
      preprocessing test file.
- [ ] No assumption about encoder presence (e.g. CLIP) without a graceful
      fallback — ref #1184.
- [ ] If `tiled_encode` / `tiled_decode` in VAE preprocessing is touched,
      output is checked against a reference (ref #1181).

## YAML config

- [ ] New YAML in `examples/train/` uses HF model IDs or env vars — no
      hard-coded local paths.
- [ ] `log_validation: true` (or equivalent) set.
- [ ] `wandb_run_name` + `tracker_project_name` set sensibly.
- [ ] References to nested configs exist on disk.

## Perf / memory

- [ ] No `.item()` / `.cpu()` in the training step hot loop (ref #1217).
- [ ] Activation offloading / gradient accumulation / mixed precision
      correctness verified.
- [ ] Memory report included in PR body if the PR claims a savings.

## Training evidence

- [ ] PR body includes a short training-curve W&B link or screenshot
      (even a few hundred steps) for method/config changes.
- [ ] Validation loss / sample quality improves or is stable.

## PR template compliance

- [ ] Title prefix matches repo tag regex (`[feat]` / `[bugfix]` / etc).
- [ ] `/test training` command referenced or run, if applicable.
