# Profiling FastVideo

!!! warning
    Profiling is only intended for FastVideo developers and maintainers to understand the proportion of time spent in different parts of the codebase. **FastVideo end-users should never turn on profiling** as it will significantly slow down inference.

## Profiling with PyTorch

FastVideo exposes a process-wide torch profiler that you can enable via environment variables. Set `FASTVIDEO_TORCH_PROFILER_DIR` to an absolute directory path to start collecting traces, and specify the regions you want recorded with `FASTVIDEO_TORCH_PROFILE_REGIONS`:

```bash
FASTVIDEO_TORCH_PROFILER_DIR=/mnt/traces/fastvideo \
FASTVIDEO_TORCH_PROFILE_REGIONS="model_loading,training_train_one_step" \
bash examples/train/run.sh /path/to/config.yaml
```

The `profiler_region_` prefix is optional. All profiled regions must be
registered in `fastvideo.profiler`; the current list includes:

- `profiler_region_model_loading` — pipeline/module loading
- `profiler_region_inference_denoising`
- `profiler_region_training_train` — the complete training run
- `profiler_region_training_train_one_step` — one complete optimizer step
- `profiler_region_training_dataloader` — fetch the next batch in the trainer process
- `profiler_region_training_forward` — method forward and loss computation
- `profiler_region_training_backward` — backward pass
- `profiler_region_training_optimizer` — gradient clipping, optimizer/scheduler step, and zeroing gradients
- `profiler_region_training_callbacks` — end-of-step callbacks such as EMA updates
- `profiler_region_training_validation`
- `profiler_region_training_save_checkpoint`
- `profiler_region_distillation_teacher_forward`
- `profiler_region_distillation_student_forward`
- `profiler_region_distillation_loss`
- `profiler_region_distillation_update`
- `profiler_region_dmd2_student_rollout` — one DMD2 student rollout, including simulated prefix steps
- `profiler_region_dmd2_generator_loss` — teacher/critic scoring for the generator loss
- `profiler_region_dmd2_critic_loss` — critic flow-matching loss, including its student rollout

### Profiling modular training

The YAML-driven trainer under `fastvideo/train/` initializes and flushes the
profiler automatically. For example, this captures six DMD2 steps: one warmup
step, four steady-state critic updates, and the configured 1-in-5 generator
update, without changing the method's update cadence:

```bash
TRACE_DIR="$(pwd)/profiler_traces/wan_dmd2"
mkdir -p "$TRACE_DIR"

NUM_GPUS=4 \
WANDB_MODE=offline \
FASTVIDEO_TORCH_PROFILER_DIR="$TRACE_DIR" \
FASTVIDEO_TORCH_PROFILE_REGIONS="training_train,training_train_one_step,training_dataloader,training_forward,training_backward,training_optimizer,training_callbacks,dmd2_student_rollout,dmd2_generator_loss,dmd2_critic_loss" \
bash examples/train/run.sh \
    examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml \
    --training.loop.max_train_steps 6 \
    --training.checkpoint.training_state_checkpointing_steps 0 \
    --callbacks.validation.every_steps 0
```

The two overrides disable validation and checkpoint writes so their I/O does
not contaminate training-step measurements. Remove them when profiling those
regions. Methods that manage optimization internally emit the enclosing
`training_train_one_step` region but do not emit the generic
dataloader/forward/backward/optimizer child regions because those boundaries
belong to the method. With `dataloader_num_workers > 0`, the
`training_dataloader` region measures the training process waiting for and
receiving the next batch; work performed inside dataloader worker subprocesses
does not appear in that process's torch-profiler trace.

The modular trainer also logs `dataloader_time_sec` on every ordinary training
step (including DMD2), so routine runs can monitor rank 0's
`next(dataloader)` wait without enabling the torch profiler. It is
intentionally not reduced across ranks; cross-rank synchronization would
perturb the hot path being measured.

While profiling is enabled, FastVideo records additional annotations:

- `fastvideo.region::<name>` spans are emitted when entering a region.

Only one profiler controller is created per process; subsequent pipelines
reuse it. Each outermost enabled region invocation produces one complete
CPU/CUDA trace segment. Enabled regions nested inside it are annotations in
that segment. Include an enclosing region such as `training_train_one_step`
when selecting its child regions so they share one trace and profiler startup
does not perturb each phase independently. If you set
`FASTVIDEO_TORCH_PROFILE_REGIONS` incorrectly (e.g. misspelled name), FastVideo
logs a warning and ignores that entry.

Additional knobs:

- `FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES`
- `FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY`
- `FASTVIDEO_TORCH_PROFILER_WITH_STACK`
- `FASTVIDEO_TORCH_PROFILER_WITH_FLOPS`

Traces can be visualized using <https://ui.perfetto.dev/>. Each rank also
writes `summary_rank<N>_segment<MMMM>_<region>.txt` and the corresponding JSON
beside every trace segment for a quick operator-level view without loading the
full timeline. The JSON contains CPU and device totals; input-shape grouping is
only enabled with `FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=1` because it makes
summary generation materially more expensive on large traces.

### Best Practices

- Keep the profiled step count small; traces can be large and slow down job shutdown while the profiler flushes data.
- After profiling, clean up trace directories to avoid filling disk storage.
- When adding new regions, register them in `fastvideo.profiler` and wrap the corresponding code block with `profiler_region("your_region")` or the `@profile_region` decorator.

## Related: Activation Trace Mode

For per-layer **numerical-divergence** debugging (parity bring-up against an
upstream reference), use the env-gated [Activation Trace Mode](activation_trace.md)
instead of the torch profiler. Activation trace dumps per-tensor stats to
JSONL for offline `diff`-ing; the torch profiler captures kernel timing.
They solve different problems and can run together if needed.
