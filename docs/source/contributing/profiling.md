# Profiling FastVideo

!!! warning
    Profiling is only intended for FastVideo developers and maintainers to understand the proportion of time spent in different parts of the codebase. **FastVideo end-users should never turn on profiling** as it will significantly slow down the inference.

## Profiling with PyTorch

FastVideo exposes a process-wide torch profiler that you can enable via environment variables. Set `FASTVIDEO_TORCH_PROFILER_DIR` to an absolute directory path to start collecting traces, and specify the regions you want recorded with `FASTVIDEO_TORCH_PROFILE_REGIONS`:

```bash
FASTVIDEO_TORCH_PROFILER_DIR=/mnt/traces/fastvideo \
FASTVIDEO_TORCH_PROFILE_REGIONS="profiler_region_model_loading,profiler_region_training_step"
```

All profiled regions must be registered in `fastvideo.profiler`; the current list includes:

- `profiler_region_model_loading` — pipeline/module loading
- `profiler_region_inference_pre_denoising`
- `profiler_region_inference_denoising`
- `profiler_region_inference_post_denoising`
- `profiler_region_training_checkpoint_saving`
- `profiler_region_training_dit`
- `profiler_region_training_validation`
- `profiler_region_training_epoch`
- `profiler_region_training_step`
- `profiler_region_training_backward`
- `profiler_region_training_optimizer`
- `profiler_region_distillation_teacher_forward`
- `profiler_region_distillation_student_forward`
- `profiler_region_distillation_loss`
- `profiler_region_distillation_update`

While profiling is enabled, FastVideo records additional annotations:

- `fastvideo.region::<name>` spans are emitted when entering a region.
- `fastvideo.profiler.enable_collection` / `fastvideo.profiler.disable_collection` events mark when torch profiler collection is toggled on or off.

Only one profiler instance is created per process; subsequent pipelines reuse the same controller. If you set `FASTVIDEO_TORCH_PROFILE_REGIONS` incorrectly (e.g. misspelled name), FastVideo logs a warning and ignores that entry.

Additional knobs:

- `FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES`
- `FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY`
- `FASTVIDEO_TORCH_PROFILER_WITH_STACK`
- `FASTVIDEO_TORCH_PROFILER_WITH_FLOPS`

Traces can be visualized using <https://ui.perfetto.dev/>.

### Best Practices

- Keep the profiled step count small; traces can be large and slow down job shutdown while the profiler flushes data.
- After profiling, clean up trace directories to avoid filling disks.
- When adding new regions, register them in `fastvideo.profiler` and wrap the corresponding code block with `with self.profiler_controller.region("your_region"):` or the `@profile_region` decorator.
