---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# initialize_pipeline() shift setting is ignored during training — train() always overwrites

## What Happened
Cosmos 2.5 training appeared to work (loss decreased, outputs were reasonable)
but was silently using the wrong sigma distribution. The model was trained with
shift=1.0 instead of the intended shift=5.0, meaning timestep sampling was
uniform rather than biased toward high noise.

## Root Cause
`TrainingPipeline.train()` unconditionally overwrites `self.noise_scheduler`
at startup:
```python
self.noise_scheduler = FlowMatchEulerDiscreteScheduler()  # shift=1.0 default
```
This happens AFTER `initialize_training_pipeline()` already ran and set
`self.noise_scheduler = self.modules["scheduler"]` (which had the correct
shift). Additionally, `post_init()` calls `initialize_training_pipeline()`
BEFORE `initialize_pipeline()`, so the custom shift set in
`initialize_pipeline()` never reaches training.

The same issue exists in `wan_training_pipeline.py` (shift=3.0 set but
ignored). This is a framework-level bug in `TrainingPipeline`.

## Fix / Workaround
Override `_sample_timesteps` in the model-specific training pipeline to
lazy-patch the scheduler on first call:
```python
def _sample_timesteps(self, batch_size, device):
    shift = self.training_args.pipeline_config.flow_shift
    if self.noise_scheduler.shift != shift:
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    return super()._sample_timesteps(batch_size, device)
```

Or flag it to the maintainer — since Wan has the same issue, the fix belongs
in the base class `train()` method.

## Prevention
- For any model with non-default flow_shift, verify that `self.noise_scheduler`
  has the correct shift after `train()` is called, not just after
  `initialize_pipeline()`.
- The training sigma distribution is set by `FlowMatchEulerDiscreteScheduler`,
  not `FlowUniPCMultistepScheduler` — they are used for different purposes
  (training vs inference respectively).
