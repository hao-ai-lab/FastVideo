---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# New preprocessing entrypoint (v1_preprocessing_new.py) may not be ported for a model

## What Happened
The example preprocessing script for Cosmos 2.5 was written to call
`v1_preprocessing_new.py` (the new WorkflowBase system). Running it failed
immediately because Cosmos 2.5 was not registered in the new system. The
error was not obvious — it looked like a config resolution failure.

## Root Cause
FastVideo has two preprocessing entrypoints:
- `v1_preprocess.py` — old flat-argparse system, works with any model.
- `v1_preprocessing_new.py` — new WorkflowBase system, requires the model
  to be explicitly registered and its preprocessing config ported.

Cosmos 2.5 was only available in the old system at the time of the port.

## Fix / Workaround
Use `v1_preprocess.py` with flat args (`--data_merge_path`, `--max_height`,
etc.) instead of `--preprocess.X` prefix form:

```bash
torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --max_height 704 --max_width 1280 --num_frames 77 \
    --train_fps 24 --preprocess_task "t2v"
```

Porting to `v1_preprocessing_new.py` is a separate ~1-2h task.

## Prevention
- Before writing the example preprocessing script for a new model, check
  whether the model is registered in `v1_preprocessing_new.py`.
- Grep for the model name in `fastvideo/pipelines/preprocess/` to confirm.
- If not registered, use `v1_preprocess.py` and flag the port work to the
  maintainer separately.
