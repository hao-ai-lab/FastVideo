---
date: 2026-04-09
experiment: Cosmos 2.5 T2W port (PR #1227)
category: porting
severity: important
---

# Hardcoded T5-XXL shape (512, 4096) in dataset/utils.py breaks non-Wan encoders

## What Happened
CFG dropout during training silently produced zero embeddings of shape
`(512, 4096)` for Cosmos 2.5's Reason1 encoder, which outputs
`(seq_len, 100352)`. The model received wrong-shaped null embeddings and
training loss was nonsensical when `--training_cfg_rate > 0`.

## Root Cause
`fastvideo/dataset/utils.py` contained hardcoded fallback shapes matching
Wan's T5-XXL encoder:
```python
data = np.zeros((512, 4096), dtype=np.float32)  # hardcoded for Wan
```
This is used in two functions: `get_torch_tensors_from_row_dict` and
`collate_rows_from_parquet_schema`. The `shape` variable was available from
the parquet schema but ignored.

## Fix / Workaround
Replace the hardcoded tuple with the `shape` variable read from the schema:
```python
data = np.zeros(shape, dtype=np.float32)
```

## Prevention
- When porting any model with a non-T5-XXL text encoder (hidden dim ≠ 4096
  or seq_len ≠ 512), immediately audit `fastvideo/dataset/utils.py` for
  hardcoded shapes.
- Search for `(512, 4096)` as a quick grep check.
- Workaround if you can't edit utils.py yet: set `--training_cfg_rate 0.0`
  to disable CFG dropout until the fix is in place.
