# fastvideo/train — Modular Training Framework

This is the new training infrastructure for FastVideo, replacing the
monolithic per-model pipeline scripts in `fastvideo/training/`.

## Key differences from `fastvideo/training/`

| | `fastvideo/training/` (legacy) | `fastvideo/train/` (new) |
|---|---|---|
| Architecture | One pipeline class per model+method combo | Composable: model plugins + method classes |
| Adding a model | Write a full training pipeline from scratch | Add a model plugin under `models/` |
| Adding a method | Fork an existing pipeline | Add a method class under `methods/` |
| Config | Flat argparse flags | Hierarchical YAML with dotted-key overrides |
| Distributed | Manual FSDP setup per pipeline | Shared trainer handles FSDP/HSDP |

## Structure

```
fastvideo/train/
├── models/       # Per-model plugins (wan/, hunyuan/, ...)
├── methods/      # Training methods (fine_tuning/, distribution_matching/, ...)
├── callbacks/    # Validation, grad clipping, etc.
├── entrypoint/   # CLI entry points (train.py, dcp_to_diffusers.py)
├── trainer.py    # Core training loop
└── utils/        # Config parsing, dataloader, etc.
```

## Usage

See `examples/train/` for configs and launch scripts.
