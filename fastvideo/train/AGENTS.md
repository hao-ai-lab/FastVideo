# `fastvideo/train/` — Modular Training Framework

**Generated:** 2026-05-02

YAML-driven trainer composed from interchangeable **methods × models × callbacks**. Preferred location for new training code. (See sibling `fastvideo/training/AGENTS.md` for the legacy stack.)

## Layout

```
train/
├── trainer.py              # Core training loop coordinator
├── README.md               # User-facing overview (legacy → new diff)
├── entrypoint/             # train.py + dcp_to_diffusers.py CLI entrypoints
├── methods/
│   ├── base.py                  #   TrainingMethod ABC
│   ├── fine_tuning/             #   FineTuneMethod, DiffusionForcingSFTMethod
│   ├── distribution_matching/   #   DMD2Method, SelfForcingMethod
│   ├── knowledge_distillation/  #   KDMethod, KDCausalMethod
│   └── consistency_model/       #   Consistency-model training methods
├── models/
│   ├── base.py             #   ModelBase / CausalModelBase wrappers
│   ├── wan/, hunyuan/, cosmos/  # Per-family training wrappers
├── callbacks/              # callback.py base + ema, grad_clip, validation
└── utils/
    ├── training_config.py  #   Hierarchical YAML config dataclasses
    ├── builder.py          #   Build trainer from config
    ├── checkpoint.py       #   DCP save/load
    ├── optimizer.py        #   AdamW / fused optimizer factory
    ├── tracking.py         #   build_tracker (W&B / TensorBoard)
    └── dataloader.py       #   StatefulDataLoader wiring
```

## Composition Model

```
Trainer = Method × Model × [Callback...] × Config
```

- A **Method** owns the loss + optimizer step (`compute_loss`, `step_post_grad`).
- A **Model** owns the forward + parameter-grouping (`forward`, `trainable_parameters`).
- **Callbacks** subscribe to lifecycle hooks (`on_train_start`,
  `on_training_step_end`, `on_before_optimizer_step`, `on_validation_begin`,
  `on_validation_end`, `on_train_end`) and compose freely.
- **Config** is a Pydantic-style hierarchical YAML resolved by
  `utils/training_config.py`. Dotted-key CLI overrides go through
  `parse_overrides`.

## Adding a New Model Plugin

1. Subclass `ModelBase` (or `CausalModelBase`) in `models/<family>/`.
2. Wrap the existing inference DiT from `fastvideo/models/dits/`. Do not
   reimplement.
3. Expose `trainable_parameters()` so the optimizer factory can group them.
4. Register in `utils/builder.py` if the trainer dispatches by name.

## Adding a New Method

1. Subclass `TrainingMethod` in `methods/<family>/`.
2. Implement `compute_loss(batch, model_outputs) -> dict`.
3. If the method needs a teacher / second model, expose a `build_extras(cfg)`
   classmethod — never instantiate inside `__init__`.

## Configs

YAML lives under `examples/train/`. Schemas in `utils/training_config.py`. Add
new fields with explicit defaults; agents and humans both rely on the dataclass
to discover knobs.

## Anti-Patterns

- Importing from `fastvideo.training.*` (legacy stack). Forbidden cross-import.
- Adding a new training method as a fork of an existing pipeline file. Compose
  via `Method` instead.
- Logging via stdlib `logging` — use `init_logger(__name__)`.
- Mutating the global state of a model in a callback. Callbacks operate on the
  trainer state object passed in.
