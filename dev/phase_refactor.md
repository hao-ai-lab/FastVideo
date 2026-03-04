# Phase: Refactor — `_target_` instantiate-first, drop the string registry

## 0) FastGen's hierarchy (the reference we follow)

```
FastGenNetwork (ABC, nn.Module)       ← backbone: owns transformer + noise_scheduler + VAE
    ├── WanNetwork                    ← Wan-specific forward, text conditioning
    └── CausalFastGenNetwork          ← adds chunk_size, clear_caches()
        └── CausalWanNetwork

FastGenModel (nn.Module)              ← base: owns self.net + self.teacher + optimizers
    ├── DMD2Model                     ← adds self.fake_score, self.discriminator, DMD2 loss
    ├── SelfForcingModel              ← adds causal rollout logic
    └── SFTModel                      ← vanilla finetuning
```

**Key lessons from FastGen:**

1. `FastGenNetwork.forward(x_t, t, condition, fwd_pred_type)` is the standardized per-role
   interface. Each role is an independent network instance. No shared dispatcher.
2. `FastGenModel` **IS the method** — it owns `self.net` (student) and `self.teacher` directly
   as attributes, just like `DMD2Model.self.fake_score`. No RoleManager.
3. Optimizers live on `FastGenModel`, not on the network: `self.net_optimizer`,
   `self.fake_score_optimizer`, etc. Method subclasses add their own via `init_optimizers()`.
4. VAE/text encoder live on `self.net` via `net.init_preprocessors()`. For us, this
   concept maps to `SharedContext` (explained below).

---

## 1) Our hierarchy

We cannot directly subclass `FastGenNetwork` because our transformers are raw
`nn.Module` objects from diffusers (not `FastGenNetwork`). But we follow the same
**structural shape**:

```
SharedContextBase (ABC)               ← VAE, scheduler, prepare_batch, dataloader, validator
    ├── WanSharedContext              ← Wan T2V: text conditioning, timestep mechanics
    └── WanGameSharedContext          ← WanGame I2V+action: image/action conditioning

ModelBase (ABC)                       ← per-role: owns ONE transformer, predict_{noise,x0}
    ├── WanModelBase                  ← shared bidi Wan forward logic
    │   ├── WanModel                  ← T2V (text conditioning)
    │   └── WanGameModel              ← I2V+action conditioning
    └── CausalModelBase               ← adds streaming ops, clear_caches
        ├── WanCausalModel            ← causal T2V
        └── WanGameCausalModel        ← causal I2V+action

DistillMethod (nn.Module, ABC)        ← base: generic optimizer loop, checkpoint props
    ├── DMD2Method                    ← self.student, self.teacher, self.fake_score
    ├── SelfForcingMethod             ← self.student (must be CausalModelBase), self.teacher, self.critic
    └── FinetuneMethod / DFSFTMethod  ← self.student only
```

**SharedContext** is analogous to `FastGenNetwork.init_preprocessors()` — it holds
what is truly shared across all roles. `prepare_batch` lives here because it is
pure preprocessing: it samples timesteps, normalizes latents, builds attn metadata,
and populates `batch.conditional_dict` / `batch.unconditional_dict`. It does not
touch any transformer.

**ModelBase** is analogous to `FastGenNetwork`. Each role is an independent instance
(like FastGen's `self.net`, `self.teacher`). It owns one transformer and implements
`predict_noise`/`predict_x0` by calling that transformer + reading precomputed
fields from `TrainingBatch`.

**DistillMethod** is analogous to `FastGenModel`. It owns the role model objects,
all optimizers, and implements `single_train_step`. RoleManager is retired.

---

## 2) Final YAML

```yaml
log:
  project: fastvideo
  group: wangame
  name: self_forcing_causal_student_bidi_teacher
  wandb_mode: online

trainer:
  output_dir: outputs/wangame_self_forcing_refactor
  max_train_steps: 100000
  seed: 1000
  mixed_precision: bf16
  grad_accum_rounds: 1

data:
  _target_: fastvideo.distillation.utils.dataloader.ParquetDataConfig
  data_path: /path/to/wangame/parquet
  dataloader_num_workers: 4

validation:
  enabled: true
  dataset_file: examples/training/finetune/WanGame2.1_1.3b_i2v/validation_random.json
  every_steps: 100
  sampling_steps: [4]
  pipeline:
    sampler_kind: sde
    scheduler:
      _target_: fastvideo.models.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler
      flow_shift: 3

shared_context:
  _target_: fastvideo.distillation.models.wangame.WanGameSharedContext
  model_path: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000

models:
  student:
    _target_: fastvideo.distillation.models.wangame.WanGameCausalModel
    init_from: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true
  teacher:
    _target_: fastvideo.distillation.models.wangame.WanGameModel
    init_from: weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers
    trainable: false
  critic:
    _target_: fastvideo.distillation.models.wangame.WanGameModel
    init_from: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true

method:
  _target_: fastvideo.distillation.methods.distribution_matching.self_forcing.SelfForcingMethod
  rollout_mode: simulate
  chunk_size: 3
  student_sample_type: sde
  context_noise: 0.0
```

For a Wan T2V DMD2 run, only the `_target_` paths change:

```yaml
shared_context:
  _target_: fastvideo.distillation.models.wan.WanSharedContext
  model_path: /path/to/wan_14b

models:
  student:
    _target_: fastvideo.distillation.models.wan.WanModel
    init_from: /path/to/wan_14b
    trainable: true
  teacher:
    _target_: fastvideo.distillation.models.wan.WanModel
    init_from: /path/to/wan_14b
    trainable: false
  fake_score:
    _target_: fastvideo.distillation.models.wan.WanModel
    init_from: /path/to/wan_14b
    trainable: true

method:
  _target_: fastvideo.distillation.methods.distribution_matching.dmd2.DMD2Method
  guidance_scale: 5.0
  student_update_freq: 5
```

---

## 3) Class interfaces

### 3.1 SharedContextBase

```python
class SharedContextBase(ABC):
    """Holds preprocessing primitives and shared components (VAE, scheduler).
    Analogous to FastGenNetwork.init_preprocessors() — owns what all roles share.
    Does NOT own any transformer or implement predict_x0/predict_noise.
    """

    dataloader: Any
    validator: Any | None
    training_args: Any
    noise_scheduler: Any
    vae: Any

    @abstractmethod
    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Normalize latents, sample timesteps, build attn_metadata, populate
        batch.conditional_dict / batch.unconditional_dict. No transformer calls."""
        ...

    @abstractmethod
    def add_noise(
        self, clean: Tensor, noise: Tensor, timestep: Tensor
    ) -> Tensor: ...

    @abstractmethod
    def on_train_start(self) -> None: ...

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        return {}
```

### 3.2 ModelBase (per-role, analogous to FastGenNetwork)

```python
class ModelBase(ABC):
    """Per-role model. Analogous to FastGenNetwork in FastGen.
    Owns ONE transformer and implements forward ops on top of shared context.
    """

    transformer: nn.Module
    ctx: SharedContextBase  # injected at construction
    trainable: bool

    @abstractmethod
    def predict_noise(
        self,
        noisy_latents: Tensor,
        timestep: Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict | None = None,
        attn_kind: str = "dense",
    ) -> Tensor: ...

    @abstractmethod
    def predict_x0(self, ...) -> Tensor: ...

    def backward(
        self, loss: Tensor, ctx: Any, *, grad_accum_rounds: int
    ) -> None:
        """Default backward. Causal subclass may need forward-context restore."""
        (loss / max(1, grad_accum_rounds)).backward()


class CausalModelBase(ModelBase):
    """Causal/streaming extension. Analogous to CausalFastGenNetwork."""

    @abstractmethod
    def predict_noise_streaming(self, ...) -> Tensor | None: ...

    @abstractmethod
    def predict_x0_streaming(self, ...) -> Tensor | None: ...

    @abstractmethod
    def clear_caches(self, *, cache_tag: str = "pos") -> None: ...
```

### 3.3 DistillMethod (analogous to FastGenModel)

```python
class DistillMethod(nn.Module, ABC):
    """Base training method. Analogous to FastGenModel.
    Owns role model objects and ALL optimizers directly as attributes.
    RoleManager is retired.
    """

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        cfg: RunConfig,
        shared_context: SharedContextBase,
        role_models: dict[str, ModelBase],
    ) -> "DistillMethod":
        """Assemble the method. Analogous to FastGenModel.__init__ → build_model().
        The classmethod reads role_models, stores them as self.student / self.teacher /
        etc., then calls init_optimizers().
        """
        ...

    @abstractmethod
    def init_optimizers(self) -> None:
        """Build self.student_optimizer, self.teacher_optimizer, etc.
        Analogous to FastGenModel.init_optimizers() + DMD2Model.init_optimizers().
        Subclasses call super().init_optimizers() then add their own.
        """
        ...

    @abstractmethod
    def single_train_step(
        self, batch: TrainingBatch, iteration: int, **kwargs
    ) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, float]]: ...

    @abstractmethod
    def get_optimizers(self, iteration: int) -> list[Optimizer]: ...

    @abstractmethod
    def get_lr_schedulers(self, iteration: int) -> list[Any]: ...

    # Checkpoint helpers (mirror FastGen's model_dict / optimizer_dict / scheduler_dict)
    @property
    @abstractmethod
    def model_dict(self) -> dict[str, nn.Module]: ...

    @property
    @abstractmethod
    def optimizer_dict(self) -> dict[str, Optimizer]: ...

    @property
    @abstractmethod
    def scheduler_dict(self) -> dict[str, Any]: ...
```

### 3.4 SelfForcingMethod (concrete example)

```python
class SelfForcingMethod(DistillMethod):
    def __init__(
        self,
        student: CausalModelBase,
        teacher: ModelBase,
        critic: ModelBase,
        shared_context: SharedContextBase,
        cfg: dict,
    ) -> None:
        super().__init__()
        if not isinstance(student, CausalModelBase):
            raise TypeError(
                f"SelfForcingMethod requires CausalModelBase student, got {type(student).__name__}"
            )
        self.student = student
        self.teacher = teacher
        self.critic = critic
        self.ctx = shared_context
        self.init_optimizers()

    @classmethod
    def build(cls, *, cfg, shared_context, role_models):
        return cls(
            student=role_models["student"],
            teacher=role_models["teacher"],
            critic=role_models["critic"],
            shared_context=shared_context,
            cfg=cfg.method,
        )

    def init_optimizers(self) -> None:
        self.student_optimizer = build_optimizer(self.student.transformer, ...)
        self.student_lr_scheduler = build_lr_scheduler(self.student_optimizer, ...)
        self.critic_optimizer = build_optimizer(self.critic.transformer, ...)
        self.critic_lr_scheduler = build_lr_scheduler(self.critic_optimizer, ...)

    @property
    def model_dict(self):
        return {"student": self.student.transformer, "critic": self.critic.transformer}

    @property
    def optimizer_dict(self):
        return {"student": self.student_optimizer, "critic": self.critic_optimizer}

    @property
    def scheduler_dict(self):
        return {"student": self.student_lr_scheduler, "critic": self.critic_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list:
        return [self.student_optimizer, self.critic_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list:
        return [self.student_lr_scheduler, self.critic_lr_scheduler]
```

---

## 4) Assembly flow (dispatch / entrypoint)

```python
cfg = load_run_config(path)

# 1. SharedContext — VAE, scheduler, dataloader, validator
shared_context = instantiate(cfg.shared_context)

# 2. Per-role models — each owns one transformer + forward ops
#    shared_context is injected so each model can read batch fields it needs
role_models = {
    role: instantiate(model_cfg, shared_context=shared_context)
    for role, model_cfg in cfg.models.items()
}

# 3. Method class assembles itself from role models (analogous to FastGenModel.__init__)
method_cls = resolve_target(cfg.method["_target_"])
method = method_cls.build(
    cfg=cfg,
    shared_context=shared_context,
    role_models=role_models,
)

# 4. Generic trainer loop, not method-aware beyond the DistillMethod interface
trainer = Trainer(cfg.trainer)
trainer.run(method, shared_context=shared_context)
```

---

## 5) What is retired / changed

| Current | New |
|---|---|
| `dispatch.py` string registry (`_MODELS`, `_METHODS`, `@register_*`) | Deleted; `_target_` + `instantiate()` |
| `ModelBase` — one class handles ALL roles via `RoleHandle` arg | Retired; replaced by per-role `ModelBase` (one instance per role) |
| `CausalModelBase` | Becomes proper ABC for streaming, same concept |
| `RoleHandle` / `RoleManager` | Retired; method owns role objects directly |
| `DistillMethod.build(cfg, bundle, model, validator)` | `DistillMethod.build(cfg, shared_context, role_models)` |
| Optimizers on `RoleHandle.optimizers` | On method: `self.student_optimizer`, etc. |
| `recipe.family` / `recipe.method` YAML keys | Deleted; `shared_context._target_` + `method._target_` |
| `roles.*` YAML section | Replaced by `models.*` with `_target_`, `init_from`, `trainable` |
| Method checkpoint uses `bundle.roles` dict | Uses `model_dict`, `optimizer_dict`, `scheduler_dict` properties |

---

## 6) Naming cleanup (separate PR, do NOT mix in)

- `DistillMethod` → `Method`
- `DistillRunConfig` → `RunConfig`
- `load_distill_run_config` → `load_run_config`
- Entrypoint `distillation.py` → `training.py` (optional)

---

## 7) TODO (this phase — wangame first, then extend to wan)

**Infrastructure**

- [ ] `fastvideo/distillation/utils/instantiate.py`
  - `resolve_target(target: str) -> type`
  - `instantiate(cfg: dict, **extra) -> Any`

- [ ] `fastvideo/distillation/utils/config.py`
  - New `RunConfig` dataclass (no `RecipeSpec`, no `RoleSpec`)
  - New `load_run_config(path)` parser for new YAML schema
  - Keep `load_distill_run_config` as deprecated shim

**Base classes**

- [ ] `fastvideo/distillation/models/base.py`
  - New `SharedContextBase` ABC
  - New `ModelBase` ABC (per-role; `self.transformer`, `self.ctx`, `predict_noise`, `predict_x0`)
  - New `CausalModelBase(ModelBase)` ABC (streaming ops, `clear_caches`)
  - Retire old `ModelBase` once migration is done

- [ ] `fastvideo/distillation/roles.py`
  - Retire `RoleHandle` / `RoleManager` (keep temporarily if trainer checkpoint code needs it, then remove)

- [ ] `fastvideo/distillation/methods/base.py`
  - New `DistillMethod.build(cfg, shared_context, role_models)` signature
  - Add abstract `init_optimizers`, `model_dict`, `optimizer_dict`, `scheduler_dict`
  - Remove `bundle`, `model`, `validator` parameters

**WanGame models**

- [ ] `fastvideo/distillation/models/wangame/shared_context.py`
  - `WanGameSharedContext(SharedContextBase)`: extracts VAE, scheduler, `prepare_batch`,
    `add_noise`, `on_train_start`, dataloader, validator from current `WanGameModel.__init__`

- [ ] `fastvideo/distillation/models/wangame/models.py`
  - `WanGameModelBase(ModelBase)`: shared bidi forward logic (input_kwargs, CFG, action cond)
  - `WanGameModel(WanGameModelBase)`: bidi transformer
  - `WanGameCausalModel(CausalModelBase, WanGameModelBase)`: causal transformer + streaming ops
  - Transformer loading moves into each class `__init__` (replaces `common._build_wangame_role_handles`)

**Wan models**

- [ ] `fastvideo/distillation/models/wan/shared_context.py`
  - `WanSharedContext(SharedContextBase)`: same extraction from current `WanModel`

- [ ] `fastvideo/distillation/models/wan/models.py`
  - `WanModelBase(ModelBase)`: shared bidi forward logic (text conditioning, CFG)
  - `WanModel(WanModelBase)`: bidi transformer
  - `WanCausalModel(CausalModelBase, WanModelBase)`: causal transformer + streaming ops

**Methods**

- [ ] Update all method `build()` signatures to new interface:
  - `SelfForcingMethod`, `DMD2Method`, `FinetuneMethod`, `DFSFTMethod`

**Dispatch & configs**

- [ ] `fastvideo/distillation/dispatch.py`: replace body with new assembly flow (or delete)
- [ ] New YAML configs for wangame self-forcing (validate end-to-end)
- [ ] Update existing wangame YAML configs to new schema
- [ ] Update wan YAML configs
