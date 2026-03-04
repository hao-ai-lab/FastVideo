# Phase: Refactor — `_target_` instantiate-first, drop the string registry

## 0) FastGen hierarchy (the reference)

```
FastGenNetwork (ABC, nn.Module)       ← per-role backbone; owns transformer + noise_scheduler
CausalFastGenNetwork (ABC, nn.Module) ← parallel root, NOT a subclass of FastGenNetwork

FastGenModel (nn.Module)              ← base method; owns self.net, self.teacher, optimizers
    DMD2Model                         ← adds self.fake_score, self.discriminator
    SelfForcingModel                  ← causal rollout
    SFTModel
```

**Key lessons:**

1. **`CausalFastGenNetwork` is parallel to `FastGenNetwork`**, not a subclass.

2. **Every role instance owns its own `noise_scheduler`** — student and teacher each get
   an independent instance of the same schedule. No sharing needed. This is because
   `FastGenNetwork.__init__` always calls `self.set_noise_schedule()`.
   The teacher's `forward(x_t, t, fwd_pred_type="x0")` uses its own scheduler internally
   for `pred_noise → x0` conversion. No "who owns the scheduler" problem.

3. **No mixin class.** FastGen doesn't need one because student and teacher are the
   **same class** (`WanNetwork`). The only construction-time difference is that
   `FastGenModel.build_model()` calls `self.net.init_preprocessors()` on the student,
   but NOT on the teacher. The class itself is neutral — it has `init_preprocessors()`
   as a method, but whether it's called is the caller's decision.
   For us: same principle. Every `ModelBase` has `init_preprocessors()`. The method's
   `build()` calls it only on the student. No mixin. No `is_student` flag.

4. **Method owns all optimizers** (`self.net_optimizer`, `self.fake_score_optimizer`, …)
   via `init_optimizers()`. Subclasses extend with `super().init_optimizers()`.

5. **`model_dict` / `optimizer_dict` / `scheduler_dict`** are properties on the method,
   used by the trainer for checkpointing.

6. **Dataloader is external** — passed into trainer separately, not owned by the model.

---

## 1) Our hierarchy

```
fastvideo/distillation/
├── models/
│   ├── base.py                        ModelBase, CausalModelBase (parallel roots, no mixin)
│   ├── wan/
│   │   └── model.py                   WanModel(ModelBase)
│   └── wangame/
│       ├── model.py                   WanGameModel(ModelBase)
│       └── model_causal.py            WanGameCausalModel(CausalModelBase)
├── methods/
│   ├── base.py                        DistillMethod (analogous to FastGenModel)
│   └── distribution_matching/
│       ├── dmd2.py                    DMD2Method
│       └── self_forcing.py            SelfForcingMethod
└── utils/
    └── instantiate.py                 resolve_target, instantiate
```

**Every model instance has its own `noise_scheduler`** (constructed in `__init__`).
VAE and `prepare_batch` are initialized via `init_preprocessors()`, called only on
the student by `DistillMethod.build()`. No mixin. No flag.

**DistillMethod** owns role model objects (`self.student`, `self.teacher`, …) and
all optimizers as attributes. `RoleManager` is retired.

---

## 2) Class interfaces

### 2.1 ModelBase and CausalModelBase

`CausalModelBase` is a **parallel root** to `ModelBase`, mirroring FastGen's design.
Each concrete method slot accepts exactly one expected type — there is no legitimate
use-case where a `fake_score` or `teacher` slot would accept either kind
interchangeably. Parallel roots keep the contracts clean: `ModelBase` for
bidirectional models, `CausalModelBase` for streaming/causal models.

```python
# fastvideo/distillation/models/base.py

class ModelBase(ABC):
    """Per-role model. Every instance owns its own noise_scheduler.
    init_preprocessors() is only called on the student by DistillMethod.build().
    """
    transformer: nn.Module
    noise_scheduler: Any  # always set in __init__

    def init_preprocessors(self, training_args: Any) -> None:
        """Load VAE, seed RNGs. Analogous to FastGenNetwork.init_preprocessors().
        Only called on the student. Default: no-op (teacher/critic skip this).
        """
        pass

    def on_train_start(self) -> None:
        pass

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        return {}

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: str = "data",
    ) -> TrainingBatch:
        raise NotImplementedError(
            f"{type(self).__name__}.prepare_batch() requires init_preprocessors() "
            "(student only)."
        )

    def add_noise(self, clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        raise NotImplementedError

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

    def backward(self, loss: Tensor, ctx: Any, *, grad_accum_rounds: int = 1) -> None:
        (loss / max(1, grad_accum_rounds)).backward()


class CausalModelBase(ABC):
    """Parallel root for causal/streaming models. NOT a subclass of ModelBase.
    Mirrors FastGen's design: CausalFastGenNetwork is parallel to FastGenNetwork.
    Concrete method slots that need a causal model declare CausalModelBase explicitly.
    """
    transformer: nn.Module
    noise_scheduler: Any  # always set in __init__

    def init_preprocessors(self, training_args: Any) -> None:
        pass

    def on_train_start(self) -> None:
        pass

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        return {}

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: str = "data",
    ) -> TrainingBatch:
        raise NotImplementedError(
            f"{type(self).__name__}.prepare_batch() requires init_preprocessors() "
            "(student only)."
        )

    def add_noise(self, clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_noise(self, noisy_latents: Tensor, timestep: Tensor, batch: TrainingBatch, *, conditional: bool, cfg_uncond: dict | None = None, attn_kind: str = "dense") -> Tensor: ...

    @abstractmethod
    def predict_x0(self, ...) -> Tensor: ...

    def backward(self, loss: Tensor, ctx: Any, *, grad_accum_rounds: int = 1) -> None:
        (loss / max(1, grad_accum_rounds)).backward()

    @abstractmethod
    def predict_noise_streaming(
        self,
        noisy_latents: Tensor,
        timestep: Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict | None = None,
        attn_kind: str = "dense",
    ) -> Tensor | None: ...

    @abstractmethod
    def predict_x0_streaming(self, ...) -> Tensor | None: ...

    @abstractmethod
    def clear_caches(self, *, cache_tag: str = "pos") -> None: ...
```

### 2.2 Dynamic config — `__init__` signature as schema

We abandon structured config dataclasses (no `DistillRunConfig`, no `RecipeSpec`,
no `attrs` configs). Instead, each `_target_` class declares its own schema via its
`__init__` type annotations. The `instantiate()` utility:

1. Pops `_target_` and resolves the class.
2. Passes the **entire remaining dict** into the constructor — each class takes what
   it needs via `**kwargs` and ignores the rest.

This is the "whole config dict flows through" model: callers don't need to know which
keys a constructor cares about, and adding a field to a class means adding it to
`__init__` — no schema file to maintain in parallel.

```python
# fastvideo/distillation/utils/instantiate.py

import importlib
from typing import Any


def resolve_target(target: str) -> type:
    module_path, cls_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def instantiate(cfg: dict[str, Any], /, **extra: Any) -> Any:
    """_target_ instantiation: pop _target_, resolve class, pass everything else through.

    Each constructor accepts what it needs via explicit params + **kwargs for the rest.
    No signature inspection — simplicity over strict validation.
    """
    cfg = dict(cfg)
    target_str = cfg.pop("_target_")
    cls = resolve_target(target_str)
    return cls(**{**cfg, **extra})
```

**Each class is self-documenting via its explicit params.** Example:

```python
class WanGameModel(ModelBase):
    def __init__(
        self,
        *,
        init_from: str,           # required — TypeError at startup if missing from YAML
        trainable: bool = True,   # optional with default
        **kwargs,                 # absorbs any other keys in the config dict
    ) -> None: ...
```

### 2.4 Concrete model classes (WanGame example)

```python
# fastvideo/distillation/models/wangame/model.py

class WanGameModel(ModelBase):
    """Bidirectional WanGame model. Can be student or teacher/critic.
    Always builds noise_scheduler in __init__.
    VAE and prepare_batch are activated by init_preprocessors() — only for student.
    """

    def __init__(self, *, init_from: str, trainable: bool = True, **kwargs) -> None:
        self.transformer = load_transformer(init_from, cls="WanGameActionTransformer3DModel")
        apply_trainable(self.transformer, trainable=trainable)
        # noise_scheduler always built — same as FastGenNetwork.__init__ calling set_noise_schedule()
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(...)
        # preprocessor state — inactive until init_preprocessors() is called
        self.vae: Any | None = None
        self._noise_gen_cpu: torch.Generator | None = None
        self._noise_gen_cuda: torch.Generator | None = None
        self._init_from = init_from

    def init_preprocessors(self, training_args: Any) -> None:
        """Activate as student. Analogous to FastGenNetwork.init_preprocessors()."""
        self.training_args = training_args
        self.vae = load_module_from_path(self._init_from, module_type="vae", ...)
        self.device = get_local_torch_device()
        self._init_timestep_mechanics()

    def on_train_start(self) -> None:
        seed = self.training_args.seed
        self._noise_gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
        self._noise_gen_cuda = torch.Generator(device=self.device).manual_seed(seed)

    def prepare_batch(self, raw_batch, *, ...) -> TrainingBatch:
        if self.vae is None:
            raise RuntimeError("prepare_batch requires init_preprocessors() (student only)")
        # ... full WanGame batch preparation
        return training_batch

    def add_noise(self, clean, noise, timestep) -> Tensor:
        return self.noise_scheduler.add_noise(...)

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, ...) -> Tensor:
        # Uses self.noise_scheduler (always available) and self.transformer
        with autocast(...), set_forward_context(...):
            kwargs = self._build_input_kwargs(noisy_latents, timestep, batch, conditional=conditional, ...)
            return self.transformer(**kwargs).permute(0, 2, 1, 3, 4)

    def predict_x0(self, noisy_latents, timestep, batch, *, conditional, ...) -> Tensor:
        pred_noise = self.predict_noise(noisy_latents, timestep, batch, conditional=conditional, ...)
        # Uses self.noise_scheduler — always available, no dependency on student
        return pred_noise_to_pred_video(pred_noise, noisy_latents, timestep, self.noise_scheduler)



# fastvideo/distillation/models/wangame/model_causal.py

class WanGameCausalModel(CausalModelBase):
    """Causal WanGame. Parallel root to WanGameModel (not a subclass).
    Same init_preprocessors() pattern — called only on student.
    """
    def __init__(self, *, init_from: str, trainable: bool = True, **kwargs) -> None:
        self.transformer = load_transformer(init_from, cls="CausalWanGameTransformer")
        apply_trainable(self.transformer, trainable=trainable)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(...)
        self.vae = None
        ...

    def init_preprocessors(self, training_args: Any) -> None:
        # Same pattern as WanGameModel.init_preprocessors()
        ...

    def predict_noise_streaming(self, ...) -> Tensor | None: ...
    def predict_x0_streaming(self, ...) -> Tensor | None: ...
    def clear_caches(self, *, cache_tag: str = "pos") -> None: ...
```

### 2.5 DistillMethod (analogous to FastGenModel)

```python
class DistillMethod(nn.Module, ABC):

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        cfg: RunConfig,
        role_models: dict[str, ModelBase | CausalModelBase],
        validator: Any | None,
    ) -> "DistillMethod":
        """Assemble the method. Calls init_preprocessors() on the student,
        then calls init_optimizers().
        Analogous to FastGenModel.__init__ → build_model() → init_preprocessors().
        Dataloader is NOT passed here — it is owned by the trainer.
        """
        ...

    @abstractmethod
    def init_optimizers(self) -> None: ...

    @abstractmethod
    def single_train_step(self, raw_batch, iteration, **kwargs) -> ...: ...

    @abstractmethod
    def get_optimizers(self, iteration: int) -> list[Optimizer]: ...

    @abstractmethod
    def get_lr_schedulers(self, iteration: int) -> list[Any]: ...

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

### 2.6 SelfForcingMethod (example)

```python
class SelfForcingMethod(DistillMethod):

    @classmethod
    def build(cls, *, cfg, role_models, validator):
        student = role_models["student"]
        teacher = role_models["teacher"]
        critic  = role_models["critic"]

        if not isinstance(student, CausalModelBase):
            raise TypeError(
                f"SelfForcingMethod requires CausalModelBase student, "
                f"got {type(student).__name__}"
            )

        # Call init_preprocessors only on student — identical to FastGen's build_model()
        # calling init_preprocessors() only on self.net, not self.teacher.
        student.init_preprocessors(cfg.training_args)

        return cls(
            student=student, teacher=teacher, critic=critic,
            validator=validator, cfg=cfg.method,
        )

    def __init__(self, student, teacher, critic, validator, cfg):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.critic = critic
        # No self.dataloader — trainer owns it and passes raw_batch into single_train_step.
        self.validator = validator
        self.init_optimizers()

    def init_optimizers(self) -> None:
        self.student_optimizer = build_optimizer(self.student.transformer, ...)
        self.student_lr_scheduler = build_scheduler(self.student_optimizer, ...)
        self.critic_optimizer = build_optimizer(self.critic.transformer, ...)
        self.critic_lr_scheduler = build_scheduler(self.critic_optimizer, ...)

    def single_train_step(self, raw_batch, iteration, **kwargs):
        batch = self.student.prepare_batch(raw_batch, ...)           # student owns this
        noisy = self.student.add_noise(batch.latents, batch.noise, batch.timesteps)

        student_x0 = self.student.predict_x0(noisy, batch.timesteps, batch, ...)
        teacher_x0 = self.teacher.predict_x0(noisy, batch.timesteps, batch, ...)
        # teacher uses its OWN noise_scheduler for pred_noise→x0 — no sharing needed
        ...

    @property
    def model_dict(self):
        return {"student": self.student.transformer, "critic": self.critic.transformer}

    @property
    def optimizer_dict(self):
        return {"student": self.student_optimizer, "critic": self.critic_optimizer}

    @property
    def scheduler_dict(self):
        return {"student": self.student_lr_scheduler, "critic": self.critic_lr_scheduler}
```

---

## 3) Final YAML

### WanGame Self-Forcing

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

No `is_student` flag in YAML. The method's `build()` decides which role gets
`init_preprocessors()`.

### Wan DMD2

```yaml
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

## 4) Assembly flow

```python
cfg = load_run_config(path)

# 1. Build per-role models. No init_preprocessors() here — that is the method's job.
#    Analogous to FastGenModel instantiating self.net and self.teacher as raw networks.
role_models = {
    role: instantiate(model_cfg)  # just transformer + noise_scheduler
    for role, model_cfg in cfg.models.items()
}

# 2. Dataloader is owned by the trainer. Build it here for trainer use only.
dataloader = instantiate(cfg.data, training_args=training_args)
validator   = build_validator(cfg.validation) if cfg.validation.enabled else None

# 3. Method build() calls init_preprocessors() on student, then init_optimizers().
#    No dataloader passed — trainer passes raw_batch into single_train_step each step.
#    Analogous to FastGenModel.__init__ → build_model() → init_preprocessors().
method_cls = resolve_target(cfg.method["_target_"])
method = method_cls.build(
    cfg=cfg,
    role_models=role_models,
    validator=validator,
)

# 4. Trainer runs the loop. Trainer is the sole owner of the dataloader.
trainer = DistillTrainer(training_args)
trainer.run(method, dataloader=dataloader, max_steps=cfg.trainer.max_train_steps, ...)
```

---

## 5) What is retired / changed

| Current | New |
|---|---|
| `dispatch.py` string registry | Deleted; `_target_` + `instantiate()` |
| `ModelBase` — dispatches via `RoleHandle` arg | Retired; per-role instance owns one transformer |
| `CausalModelBase(ModelBase)` subclass | `CausalModelBase(ABC)` parallel root — mirrors FastGen |
| `RoleHandle` / `RoleManager` | Retired; method owns `self.student`, `self.teacher`, etc. |
| Optimizers on `RoleHandle.optimizers` | On method: `self.student_optimizer`, etc. |
| `DistillMethod.build(cfg, bundle, model, validator)` | `DistillMethod.build(cfg, role_models, validator)` — no dataloader |
| `recipe.family` / `recipe.method` YAML | Deleted; `models.<role>._target_` + `method._target_` |
| `roles.*` YAML section | `models.*` with `_target_`, `init_from`, `trainable` |
| `is_student` / mixin / SharedContext | None of these. `build()` calls `init_preprocessors()` on student. |

---

## 6) Naming cleanup (separate PR)

- `DistillMethod` → `Method`
- `DistillRunConfig` → `RunConfig`
- Entrypoint `distillation.py` → `training.py` (optional)

---

## 7) TODO (wangame first, then wan)

**Infrastructure**

- [ ] `fastvideo/distillation/utils/instantiate.py`
  - `resolve_target(target: str) -> type`
  - `instantiate(cfg: dict, **extra) -> Any`
  - Uses `inspect.signature()` for field validation and filtering (see §2.5)

- [ ] `fastvideo/distillation/utils/config.py`
  - New `RunConfig` dataclass
  - New `load_run_config(path)` parser
  - Keep `load_distill_run_config` as deprecated shim

**Base classes**

- [ ] `fastvideo/distillation/models/base.py`
  - New `ModelBase` ABC with `init_preprocessors` (no-op default), `prepare_batch` (raises), `add_noise` (raises)
  - New `CausalModelBase` ABC (parallel root, same pattern + streaming methods)
  - Retire old `ModelBase` / `CausalModelBase`

- [ ] `fastvideo/distillation/roles.py` — retire `RoleHandle` / `RoleManager`

- [ ] `fastvideo/distillation/methods/base.py`
  - New `DistillMethod.build(cfg, role_models, dataloader, validator)` signature
  - Add abstract `init_optimizers`, `model_dict`, `optimizer_dict`, `scheduler_dict`

**WanGame**

- [ ] `fastvideo/distillation/models/wangame/model.py`
  - `WanGameModel(ModelBase)`: always builds `noise_scheduler`; `init_preprocessors()` loads VAE

- [ ] `fastvideo/distillation/models/wangame/model_causal.py`
  - `WanGameCausalModel(CausalModelBase)`: same pattern + streaming ops

**Wan**

- [ ] `fastvideo/distillation/models/wan/model.py`
  - `WanModel(ModelBase)`: always builds `noise_scheduler`; `init_preprocessors()` loads VAE + negative prompt

**Methods**

- [ ] Update all `build()` signatures: `SelfForcingMethod`, `DMD2Method`, `FinetuneMethod`, `DFSFTMethod`
- [ ] `DMD2Method.build()`: calls `init_preprocessors()` on student; `init_optimizers()` adds `fake_score_optimizer`

**Dispatch & configs**

- [ ] `fastvideo/distillation/dispatch.py`: replace with new assembly flow (or delete)
- [ ] New YAML for wangame self-forcing; validate end-to-end
- [ ] Migrate existing wangame and wan YAML configs
