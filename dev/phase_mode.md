# Phase：Model Plugin “去 Adapter 化”（命名收敛 + 去掉 ModelComponents 中间层）

> 本阶段只做 **结构与命名** 的收敛，目标是让代码读起来更直观、更少概念。
> 训练/验证行为应保持不变（同 YAML、同 loss、同 pipeline 选择）。
>
> 你提出的方向：`ModelBase / WanModel / WanGameModel`，模型构建由 `__init__`（或 `from_config`）自行管理，
> 从而不再需要 `ModelComponents` 这种“中间件”——我认为是合理且优雅的。

---

## 0) 背景：为什么现在 “Adapter” 这个词让人困惑

当前 `fastvideo/distillation/models/{wan,wangame}.py` 里所谓的 `*Adapter`，实际上已经承担了：

- family 相关的 runtime primitives（`prepare_batch/add_noise/predict_noise/predict_x0/backward/...`）
- shared components 的持有（`vae/noise_scheduler`）与生命周期管理
- 训练期状态（RNG、negative conditioning cache 等）

这更像“模型运行时封装（model plugin / runtime model）”，而不是一个“薄 adapter”。

同时 `ModelComponents` 只是把这些东西再打包一次，本质信息冗余。

---

## 1) 目标 / 非目标

### ✅ 目标（本阶段要做到）

1. **命名收敛**：彻底去掉 `Adapter` 这个词。
   - `ModelAdapter` → `ModelBase`
   - `WanAdapter` → `WanModel`
   - `WanGameAdapter` → `WanGameModel`
2. **去掉 build-time 中间件**：删除 `ModelComponents`（以及对应的 `models/components.py`）。
   - “构建产物”不再是 dataclass，而是 **模型对象本身**（它天然持有 bundle/dataloader/validator）。
3. **dispatch 更直觉**：
   - `@register_model("wan")` 直接注册 `WanModel` 类（或其 factory），而不是注册 `build_wan_components()` 这种函数。
   - `dispatch.build_runtime_from_config()` 变成：`model = WanModel(cfg)` → `method = ...build(..., model=model)`。
4. **保持 method 仍是 role-centric**：
   - method 仍通过 role（student/teacher/critic/…）决定算法逻辑；
   - model 只提供 operation-centric primitives，不因为 role 增多而出现“role 爆炸式 API”。

### ❌ 非目标（本阶段不做）

- 不改 YAML schema（保持现有 schema v2）。
- 不改算法行为（DMD2/finetune/dfsft 的 loss、rollout、optimizer cadence 不变）。
- 不改 pipeline/validator 选择逻辑（仍由 method/validator 按 config 决定）。
- 不引入 multi-family roles（先保持 `recipe.family` 主导一个 family）。

---

## 2) 新的核心抽象：ModelBase / WanModel / WanGameModel

### 2.1 `ModelBase`（替代 `ModelAdapter` + `ModelComponents`）

建议文件：`fastvideo/distillation/models/base.py`

`ModelBase` 是一个 **“模型插件对象”**，它既是构建产物，也是 runtime boundary。

它必须持有（构建期产物）：

- `training_args`
- `bundle: RoleManager`
- `dataloader`
- `validator`（可选）
- `start_step`（用于 resume）

同时提供（runtime primitives）：

- `num_train_timesteps`
- `shift_and_clamp_timestep(t)`
- `on_train_start()` / `get_rng_generators()`
- `prepare_batch(...)`
- `add_noise(...)`
- `predict_noise(handle, ...)`
- `predict_x0(handle, ...)`
- `backward(loss, ctx, grad_accum_rounds)`

> 说明：这基本就是现有 `ModelAdapter` 的抽象 + `ModelComponents` 的字段合并。

---

## 3) 代码改动设计（按文件列 TODO）

### 3.1 `fastvideo/distillation/models/adapter.py` → `.../models/base.py`

- [ ] 文件改名：`adapter.py` → `base.py`
- [ ] 类改名：`ModelAdapter` → `ModelBase`
- [ ] 文档/注释同步：强调这是 “model plugin object”，而非薄 adapter

### 3.2 `fastvideo/distillation/models/components.py` 删除

- [ ] 删除 `ModelComponents` dataclass
- [ ] 删除其在 dispatch / models 插件中的引用

### 3.3 `fastvideo/distillation/models/wan.py`

将结构从：

- `class WanAdapter(ModelAdapter): ...`
- `@register_model("wan") def build_wan_components(...) -> ModelComponents`

改成：

- `@register_model("wan") class WanModel(ModelBase): ...`

#### 建议实现形态

```py
@register_model("wan")
class WanModel(ModelBase):
    def __init__(self, *, cfg: DistillRunConfig) -> None:
        # 1) parse + validate cfg
        # 2) build shared components (vae/noise_scheduler)
        # 3) build roles -> RoleManager
        # 4) build validator (optional)
        # 5) build dataloader
        # 6) init runtime caches (rng / negative conditioning state)
```

把原本 `WanAdapter` 的方法体原封不动迁到 `WanModel` 上即可（第一版只做搬迁/改名）。

> 注意：`ensure_negative_conditioning()` 目前依赖 `prompt_handle`（student transformer + prompt encoding pipeline）。
> `WanModel` 仍可用 `self.student_handle = self.bundle.role("student")` 解决。

### 3.4 `fastvideo/distillation/models/wangame.py`

同 `wan.py`：

- `WanGameAdapter` → `WanGameModel`
- `build_wangame_components(...) -> ModelComponents` → `WanGameModel(cfg)`

需要保持：

- streaming validation 的 `num_frames` 约束（`1 + 4k` 且 latent 可被 `num_frame_per_block` 整除）
- validator pipeline 选择逻辑不变（parallel vs streaming / ode vs sde / ode_solver）

### 3.5 `fastvideo/distillation/dispatch.py`

当前：

- `_MODELS: dict[str, ModelBuilder]`（builder 返回 `ModelComponents`）

改为：

- `_MODELS: dict[str, type[ModelBase]]`（或 `Callable[[DistillRunConfig], ModelBase]`）

并把 `build_runtime_from_config()` 改为：

```py
model_cls = get_model(cfg.recipe.family)
model = model_cls(cfg=cfg)

method_cls = get_method(cfg.recipe.method)
method = method_cls.build(cfg=cfg, bundle=model.bundle, model=model, validator=model.validator)

return DistillRuntime(
    training_args=model.training_args,
    method=method,
    dataloader=model.dataloader,
    start_step=model.start_step,
)
```

> 这里建议把传参从 `adapter=` 改名为 `model=`，让含义更直观。

### 3.6 `fastvideo/distillation/methods/base.py`（以及所有 methods）

目标：把 method 里对 `self.adapter` 的依赖改成对 `self.model` 的依赖。

- [ ] `DistillMethod.build(...)` 签名建议改为：
  - `build(cfg, bundle, model, validator)`（或更简化：`build(cfg, model)`）
- [ ] methods 内部字段：
  - `self.model` 替代 `self.adapter`
- [ ] `on_train_start()` 里调用 `self.model.on_train_start()`
- [ ] `get_rng_generators()` 读取 `self.model.get_rng_generators()`

> 这一改动对行为应是 0 diff（只是字段名变化）。

### 3.7 文档与 “Config keys used” 头部更新

- [ ] `models/*.py` 的 “Config keys used” 头部同步改成 “consumed by WanModel/WanGameModel”
- [ ] 方法文件、validator 文件不强制改（但建议把 “adapter” 的文字替换为 “model”）

---

## 4) 兼容性与风险点（需要提前明确）

### 4.1 导入/注册顺序（避免循环 import）

要求：

- `dispatch.ensure_builtin_registrations()` 仍然显式 import `fastvideo.distillation.models.wan` 等模块，
  让 `@register_model` 在 import 时完成注册。
- `models/*.py` 只 import `register_model`（不要反向 import dispatch 里的 heavy objects）。

### 4.2 Checkpoint/RNG 断点续训

目前 checkpoint manager 通过：

- `runtime.method.get_rng_generators()`（优先）
- fallback 到 `runtime.method.adapter.get_rng_generators()`

重构后建议：

- `runtime.method.get_rng_generators()` 永远存在并返回 `self.model.get_rng_generators()`，
  不再做 adapter fallback。

### 4.3 行为不变（Definition of Done）

必须满足：

- `fastvideo/training/distillation.py --config <existing yaml>` 能跑通：
  - wan: DMD2 / finetune
  - wangame: finetune / dmd2 / dfsft（尤其 streaming validation）
- 静态检查至少通过：
  - `python -m py_compile`（相关文件）
  - `ruff check`（相关文件）

---

## 5) 实施顺序（推荐最小风险落地）

1) **纯改名 + 0 行为改动**
- 先在 models 内部把 `WanAdapter` 改名 `WanModel`（类名、注释、引用）
- `ModelAdapter` 改名 `ModelBase`

2) **去掉 ModelComponents**
- model plugin 不再返回 dataclass，而是返回模型对象本身
- dispatch 改为实例化模型对象

3) **methods 参数名收敛**
- `adapter` 全部改为 `model`

做到这一步就能得到一个更直觉的结构：

- `models/` 里就是“模型插件对象”
- `methods/` 只看 `model`（operation-centric）+ `bundle`（role-centric）
- dispatch 就是 “cfg -> model -> method -> trainer”

