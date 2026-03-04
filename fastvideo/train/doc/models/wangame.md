# `fastvideo/distillation/models/wangame.py`

**定位**
- `@register_model("wangame")` 的 model plugin（实现：`WanGameModel(cfg)`）：
  - 负责把 YAML config 装配成一个可运行的 `WanGameModel`（同时实现 `ModelBase` primitives）
  - 包含 WanGame 特有的模块加载、dataset schema、validator 选择等逻辑

**产物**
- `WanGameModel` 实例（关键字段）：
  - `training_args`, `bundle`, `dataloader`, `validator`, `start_step`
  - 以及 `ModelBase` 的 primitives（`prepare_batch/add_noise/predict_*/backward/...`）

**主要职责**
1) **加载 shared components**
   - `vae`：从 `training.model_path`（默认 student.path）加载
   - `noise_scheduler`：`FlowMatchEulerDiscreteScheduler(shift=flow_shift)`
2) **按 roles 加载 transformer 模块**
   - 对每个 role：加载 `transformer`（可选 `transformer_2`）
   - role-level transformer 类型由 `roles.<role>.family` 决定（而不是 `variant`）：
     - `roles.<role>.family: wangame` → `WanGameActionTransformer3DModel`
     - `roles.<role>.family: wangame_causal` → `CausalWanGameActionTransformer3DModel`
   - 根据 `RoleSpec.trainable` 设置 `requires_grad`
   - 可选开启 activation checkpoint（仅对 trainable role）
3) **构建 bundle / dataloader / validator**
   - `bundle = RoleManager(roles=role_handles)`
   - dataloader：parquet + `pyarrow_schema_wangame`
   - validator（可选）：`WanGameValidator`（当 `training.validation.enabled=true`，或 `training.validation` 非空）
   - runtime primitives 由 `WanGameModel` 直接实现（不再额外分一层 `*Adapter` 类/文件）

**关于 roles.family**
- `recipe.family: wangame`（bidirectional）：
  - 只支持 `roles.<role>.family="wangame"`，否则直接报错。
- `recipe.family: wangame_causal`（causal-capable）：
  - 支持 `roles.<role>.family in {"wangame","wangame_causal"}`，用于 self-forcing
    等场景下的 “student causal + teacher/critic bidirectional” 组合。
