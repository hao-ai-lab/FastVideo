# `fastvideo/distillation/models/wangame.py`

**定位**
- `@register_model("wangame")` 的 build-time 插件（实现：`build_wangame_components(...)`）：
  - 负责把 YAML config 装配成 `ModelComponents`
  - 包含 WanGame 特有的模块加载、dataset schema、adapter/validator 选择等逻辑

**产物**
- `ModelComponents(training_args, bundle, adapter, dataloader, validator, start_step)`

**主要职责**
1) **加载 shared components**
   - `vae`：从 `training.model_path`（默认 student.path）加载
   - `noise_scheduler`：`FlowMatchEulerDiscreteScheduler(shift=flow_shift)`
2) **按 roles 加载 transformer 模块**
   - 对每个 role：加载 `transformer`（可选 `transformer_2`）
   - 支持 role-level transformer 变体（通过 `RoleSpec.extra`）：
     - `roles.<role>.variant: bidirectional` → `WanGameActionTransformer3DModel`
     - `roles.<role>.variant: causal` → `CausalWanGameActionTransformer3DModel`
   - 根据 `RoleSpec.trainable` 设置 `requires_grad`
   - 可选开启 activation checkpoint（仅对 trainable role）
3) **构建 bundle / adapter / dataloader / validator**
   - `bundle = RoleManager(roles=role_handles)`
   - `adapter = WanGameAdapter(...)`
   - dataloader：parquet + `pyarrow_schema_wangame`
   - validator（可选）：`WanGameValidator`（当 `training.validation.enabled=true`，或 `training.validation` 非空）

**关于 roles.family**
- 当前 `wangame` plugin 只支持 `family="wangame"` 的 role。
  这让 build-time 逻辑保持高内聚：模型加载、batch schema 与 adapter 能保持一致。
