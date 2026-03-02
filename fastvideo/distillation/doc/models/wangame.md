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
   - 支持 role-level transformer 变体（通过 `RoleSpec.extra`）：
     - `roles.<role>.variant: bidirectional` → `WanGameActionTransformer3DModel`
     - `roles.<role>.variant: causal` → `CausalWanGameActionTransformer3DModel`
   - 根据 `RoleSpec.trainable` 设置 `requires_grad`
   - 可选开启 activation checkpoint（仅对 trainable role）
3) **构建 bundle / dataloader / validator**
   - `bundle = RoleManager(roles=role_handles)`
   - dataloader：parquet + `pyarrow_schema_wangame`
   - validator（可选）：`WanGameValidator`（当 `training.validation.enabled=true`，或 `training.validation` 非空）
   - runtime primitives 由 `WanGameModel` 直接实现（不再额外分一层 `*Adapter` 类/文件）

**关于 roles.family**
- 当前 `wangame` plugin 只支持 `family="wangame"` 的 role。
  这让 build-time 逻辑保持高内聚：模型加载、batch schema 与 primitives 能保持一致。
