# `fastvideo/distillation/dispatch.py`

**目的**
- 把 “可扩展注册（models/methods）” 与 “从 YAML 装配可运行 runtime” 收敛到一个入口文件：
  - 新增 model 的成本 ≈ N
  - 新增 method 的成本 ≈ M
  - 不需要写 N×M 的 if/else 组合逻辑

**关键概念**
- `ModelBuilder(cfg) -> ModelComponents`
- `MethodBuilder(cfg, bundle, adapter, validator) -> DistillMethod`

**关键 API**
- `register_model(name)` / `register_method(name)`：装饰器注册
- `get_model(name)` / `get_method(name)`：查询（会触发内置注册）
- `available_models()` / `available_methods()`
- `build_runtime_from_config(cfg) -> DistillRuntime`
  - 选择 model plugin：`get_model(cfg.recipe.family)`
  - 选择 method：`get_method(cfg.recipe.method)`

**边界**
- ✅ 这里只做“装配 + dispatch”，不包含训练 loop / loss / rollout / optimizer policy。
- ✅ method 层保持算法高内聚；model plugin 层保持集成高内聚。

