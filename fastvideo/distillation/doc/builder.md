# `fastvideo/distillation/builder.py`

**目的**
- 把 YAML config（`DistillRunConfig`）装配成一个可运行的 `DistillRuntime`：
  - `model plugin` 负责 build-time 产物（`FamilyComponents`）
  - `method` 负责算法（`DistillMethod`）

**关键 API**
- `build_runtime_from_config(cfg) -> DistillRuntime`
  - `model_builder = registry.get_model(cfg.recipe.family)`
  - `method_builder = registry.get_method(cfg.recipe.method)`
  - `method = method_builder(cfg=cfg, bundle=components.bundle, adapter=components.adapter, validator=components.validator)`

**边界**
- ✅ 这里不写 `if model==... and method==...` 的 N×M 组合逻辑。
- ✅ 这里只做“装配”，不包含训练 loop / loss / rollout / optimizer policy。
