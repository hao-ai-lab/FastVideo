# `fastvideo/distillation/registry.py`

**目的**
- 为 distillation 的 “family / method” 提供轻量 registry：
  - 新增 family 的成本 ≈ N
  - 新增 method 的成本 ≈ M
  - build 组合不需要写 N×M 的 if/else

**关键概念**
- `FamilyBuilder(cfg) -> FamilyArtifacts`
- `MethodBuilder(cfg, bundle, adapter, validator) -> DistillMethod`

**关键 API**
- `register_family(name)` / `register_method(name)`：装饰器注册
- `get_family(name)` / `get_method(name)`：查询（会触发内置注册）
- `ensure_builtin_registrations()`：显式 import 内置实现，避免 import 顺序隐式 bug
- `available_families()` / `available_methods()`

**扩展方式**
- 新增 family：实现 `fastvideo/distillation/models/<name>.py` 并用 `@register_family("<name>")`
- 新增 method：实现 `fastvideo/distillation/methods/...` 并用 `@register_method("<name>")`
