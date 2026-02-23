# `fastvideo/distillation/bundle.py`

**目的**
- 用统一的数据结构表达 “多角色（roles）参与的训练/蒸馏”：
  - roles 是字符串 key（`"student"`, `"teacher"`, `"critic"`, `"reward"`, ...）
  - 每个 role 对应一个 `RoleHandle`

**关键类型**
- `RoleHandle`
  - `modules: dict[str, nn.Module]`：该 role 持有的模块（例如 `transformer`）
  - `optimizers: dict[str, Optimizer]` / `lr_schedulers: dict[str, Any]`
  - `trainable: bool`
  - `require_module(name)`：强制获取模块（缺失则报错）
- `ModelBundle`
  - `roles: dict[str, RoleHandle]`
  - `require_roles([...])`：method 在构造时校验依赖的 role 是否齐全
  - `role(name)`：获取 handle

**Phase 2.9 约定**
- family 负责 **load modules + 设置 trainable** 并创建 `ModelBundle`
- method 负责 **(按算法) 创建 optimizers/schedulers** 并写回对应的 `RoleHandle`

