# `fastvideo/distillation/__init__.py`

**目的**
- 提供 distillation 子系统的最小“公共入口”，避免上层到处 import 内部实现细节。

**当前导出**
- `DistillTrainer`：训练 loop（infra only）
- `ModelBundle` / `RoleHandle`：multi-role 模型与训练状态容器

**设计意图**
- 让上层（例如 `fastvideo/training/distillation.py`）只依赖稳定 API：
  - `DistillTrainer.run(method, dataloader, ...)`
  - `bundle.role("student")` 等 role 访问模式

