# `fastvideo/distillation/methods/knowledge_distillation/__init__.py`

**状态**
- 当前是占位目录（`__all__ = []`），用于未来加入经典 KD 类方法（logit/feature matching 等）。

**期望的扩展方式**
- 新增 KD method 时：
  - method 定义需要哪些 roles（student/teacher/aux_teacher/...）
  - family 只负责加载这些 roles 的 modules 并构建 bundle/adapter
  - optimizer/scheduler 由 method 创建并写回 handle

