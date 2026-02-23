# `fastvideo/distillation/methods/fine_tuning/__init__.py`

**状态**
- 当前是占位目录（`__all__ = []`），用于未来加入 finetuning（可视为只有 student 的特殊训练 recipe）。

**与 distillation 框架的关系**
- finetune 可以复用同一套：
  - YAML config（models 里只提供 student）
  - family（加载 student 模块与数据）
  - trainer（infra）
- method 则实现：
  - loss（SFT/finetune objective）
  - optimizer/scheduler policy

