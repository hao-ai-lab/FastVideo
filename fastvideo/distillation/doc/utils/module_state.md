# `fastvideo/distillation/utils/module_state.py`

**目的**
- 提供最小且通用的 module 训练状态设置，避免 model plugin 里到处散落：
  - `requires_grad_(...)`
  - `train()` / `eval()`

**当前包含**
- `apply_trainable(module, trainable: bool)`

**边界**
- ✅ 不涉及 optimizer/scheduler（由 method 管理）。
- ✅ 不涉及激活检查点策略（由 model plugin 在加载后按需启用）。

