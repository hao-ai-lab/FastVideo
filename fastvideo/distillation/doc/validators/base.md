# `fastvideo/distillation/validators/base.py`

**目的**
- 定义 distillation validator 的最小抽象接口。

**接口**
- `log_validation(step: int) -> None`

**设计意图**
- trainer 与 method 不需要知道 validator 的实现细节：
  - adapter 可以持有 validator，并在合适的时机调用
  - 或 method/trainer 通过 hook 调用 `log_validation()`

