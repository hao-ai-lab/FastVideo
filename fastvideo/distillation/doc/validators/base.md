# `fastvideo/distillation/validators/base.py`

**目的**
- 定义 distillation validator 的最小抽象接口。

**接口**
- `log_validation(step: int, request: ValidationRequest | None = None) -> None`

`ValidationRequest` 用于 method 覆盖关键采样配置（steps/guidance/output_dir 等），让 validator
保持 family-specific、但 method-agnostic。

`ValidationRequest.sample_handle` 用于由 method 明确指定“本次 validation 要采样哪个模型/权重”
（例如 student / student_ema / refiner / ...）。validator 不应自行 hardcode 角色语义。

**设计意图**
- trainer 只调用 `method.log_validation(step)`。
- method 决定是否做 validation，并把 `ValidationRequest` 传给 family-specific validator。
