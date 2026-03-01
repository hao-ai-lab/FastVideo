# `fastvideo/distillation/validators/__init__.py`

**目的**
- 统一导出 validator 接口与 Wan validator 实现。

**当前导出**
- `DistillValidator`：最小 validation 接口
- `ValidationRequest`：method 提供的 validation overrides
- `WanValidator`：Wan model plugin 的 validation 采样与记录实现
