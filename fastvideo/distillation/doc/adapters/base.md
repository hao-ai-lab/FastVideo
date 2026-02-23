# `fastvideo/distillation/adapters/base.py`

**目的**
- 定义 adapter 的最小抽象接口。

**当前最小契约**
- `prepare_batch(raw_batch, current_vsa_sparsity=...) -> TrainingBatch`

**为什么接口这么小？**
- adapter 的“完整能力”通常是 **method-specific protocol**（duck typing），例如：
  - `predict_x0(handle, ...)`
  - `predict_noise(handle, ...)`
  - `add_noise(...)`
  - `backward(loss, ctx, ...)`
- 这些能力应由具体 method 在自己的 `Protocol` 里声明（例如 `DMD2Method` 的 `_DMD2Adapter`），
  从而保持 adapter 的基类稳定、method 的需求显式可读。

