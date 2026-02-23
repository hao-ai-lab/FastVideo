# `fastvideo/distillation/trainer.py`

**目的**
- 提供与算法无关的训练 loop（infra only），把 “怎么训练” 固定下来，把
  “训练什么（loss/rollout/update policy）” 留给 method。

**关键类型**
- `DistillTrainer`
  - `run(method, dataloader, max_steps, ...)`
  - 支持：
    - grad accumulation
    - tracker logging（rank0）
    - validation hook（`method.log_validation(step)`）
    - checkpoint hook（通过 `checkpoint_manager` 注入）

**与 Method 的契约**
`run()` 通过 duck-typing 调用（存在则调用）：
- `method.on_train_start()`
- `method.single_train_step(batch, step, current_vsa_sparsity=...)`
- `method.backward(loss_map, outputs, grad_accum_rounds=...)`
- `method.optimizers_schedulers_step(step)`
- `method.optimizers_zero_grad(step)`

**重要边界**
- trainer 不应知道 roles（student/teacher/critic/...）也不应知道具体算法；
  optimizer cadence、multi-optimizer 更新策略都应由 method 决定并暴露为 hook。

