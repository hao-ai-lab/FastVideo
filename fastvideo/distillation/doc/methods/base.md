# `fastvideo/distillation/methods/base.py`

**定位**
- `DistillMethod` 是算法层抽象：
  - 负责实现 `single_train_step()`（loss 构造）
  - 负责定义 update policy（哪些 optimizer/scheduler 在何时 step）
  - 不负责训练 loop（由 `DistillTrainer` 承担）

**关键点**
- `DistillMethod` 持有 `bundle: RoleManager`，并把所有 role 的 modules 放进
  `self.role_modules: ModuleDict`，便于 DDP/FSDP/ckpt 系统统一发现参数。

**需要子类实现的抽象方法**
- `single_train_step(batch, iteration, current_vsa_sparsity=...)`
  - 返回：`(loss_map, outputs, metrics)`
    - `loss_map: dict[str, Tensor]`：必须包含 `total_loss`（用于 backward）
    - `metrics: dict[str, scalar]`：额外要 log 的标量（float/int/0-dim Tensor）
- `get_optimizers(iteration)`
- `get_lr_schedulers(iteration)`

**默认实现**
- `backward()`：对 `loss_map["total_loss"]` 做 backward（子类可覆写以处理多 ctx）
- `optimizers_schedulers_step()`：按 `get_optimizers/get_lr_schedulers` 的结果 step
- `optimizers_zero_grad()`：对当前 iteration 的 optimizers 清梯度
