# `fastvideo/distillation/checkpoint.py`

**目的**
- Phase 2 的 role-based checkpoint/save-resume 管理：
  - 按 role 保存/恢复 modules、optimizers、schedulers
  - 可选保存 dataloader 状态（如果 dataloader 是 stateful）
  - 保存 RNG（全局 RNG + method 暴露的额外 generators，例如 adapter/validator 的 RNG）

**关键类型**
- `DistillCheckpointConfig`
  - `save_steps` / `keep_last`
- `DistillCheckpointManager`
  - `maybe_resume(resume_from_checkpoint=...) -> step | None`
  - `maybe_save(step)`
  - `save_final(step)`

**关键机制**
- 只对 `handle.trainable == True` 的 role 保存 optimizer/scheduler 状态。
- 使用 `torch.distributed.checkpoint (dcp)` 做分布式 checkpoint。
- `resume_from_checkpoint` 支持：
  - `checkpoint-<step>` 目录
  - `checkpoint-<step>/dcp`
  - `output_dir`（自动选择最新 checkpoint）

**与 Method 的关系**
- 该文件假设：训练开始前 `RoleHandle.optimizers/lr_schedulers` 已经就绪。
  Phase 2.9 开始，它们通常由 method（例如 `DMD2Method`）在构造时创建并写回 handle。
