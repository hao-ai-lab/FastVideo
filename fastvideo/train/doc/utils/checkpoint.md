# `fastvideo/train/utils/checkpoint.py`

**目的**
- Role-based checkpoint/save-resume 管理：
  - 按 role 保存/恢复 modules、optimizers、schedulers（仅 trainable roles）
  - 可选保存 dataloader 状态（如果 dataloader 是 stateful）
  - 保存 RNG（全局 RNG + method 暴露的额外 generators）
  - 保存 callback 状态（如 validation RNG）
  - 支持 extra_role_modules（如 EMA shadow weights）

**关键类型**
- `CheckpointConfig`
  - `save_steps` / `keep_last`
- `CheckpointManager`
  - `maybe_resume(resume_from_checkpoint=...) -> step | None`
  - `maybe_save(step)`
  - `save_final(step)`

**关键机制**
- 只对 `model._trainable == True` 的 role 保存 optimizer/scheduler 状态。
- 使用 `torch.distributed.checkpoint (dcp)` 做分布式 checkpoint。
- `resume_from_checkpoint` 支持：
  - `checkpoint-<step>` 目录
  - `checkpoint-<step>/dcp`
  - `output_dir`（自动选择最新 checkpoint）

**独立函数**
- `maybe_warmstart_role_modules`: 从 DCP checkpoint best-effort 加载模型权重（不恢复 optimizer/step）。
- `save_role_pretrained`: 导出 role modules 为 diffusers-style 模型目录。
