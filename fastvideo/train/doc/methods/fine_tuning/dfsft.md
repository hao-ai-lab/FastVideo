# `fastvideo/distillation/methods/fine_tuning/dfsft.py`

**定位**
- `@register_method("dfsft")`：Diffusion-Forcing SFT（DFSFT）baseline。
- 只训练 `roles.student`，不依赖 teacher/critic。
- 目标：在 SFT/flow-matching loss 的基础上，引入 **chunk-wise inhomogeneous timesteps**
 （`t_inhom`）来覆盖“历史上下文不总是干净”的噪声分布（为 causal/streaming 训练做铺垫）。

**核心训练逻辑（单步）**
1) `model.prepare_batch(...)` 产出 `TrainingBatch`（包含 `x0` video latents + conditioning）。
2) 采样 `t_inhom`：
   - 先采样每个 chunk 的 timestep index（`method_config.chunk_size` 控制 chunk 划分）
   - 再 repeat 到每帧（`[B, T_lat]`）
3) 采样 `noise ~ N(0, I)`，得到 `x_t = model.add_noise(x0, noise, t_inhom_flat)`
4) 学生预测 `pred = model.predict_noise(student, x_t, t_inhom, batch, ...)`
5) loss：
   - 默认 flow-matching：`MSE(pred, noise - x0)`
   - 若 `training.precondition_outputs=true`：precondition 到 `x0` 再回归 `x0`

**关键 config**
- `method_config`（DFSFT 专属）
  - `chunk_size: int`（默认 3）：chunk-wise timestep 的 block size
  - `min_timestep_ratio / max_timestep_ratio: float`：采样 index 范围（映射到 scheduler 的 train steps）
  - `attn_kind: dense|vsa`：选择 model 的 dense/VSA attention metadata 路径

**约束**
- 如果 student transformer 暴露 `num_frame_per_block`，DFSFT 会要求
  `method_config.chunk_size == transformer.num_frame_per_block`，否则直接报错（避免配错造成语义不一致）。

**Validation**
- DFSFT 依赖 `training.validation`（由 method 驱动 `validator.log_validation(...)`）。
- 当前 validator 仍是 “full-video pipeline” 语义；真正的 streaming/causal rollout
 仍需要在后续阶段实现（避免把 rollout policy 藏进 validator/model plugin）。
