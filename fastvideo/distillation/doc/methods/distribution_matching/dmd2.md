# `fastvideo/distillation/methods/distribution_matching/dmd2.py`

**定位**
- `DMD2Method`：DMD2 distillation 的算法实现（method layer）。
- 该文件可以出现 DMD2/critic/fake_score 等算法术语；这些语义不应泄漏到 adapter/family。

**依赖与边界**
- ✅ 不 import 任何具体模型/管线实现（Wan/SDXL/...）。
- ✅ 只依赖：
  - `ModelBundle`/`RoleHandle`（获取 student/teacher/critic）
  - adapter 提供的 primitives（通过 `_DMD2Adapter` Protocol）

**需要的 roles**
- `student`：可训练（trainable=true）
- `teacher`：冻结（trainable=false）
- `critic`：可训练（trainable=true）

**算法结构（高层）**
1) `prepare_batch()`：交给 adapter
2) student 更新（按 `generator_update_interval` 控制频率）
   - 先做 student rollout 得到 `generator_pred_x0`
   - 再计算 DMD loss（teacher CFG 引导 + critic 输出构造梯度）
3) critic 更新（每 step）
   - 使用 student rollout（no-grad）构造 flow matching loss
4) backward
   - 由于 Wan 的 forward_context 约束，需要 adapter.backward(loss, ctx)

**few-step rollout policy（Phase 2.9）**
- rollout 的 step list / simulate 逻辑由 method 管理：
  - `pipeline_config.dmd_denoising_steps` → `_get_denoising_step_list()`
  - `simulate_generator_forward` 控制单步/多步模拟路径
  - 可选 `warp_denoising_step`（通过 adapter.noise_scheduler.timesteps duck-typing）
- adapter 只提供单步 primitives：
  - `predict_x0()` / `predict_noise()` / `add_noise()`

**optimizer/scheduler 的归属（Phase 2.9）**
- `DMD2Method` 在初始化时创建并写回：
  - student 的 optimizer/scheduler：使用 `training.learning_rate/betas/lr_scheduler`
  - critic 的 optimizer/scheduler：优先使用 `training.fake_score_*` 覆盖（否则回退到 student）
- 这样 Wan family 可以完全不“懂” DMD2 的 critic 超参，从 build-time 层面解耦。

**配置语义的 TODO（Phase 3）**
- 目前仍从 `training_args` 读取 DMD2/critic 专属字段（例如 `fake_score_*`、`dmd_denoising_steps`）。
  Phase 3 计划引入 `method_config`，把这些算法超参从 `training:` / `pipeline_config:` 中迁移出去。

