# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`

这是一个可运行示例（schema v2）：**Wan few-step distillation（8 steps）+ DMD2**。

---

## 顶层结构

- `recipe:`
  - `family: wan` → registry dispatch 到 `families/wan.py`
  - `method: dmd2` → registry dispatch 到 `methods/distribution_matching/dmd2.py`
- `models:`
  - `student / teacher / critic` 三个 roles（role 名称本身由 method 解释语义）
  - 每个 role 指定：
    - `family`（默认可省略，继承 `recipe.family`）
    - `path`（权重路径）
    - `trainable`（是否训练）
    - `disable_custom_init_weights`（可选；用于 teacher/critic 等 auxiliary roles 的加载语义）
- `training:`
  - 主要复用 `TrainingArgs.from_kwargs()` 的字段集合（batch/shape/steps/logging 等）
- `pipeline_config:`
  - Phase 2 允许 inline 提供（也可用 `pipeline_config_path` 指向文件）

---

## 关键语义归属（Phase 2.9 视角）

**Family（Wan）关心：**
- `models.*.path / trainable`
- `training.data_path / dataloader_num_workers / train_batch_size / seed / output_dir`
- Wan 相关的 shape 信息（`num_latent_t/num_height/num_width/...`）

**Method（DMD2）关心：**
- update policy：`generator_update_interval`
- student rollout 相关：`method_config.rollout_mode`
- optimizer/scheduler（Phase 2.9 已迁移到 method 创建）：
  - student：`learning_rate / betas / lr_scheduler`
  - critic（DMD2 专属覆盖）：`fake_score_learning_rate / fake_score_betas / fake_score_lr_scheduler`
- few-step step list（single source of truth 在 `method_config`）：
  - `method_config.dmd_denoising_steps`

**Adapter（WanAdapter）关心：**
- 把 FastVideo/Wan 的 forward primitives 暴露给 method（不包含 step list/policy）

---

## 备注（Phase 3.2 已完成）

Phase 3.2 已将 validation sampling 的 ODE/SDE loop 做成可插拔 sampler：
- method 通过 `ValidationRequest(sampler_kind=..., sampling_timesteps=...)` 显式指定采样方式与 timesteps
- validator 使用统一的 `WanPipeline` 执行采样（不再依赖 `WanDMDPipeline`）
- 因此该 YAML 不再需要 `pipeline_config.dmd_denoising_steps` 这类重复字段
