# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`

这是一个 Phase 2/2.9 的可运行示例：**Wan few-step distillation（8 steps）+ DMD2**。

---

## 顶层结构

- `distill:`
  - `model: wan` → registry dispatch 到 `families/wan.py`
  - `method: dmd2` → registry dispatch 到 `methods/distribution_matching/dmd2.py`
- `models:`
  - `student / teacher / critic` 三个 roles（role 名称本身由 method 解释语义）
  - 每个 role 指定：
    - `family`（默认可省略，继承 `distill.model`）
    - `path`（权重路径）
    - `trainable`（是否训练）
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
- student rollout 相关：`simulate_generator_forward`
- optimizer/scheduler（Phase 2.9 已迁移到 method 创建）：
  - student：`learning_rate / betas / lr_scheduler`
  - critic（DMD2 专属覆盖）：`fake_score_learning_rate / fake_score_betas / fake_score_lr_scheduler`
- few-step step list（目前仍放在 `pipeline_config`）：
  - `pipeline_config.dmd_denoising_steps`

**Adapter（WanAdapter）关心：**
- 把 FastVideo/Wan 的 forward primitives 暴露给 method（不包含 step list/policy）

---

## TODO（Phase 3）

为进一步减少 “training/pipeline_config 承载算法语义”，建议迁移：
- `fake_score_*` → `method_config.optimizers.critic.*`
- `dmd_denoising_steps` → `method_config.rollout.steps`（或类似命名）

