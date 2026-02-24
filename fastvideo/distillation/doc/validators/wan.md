# `fastvideo/distillation/validators/wan.py`

**定位**
- `WanValidator`：Wan distillation 的 validation/sampling 实现（Phase 2 standalone）。
- 负责：
  - 读取 validation dataset（json）
  - 调用 Wan pipeline 生成视频样本
  - 保存 mp4 到 `output_dir`
  - 通过 tracker（例如 wandb）记录 artifacts

**关键点**
- validator 运行在分布式环境下：
  - 以 SP group 为单位做采样，最终由 global rank0 聚合写文件与 log
- 通过 `ValidationRequest.sample_handle` 获取本次要采样的 transformer，
  并以 `loaded_modules={"transformer": transformer}` 复用训练中的权重。
- method 通过 `ValidationRequest` 覆盖采样配置（例如 sampling steps / guidance / output_dir）。

**依赖**
- 当前使用 `WanDMDPipeline` 做采样推理（对齐 DMD2/SDE rollout，便于与 legacy apples-to-apples 对比）。
- 这不是最终形态：Phase 3 计划把 `WanPipeline` 的 denoising loop 抽象成可插拔的 sampler（ODE/SDE），
  从而淘汰 `WanDMDPipeline`，让 validator 回到 method-agnostic。

**可演进方向（Phase 3+）**
- 将 validation steps/guidance 等采样配置从 `TrainingArgs` 迁移到更明确的配置块（例如 `validation:`）。
- 进一步抽象 validator API，使其更容易被不同 family/method 复用。
