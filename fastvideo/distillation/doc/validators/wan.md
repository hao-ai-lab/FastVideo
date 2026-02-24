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
- 使用统一的 `WanPipeline` 做采样推理：
  - `ValidationRequest.sampler_kind={ode|sde}` 选择 denoising loop
  - `ValidationRequest.sampling_timesteps` 提供 few-step schedule（写入 `ForwardBatch.sampling_timesteps`）
- 这样 validator 不再依赖 `<Model><Method>Pipeline`（例如 `WanDMDPipeline`），保持 method-agnostic。

**可演进方向（Phase 3+）**
- 将 validation steps/guidance 等采样配置从 `TrainingArgs` 迁移到更明确的配置块（例如 `validation:`）。
- 进一步抽象 validator API，使其更容易被不同 family/method 复用。
