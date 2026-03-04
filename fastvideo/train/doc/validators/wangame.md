# `fastvideo/distillation/validators/wangame.py`

**定位**
- `WanGameValidator` 是 WanGame 的 standalone validator：
  - 由 model plugin 构建并返回（当 `training.validation.enabled=true`，或 `training.validation` 非空）
  - 由 method 决定何时调用，并通过 `ValidationRequest` 指定采样细节（包含 dataset 与采样策略）

**pipeline 选择**
- 统一使用 `WanGameActionImageToVideoPipeline`，并通过 `sampler_kind={ode|sde}` 切换采样 loop 语义。

**调用方式（method-managed validation）**
- method 通过 `ValidationRequest(sample_handle=..., dataset_file=..., sampler_kind=..., ...)` 指定：
  - 用哪个 role 的 transformer 做采样（通常是 student）
  - validation dataset（`dataset_file`）
  - 采样步数（`sampling_steps`）
  - 若是 SDE 采样：`sampling_timesteps`（few-step rollout 的 explicit steps）
