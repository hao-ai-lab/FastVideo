# `fastvideo/distillation/validators/wangame.py`

**定位**
- `WanGameValidator` 是 WanGame 的 standalone validator：
  - 由 model plugin 构建并返回（当 `training.log_validation=true`）
  - 由 method 决定何时调用，并通过 `ValidationRequest` 指定采样细节

**pipeline 选择（最小侵入式）**
- `sampler_kind="ode"`：
  - 使用 `WanGameActionImageToVideoPipeline`（ODE/UniPC 采样）
- `sampler_kind="sde"`：
  - 使用 `WanGameCausalDMDPipeline`（SDE-style rollout / DMD sampling）

> 说明：当前阶段我们用“切换 pipeline class”的方式区分 ODE/SDE，
> 以降低接入风险。后续若统一到同一个 pipeline 的 `sampler_kind` 分支，
> 可以减少重复逻辑，但改动会更侵入（属于后续 phase）。

**调用方式（method-managed validation）**
- method 通过 `ValidationRequest(sample_handle=..., sampler_kind=..., ...)` 指定：
  - 用哪个 role 的 transformer 做采样（通常是 student）
  - 采样步数（`sampling_steps`）
  - 若是 SDE 采样：`sampling_timesteps`（few-step rollout 的 explicit steps）

