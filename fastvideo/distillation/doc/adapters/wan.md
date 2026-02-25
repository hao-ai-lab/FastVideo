# `fastvideo/distillation/adapters/wan.py`

**定位**
- `WanAdapter` 是 Wan model plugin 的 runtime 边界：
  - 把 FastVideo/Wan 的 batch schema、forward_context、attention metadata 等细节
    封装为一组 **operation-centric primitives**
  - 不实现任何 method 的 rollout policy / step list / loss（这些属于 method）

**关键依赖**
- `RoleHandle`：adapter 不认识 role 字符串，method 传 handle 进来，adapter 只用 handle 拿模块。
- `fastvideo.forward_context.set_forward_context`：Wan forward/backward 依赖全局上下文。
- attention metadata builder（VSA / VMOBA）与 `envs.FASTVIDEO_ATTENTION_BACKEND`。

**主要 API（被 method 通过 protocol 调用）**
- `prepare_batch(...) -> TrainingBatch`
  - 处理 raw_batch → latents/noise/timesteps/sigmas
  - 构建 `conditional_dict` / `unconditional_dict`（含 negative prompt embeds）
  - 构建 attention metadata（dense / vsa 两套）
- `predict_x0(handle, noisy_latents, timestep, batch, conditional, attn_kind) -> Tensor`
- `predict_noise(handle, noisy_latents, timestep, batch, conditional, attn_kind) -> Tensor`
- `add_noise(clean_latents, noise, timestep) -> Tensor`
- `shift_and_clamp_timestep(t) -> Tensor` + `num_train_timesteps`
- `backward(loss, ctx, grad_accum_rounds=...)`

**关于 negative/unconditional conditioning**
- `ensure_negative_conditioning()` 只做 prompt encoding（无 denoise）。
- 为避免算法命名耦合，prompt encoding 使用 `WanPipeline`（而不是带 method 语义的 pipeline 名称）。

**边界（Phase 2.9）**
- ✅ adapter 不保存/管理 few-step 的 denoising step list，也不决定 rollout 策略。
- ✅ adapter 不区分 `student/teacher/critic` 的专用方法；只提供通用操作，role 语义由 method 管理。
