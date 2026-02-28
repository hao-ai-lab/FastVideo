# `fastvideo/distillation/adapters/wangame.py`

**定位**
- `WanGameAdapter` 是 WanGame model plugin 的 runtime 边界：
  - 把 wangame parquet batch（I2V + action）转成 method 可消费的
    **operation-centric primitives**
  - 不实现任何 method 的 rollout policy / step list / loss（这些属于 method）

**和 `WanAdapter` 的关键差异**
- WanGame 是 **I2V + action conditioning**：
  - transformer 的 `hidden_states` 不是纯视频 noisy latent，而是：
    `cat([noisy_video_latents, mask_lat_size, image_latents], dim=1)`
  - 还需要额外输入：
    - `encoder_hidden_states_image`（来自 `clip_feature`）
    - bidirectional transformer：`viewmats / Ks / action`
      （由 `process_custom_actions(keyboard, mouse)` 生成）
    - causal transformer：`mouse_cond / keyboard_cond`（raw action sequences）
- 目前 WanGame 的 `conditional/unconditional`（文本 CFG）语义**不成立**：
  - adapter 仍保留 `conditional: bool` 形参以匹配 method protocol，
    但当前实现把它当作 no-op（cond/uncond 同路）。

**主要 API（被 method 通过 protocol 调用）**
- `prepare_batch(...) -> TrainingBatch`
  - 处理 raw batch：
    - `vae_latent`（video x0）
    - `first_frame_latent`（I2V image latents）
    - `clip_feature`（image embeds）
    - `keyboard_cond` / `mouse_cond`（action）
  - 采样 timesteps + 生成 noise/sigmas + 生成 noisy video latents
  - 构建 attention metadata（dense / vsa 两套）
  - 生成 `mask_lat_size` 并预处理 action（`viewmats/Ks/action`）
- `predict_noise(handle, noisy_latents, timestep, batch, conditional, attn_kind) -> Tensor`
- `predict_x0(handle, noisy_latents, timestep, batch, conditional, attn_kind) -> Tensor`
- `add_noise(clean_latents, noise, timestep) -> Tensor`
- `shift_and_clamp_timestep(t) -> Tensor` + `num_train_timesteps`
- `backward(loss, ctx, grad_accum_rounds=...)`

**边界 / TODO**
- ✅ adapter 不保存/管理 few-step denoising step list，也不决定 rollout 策略。
- ✅ adapter 不引入 DMD2 专属概念（例如 “generator/critic”）。
- ✅ adapter 支持 **per-frame timesteps**（例如 DFSFT 的 `t_inhom`），但当启用 MoE
  `transformer_2 + boundary_timestep` 时要求 timestep 为标量（否则无法定义“跨帧选择哪个 transformer”）。
- TODO：若未来需要在 wangame 上定义 “uncond” 语义（例如 `zero_action/zero_image`），
  应通过 `method_config` 声明，并由 adapter 提供可解释的操作入口（而不是硬编码到 adapter 内部逻辑）。
