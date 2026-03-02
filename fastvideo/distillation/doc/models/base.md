# `fastvideo/distillation/models/base.py`

**定位**
- 定义 `ModelBase` 抽象接口：由 model plugin 提供的 *运行时 primitives*（operation-centric）。
- Method 层不应该 import/依赖任何具体 pipeline/model，只依赖这些 primitives（duck typing / 抽象基类）。

**核心思想**
- `ModelBase` **不关心 roles 语义**（student/teacher/critic/... 由 method 定义）。
- `ModelBase` 提供 “对某个 role handle 执行某个操作” 的 API：
  - `predict_noise(handle, ...)`
  - `predict_x0(handle, ...)`
  - `add_noise(...)`
  - `prepare_batch(...)`
- 这样可以避免 “每个 role 一个函数” 的 role 爆炸。

**接口概览（必需）**
- `prepare_batch(raw_batch, current_vsa_sparsity, latents_source) -> TrainingBatch`
- `add_noise(clean_latents, noise, timestep) -> Tensor`
- `predict_noise(handle, noisy_latents, timestep, batch, conditional, cfg_uncond?, attn_kind) -> Tensor`
- `predict_x0(handle, noisy_latents, timestep, batch, conditional, cfg_uncond?, attn_kind) -> Tensor`
- `num_train_timesteps` / `shift_and_clamp_timestep(timestep)`
- `on_train_start()`
- `backward(loss, ctx, grad_accum_rounds)`

**接口概览（可选）**
- `get_rng_generators() -> dict[str, torch.Generator]`
  - Trainer/ckpt manager 用于保存 RNG state，实现 “exact resume”。
