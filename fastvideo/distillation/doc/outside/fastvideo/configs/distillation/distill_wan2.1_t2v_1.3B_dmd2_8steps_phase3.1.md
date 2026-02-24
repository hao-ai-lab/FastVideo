# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.1.yaml`

这是一个 **Phase 3.1** 的可运行示例（schema v2：`recipe` + `method_config`）：
Wan few-step distillation（8 steps）+ DMD2。

它与 `distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml` 的结构基本一致，主要用于强调 Phase 3.1 的配置语义：
- 顶层选择从 `distill` 升级为 `recipe`。
- method knobs（例如 `rollout_mode` / `dmd_denoising_steps`）进入 `method_config`。
- `WanAdapter.prepare_batch()` 不再读取 legacy 的 `training.simulate_generator_forward`。

备注：
- Phase 3.2 已将 validation 采样升级为可插拔的 ODE/SDE sampler：
  - method 通过 `ValidationRequest` 显式指定 `sampler_kind` 与 `sampling_timesteps`
  - validator 使用统一的 `WanPipeline` 执行采样（不再依赖 `WanDMDPipeline`）
