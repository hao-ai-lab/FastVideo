# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.1.yaml`

这是一个 **Phase 3.1** 的可运行示例（schema v2：`recipe` + `method_config`）：
Wan few-step distillation（8 steps）+ DMD2。

它与 `distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml` 的结构基本一致，主要用于强调 Phase 3.1 的配置语义：
- 顶层选择从 `distill` 升级为 `recipe`。
- method knobs（例如 `rollout_mode` / `dmd_denoising_steps`）进入 `method_config`。
- `WanAdapter.prepare_batch()` 不再读取 legacy 的 `training.simulate_generator_forward`。

备注：
- 由于当前 validation 仍使用 legacy SDE sampling（`WanDMDPipeline`），pipeline 会读取
  `pipeline_config.dmd_denoising_steps`，因此该字段短期会与 `method_config` 重复；
  Phase 3.2 会移除此重复（sampler 可插拔 + 显式 timesteps request）。

