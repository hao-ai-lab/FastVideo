# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase2.9.yaml`

这是一个 **Phase 2.9** 的可运行示例（schema v2）：
Wan few-step distillation（8 steps）+ DMD2。

它与 `distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml` 的结构基本一致，主要差异在于：
- `training.output_dir / tracker_project_name / wandb_run_name` 用于区分实验阶段。

备注：
- few-step step list 的 single source of truth 在 `method_config.dmd_denoising_steps`。
- Phase 3.2 已将 validation 采样升级为可插拔的 ODE/SDE sampler：
  - method 通过 `ValidationRequest` 显式指定 `sampler_kind` 与 `sampling_timesteps`
  - validator 使用统一的 `WanPipeline` 执行采样（不再依赖 `WanDMDPipeline`）
