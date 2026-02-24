# `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase2.9.yaml`

这是一个 **Phase 2.9** 的可运行示例（schema v2）：
Wan few-step distillation（8 steps）+ DMD2。

它与 `distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml` 的结构基本一致，主要差异在于：
- `training.output_dir / tracker_project_name / wandb_run_name` 用于区分实验阶段。

备注：
- few-step step list 的 single source of truth 在 `method_config.dmd_denoising_steps`。
- 由于当前 validation 仍使用 legacy SDE sampling（`WanDMDPipeline`），pipeline 会读取
  `pipeline_config.dmd_denoising_steps`，因此该字段短期会与 `method_config` 重复；
  Phase 3.2 会移除此重复（sampler 可插拔 + 显式 timesteps request）。

