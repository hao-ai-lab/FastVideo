# `fastvideo/distillation/outside/fastvideo/configs/distillation/finetune_wan2.1_t2v_1.3B_phase3.3.yaml`

这是一个 **Phase 3.3** 的可运行示例：把 finetuning 作为一种 method 接入 Phase 2+ distillation scaffold。

关键点：
- `recipe.method = finetune`
- `roles` 里只提供 `student`（no teacher/critic）
- 训练 loss 由 `FineTuneMethod` 实现（与 legacy `training_pipeline.py` 的目标对齐）
- validation 通过 `ValidationRequest + WanValidator + WanPipeline` 执行（默认走 `ode` sampler）
