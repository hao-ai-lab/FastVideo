# Distillation docs (file-by-file)

本目录用于帮助 reviewer/贡献者快速理解 `fastvideo/distillation/` 的 Phase 2/2.9 架构。

设计原则（对应 Phase 2.9）：
- **Trainer** 只做 infra（loop/accum/日志/ckpt/validate 调用），不包含算法策略。
- **Method** 只做算法（loss + update policy + 需要哪些 roles）。
- **Model plugin** 只做装配（build-time：加载 modules、构建 bundle/adapter/dataloader/validator；代码在 `models/`）。
- **Adapter** 只做运行时 primitive（step-time：prepare_batch/forward_context/predict/backward 等），
  API 以 operation 为中心，不以 role 为中心（避免 role 爆炸）。

快速入口（从运行到训练）：
`fastvideo/training/distillation.py` → `utils.config.load_distill_run_config()` →
`dispatch.build_runtime_from_config()` →
`ModelComponents + DistillMethod` → `DistillTrainer.run()`

---

## Index

### Core
- `__init__.md`
- `utils/__init__.md`
- `utils/config.md`
- `utils/dataloader.md`
- `utils/moduleloader.md`
- `utils/module_state.md`
- `utils/tracking.md`
- `utils/checkpoint.md`
- `dispatch.md`
- `roles.md`
- `trainer.md`

### adapters/
- `adapters/__init__.md`
- `adapters/base.md`
- `adapters/wan.md`

### models/
- `models/__init__.md`
- `models/wan.md`

### methods/
- `methods/__init__.md`
- `methods/base.md`
- `methods/distribution_matching/__init__.md`
- `methods/distribution_matching/dmd2.md`
- `methods/consistency_model/__init__.md`
- `methods/knowledge_distillation/__init__.md`
- `methods/fine_tuning/__init__.md`
- `methods/fine_tuning/finetune.md`

### validators/
- `validators/__init__.md`
- `validators/base.md`
- `validators/wan.md`
