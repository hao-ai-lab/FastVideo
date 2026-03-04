# `fastvideo/distillation/methods/fine_tuning/finetune.py`

## 目的
- 将 “finetuning / SFT” 以 `DistillMethod` 的方式接入 Phase 2+ 架构：
  - 复用 `DistillTrainer`（infra loop / accum / step / ckpt / validate 调用）
  - 复用 `RoleManager`（角色容器，finetune 只需要 `student`）
  - 复用 model plugin/adapter（加载与 primitives）

finetune 可以被视为一种特殊的 distillation recipe：**只有 student + dataset**。

## 角色依赖
- 必需：`student`
- 不需要：`teacher / critic / reward / ...`

方法会强制：
- `roles.student.trainable=true`

## 核心训练逻辑
`FineTuneMethod.single_train_step()`：
1. `adapter.prepare_batch(..., latents_source="data")`
2. 用 student 做 `adapter.predict_noise(student, noisy_latents, timesteps, batch, conditional=True)`
3. 计算 loss（与 legacy `training_pipeline.py` 对齐）：
   - 默认（`training.precondition_outputs=false`）：
     - target = `noise - x0`
     - loss = `mse(pred, target)`
   - 若 `training.precondition_outputs=true`：
     - 先 precondition 到 `x0`：`pred_x0 = x_t - sigma * pred`
     - loss = `mse(pred_x0, x0)`
4. backward 通过 `adapter.backward(loss, ctx, ...)` 执行（确保 forward-context/activation ckpt 兼容）

## Optimizer / Scheduler
- 由 method 创建（而非 model plugin）：
  - 使用 `training.learning_rate / training.betas / training.lr_scheduler / ...`
  - 只为 `student` role 创建 `optimizer + lr_scheduler`

## Validation
- `FineTuneMethod.log_validation()` 构造 `ValidationRequest(sample_handle=student, ...)`
- 具体 pipeline 与采样 loop 由 validator + `pipeline_config.sampler_kind` 决定（默认 `ode`）

## 配置示例
- `examples/distillation/phase3_3/finetune_wan2.1_t2v_1.3B_phase3.3.yaml`

## 注册方式（Phase 3.4）
- `FineTuneMethod` 通过 `@register_method("finetune")` 直接注册到 class。
- 由 `FineTuneMethod.build(...)` 负责把 `cfg.method_config` 等注入到实例。
