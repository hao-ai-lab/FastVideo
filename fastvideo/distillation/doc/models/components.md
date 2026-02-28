# `fastvideo/distillation/models/components.py`

**定位**
- `ModelComponents` 是 model plugin 的 build-time 产物（装配输出）。
- `dispatch.build_runtime_from_config()` 先构建 `ModelComponents`，再把其中的
  `bundle/adapter/validator` 注入到 method，最终交给 trainer 执行训练。

**字段说明**
- `training_args`
  - 来自 YAML 的 `training`（通过 `TrainingArgs.from_kwargs` 解析）。
- `bundle`
  - `RoleManager(roles=...)`：role → `RoleHandle(modules/optimizers/schedulers/...)`。
- `adapter`
  - 对应模型家族的运行时 primitive（例如 `WanAdapter` / `WanGameAdapter`）。
- `dataloader`
  - 当前 run 的训练 dataloader（由通用 dataloader builder 构建）。
- `validator`（可选）
  - 模型家族对应的 validator backend；是否启用由 method 的 validation 配置决定。
- `start_step`
  - autoresume/ckpt 恢复后从哪个 step 开始继续。

