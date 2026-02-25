# `fastvideo/distillation/models/wan.py`

**定位**
- `@register_model("wan")` 的 build-time 插件（实现：`build_wan_components(...)`）：
  - 负责把 YAML config 装配成 `ModelComponents`
  - 包含 Wan 特有的模块加载、shared components、dataloader schema 等逻辑

**产物**
- `ModelComponents(training_args, bundle, adapter, dataloader, tracker, validator, start_step)`

**主要职责**
1) **加载 shared components**
   - `vae`：从 student 的 base `model_path` 加载
   - `noise_scheduler`：`FlowMatchEulerDiscreteScheduler(shift=flow_shift)`
2) **按 roles 加载 transformer 模块**
   - 对每个 role：加载 `transformer`（teacher 可选 `transformer_2`）
   - 根据 `RoleSpec.trainable` 设置 `requires_grad` + `train()/eval()`
   - 可选开启 activation checkpoint（仅对 trainable role）
3) **构建 bundle / adapter / dataloader**
   - `bundle = ModelBundle(roles=role_handles)`
   - `adapter = WanAdapter(prompt_handle=student_handle, ...)`
   - dataloader：parquet + `pyarrow_schema_t2v`
4) **tracker / validator（可选）**
   - tracker：`initialize_trackers(...)`（rank0 才启用）
   - validator：`WanValidator`（当 `training_args.log_validation=true`）
     - model plugin 只负责构建并返回 `validator`
     - validator 本身不应 hardcode `bundle.role("student")` 等角色语义；
       method 通过 `ValidationRequest.sample_handle` 指定要采样的模型
     - 是否调用、用什么采样配置由 method 决定（method-managed validation）

**Phase 2.9 的关键变化**
- ✅ model plugin 不再创建 optimizers/schedulers。
  - 这类 update policy（哪些 role 训练、各自超参）属于 method/算法语义。
  - 当前由 `DMD2Method` 在初始化时创建并写回 `RoleHandle.optimizers/lr_schedulers`。

**注意 / TODO**
- YAML 中目前仍使用 `training.fake_score_*` 这类字段作为 DMD2 的 critic 超参来源；
  Phase 3 计划把它们迁移到 `method_config`，进一步减少 “training_args 承载算法语义”。
