# Phase 3：优雅 Dispatch（N+M）+ `recipe` 配置语义升级 + Finetuning 接入

Phase 3 的定位：在 Phase 2 已经证明“新 distill 框架可独立运行”的基础上，把它推进到
“长期可扩展、可承载更多训练 recipe（包括 finetune）”的状态。

本 phase 目标（你已拍板）：

1) **彻底优雅的 dispatch**：避免 `wan+dmd2` 这种硬编码分支；扩展成本从 N×M 降到 N+M。  
2) **YAML schema 升级**：顶层从 `distill` 改为 `recipe`，并新增 `method_config`。  
3) **新增 finetuning 支持**：把 finetune 作为一种 method 接入框架（only student + dataset）。

约束：
- 不新增 entry file：继续使用 `fastvideo/training/distillation.py` 作为统一入口。
- Phase 3 仍遵循边界：**forward/backward context 由 adapter 托管**，method 只实现算法/编排。

---

## A) Phase 3 交付目标（Definition of Done）

- `fastvideo/training/distillation.py` 不再出现 `if family==... and method==...` 的组合硬编码；
  而是通过 registry 查找 family/method 组件来构建 runtime。
- YAML schema 支持 `recipe + models + training + pipeline_config + method_config`：
  - `distill:` schema v1 可以选择性兼容（见“风险/决策点”）。
  - 文档与 examples 更新到 schema v2（`recipe`）。
- 能跑通两个 recipe：
  1) `recipe.method=dmd2`（现有 few-step wan DMD2）  
  2) `recipe.method=finetune`（wan finetune 最小版本，only student）
- `method_config` 在代码中**真实生效**（至少 DMD2 的 update policy + guidance scale 使用它）。

---

## B) TODO List（Review Checklist）

### B1. 彻底优雅的 dispatch（registry + 通用 builder）

- [ ] 新增 `fastvideo/distillation/registry.py`
  - `register_family(name)` / `register_method(name)` 装饰器
  - `get_family(name)` / `get_method(name)`：带可用项提示的错误信息
  - `ensure_builtin_registrations()`：导入内置 family/method 以完成注册
- [ ] 新增 `fastvideo/distillation/families/`（model family 插件）
  - `fastvideo/distillation/families/__init__.py`
  - `fastvideo/distillation/families/wan.py`：`WanFamily`（复用 Phase 2 的加载与数据构建逻辑）
- [ ] 改造 `fastvideo/distillation/builder.py`
  - 把当前 `build_wan_dmd2_runtime_from_config()` 收敛为：
    - `build_runtime_from_config(cfg: RunConfig) -> DistillRuntime`
  - `DistillRuntime.method` 类型从 `DMD2Method` 泛化为 `DistillMethod`
- [ ] 改造入口 `fastvideo/training/distillation.py`
  - 入口只做：`cfg = load_run_config()` → `runtime = build_runtime_from_config(cfg)` → `trainer.run(...)`
  - 不再写组合分支（`wan+dmd2` 等）

### B2. YAML schema v2：`recipe` + `method_config`

- [ ] 更新 spec：`fastvideo/distillation/specs.py`
  - 新增 `RecipeSpec(family: str, method: str)`
  - 保留 `RoleSpec(family/path/trainable)`，与 Phase 2 兼容
- [ ] 更新 YAML loader：`fastvideo/distillation/yaml_config.py`
  - 新 schema：
    - `recipe: {family, method}`
    - `method_config: { ... }`（可选，默认 `{}`）
  - 入口层推导 `TrainingArgs.mode`：
    - `recipe.method == "finetune"` → `ExecutionMode.FINETUNING`
    - 其它 method → `ExecutionMode.DISTILLATION`
  - （可选）兼容 v1：若仅提供 `distill: {model, method}`：
    - 转换为 `recipe.family = distill.model`，`recipe.method = distill.method`
    - 打 warning 提示迁移（见“风险/决策点”）

### B3. `method_config` 接入到 DMD2（让它真正生效）

- [ ] 更新 `fastvideo/distillation/methods/distribution_matching/dmd2.py`
  - `DMD2Method` 构造函数增加 `method_config: dict[str, Any] | None`
  - 读取优先级：`method_config` > `training_args` 默认值（用于平滑迁移）
    - `generator_update_interval`
    - `real_score_guidance_scale`
    - （可选）`simulate_generator_forward`（若继续存在）

### B4. Finetuning method（only student）

- [ ] 新增 `fastvideo/distillation/methods/fine_tuning/finetune.py`
  - `FineTuneMethod(DistillMethod)`
  - `bundle.require_roles(["student"])`
  - `single_train_step()`：
    - `training_batch = adapter.prepare_batch(...)`
    - `pred = adapter.student_predict(...)`
    - `loss = adapter.finetune_loss(...)`
  - update policy：
    - `get_optimizers()` / `get_lr_schedulers()` 始终返回 student 的
- [ ] 在 `fastvideo/distillation/methods/fine_tuning/__init__.py` 暴露该 method（并注册）

### B5. WanAdapter 增强 finetune primitives（不让 method 管 forward_context）

- [ ] 更新 `fastvideo/distillation/adapters/wan.py`
  - 新增 `_FineTuneAdapter(Protocol)` 所需 primitives：
    - `student_predict_for_finetune(batch) -> torch.Tensor`
    - `finetune_loss(batch, pred) -> torch.Tensor`
    - （如 activation checkpoint/backward 重算需要）`backward_student(loss, ctx, ...)`
  - 复用现有的：
    - `prepare_batch()`（要求 finetune 路径必须提供真实 `vae_latent`）
    - `set_forward_context(...)` 的管理继续留在 adapter

### B5.5 彻底去除 adapter 对 method knob 的读取（例如 `simulate_generator_forward`）

动机：当前 `WanAdapter.prepare_batch()` 仍读取 `training_args.simulate_generator_forward` 来决定
是否需要 `vae_latent`（或构造 placeholder）。这会把 DMD2/student-rollout 的语义泄漏到 adapter，
违背 Phase 2.9 的 operation-centric 边界。

- [ ] 调整 `WanAdapter` 的 batch API，使其不再读取 `training_args.simulate_generator_forward`
  - 选项 A（推荐）：拆成两个显式入口
    - `prepare_batch_with_vae_latent(raw_batch, ...)`：必须有 `vae_latent`
    - `prepare_batch_text_only(raw_batch, ...)`：不依赖 `vae_latent`，只根据 family 规则构造 latents/shape
  - 选项 B：保留单入口但显式参数化
    - `prepare_batch(raw_batch, *, latents_source=...)`
    - 由 method/`method_config` 决定使用 data latents 还是 placeholder latents
- [ ] 将 “是否 simulate / rollout 模式” 的开关迁移到 `method_config`
  - DMD2：例如 `method_config.rollout_mode = "data_latent" | "simulate"`
  - adapter 不读取该开关；method 选择调用哪个 adapter API

### B6. examples / outside configs 更新到 schema v2

- [ ] 更新 DMD2 YAML（schema v2）：
  - `fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`
    - `distill -> recipe`
    - 新增 `method_config`（至少包含 DMD2 的 interval + guidance scale + simulate flag）
- [ ] 新增 finetune YAML（schema v2）：
  - `fastvideo/distillation/outside/fastvideo/configs/distillation/finetune_wan2.1_t2v_1.3B.yaml`（命名可再讨论）
- [ ] 更新 `examples/distillation/phase2/README.md` 与脚本说明（指向 `recipe` schema）

### B7.（可选但推荐）最小单测

- [ ] `fastvideo/tests/distillation/test_yaml_schema_v2.py`
  - schema v2 能 parse
  - （如兼容 v1）schema v1 → v2 conversion 能 parse
- [ ] `fastvideo/tests/distillation/test_registry_dispatch.py`
  - registry 能注册并 resolve `wan` + `dmd2` / `finetune`

---

## C) 核心代码设计（具体到类/函数）

### C1. `registry.py`：N+M 组合的关键

目标：入口/Builder 只写一次组合逻辑，不写 `build_<family>_<method>()`。

建议接口（简化版）：

- `DistillFamily`（family 插件）
  - `build_bundle(cfg) -> ModelBundle`
  - `build_shared_components(cfg) -> ...`（如 vae/scheduler）
  - `build_adapter(bundle, cfg, shared) -> DistillAdapter`
  - `build_validator(bundle, cfg, tracker, shared) -> DistillValidator | None`
  - `build_dataloader(cfg) -> DataLoaderLike`
  - `build_optimizers_schedulers(bundle, cfg) -> None`（写入 RoleHandle）

- `DistillMethodFactory`（method 插件）
  - `build(bundle, adapter, cfg) -> DistillMethod`

注册方式：
- `@register_family("wan") class WanFamily: ...`
- `@register_method("dmd2") def build_dmd2(...): ...`
- `@register_method("finetune") def build_finetune(...): ...`

> 这样新增一个 family 或 method 是一次注册（N+M），不会产生 N×M 组合函数。

### C2. 通用 `build_runtime_from_config(cfg)`

`fastvideo/distillation/builder.py` 收敛为一个通用入口：

1) `ensure_builtin_registrations()`
2) `family = get_family(cfg.recipe.family)`
3) `method_factory = get_method(cfg.recipe.method)`
4) `bundle = family.build_bundle(cfg)`
5) `family.build_optimizers_schedulers(bundle, cfg)`（只给 trainable roles 创建）
6) `shared = family.build_shared_components(cfg)`
7) `tracker = family.build_tracker(cfg)`（或 builder 复用 training/trackers）
8) `validator = family.build_validator(...)`（可选）
9) `adapter = family.build_adapter(bundle, cfg, shared, validator)`
10) `method = method_factory.build(bundle=bundle, adapter=adapter, cfg=cfg)`
11) `dataloader = family.build_dataloader(cfg)`
12) 返回 `DistillRuntime(training_args, method, dataloader, tracker, start_step=0)`

### C3. YAML loader（schema v2）

`fastvideo/distillation/yaml_config.py` 产出新的 `RunConfig`（命名可沿用 `DistillRunConfig`）：

- `recipe: RecipeSpec`
- `roles/models: dict[str, RoleSpec]`
- `training_args: TrainingArgs`
- `method_config: dict[str, Any]`
- `raw: dict[str, Any]`

并将 `training_args.mode` 与 `inference_mode` 强制一致（Phase 2 的经验）：
- `finetune` → `FINETUNING` + `inference_mode=False`
- distill methods → `DISTILLATION` + `inference_mode=False`

### C4. DMD2：method_config 生效点

Phase 3 最小要求：以下两个参数从 `method_config` 读取（否则 `method_config` 只是摆设）：
- `generator_update_interval`
- `real_score_guidance_scale`

兼容策略（建议）：如果 `method_config` 缺失，则回落到 `training_args` 字段，避免破坏旧 YAML。

### C5. Finetune：method + adapter contract

Finetune 的边界：
- method 负责算法编排（loss/optim schedule）
- adapter 负责 forward_context + model/pipeline 差异

建议 `_FineTuneAdapter` 的最小 surface：
- `prepare_batch(raw_batch, current_vsa_sparsity=...) -> TrainingBatch`
- `student_predict_for_finetune(batch) -> torch.Tensor`
- `finetune_loss(batch, pred) -> torch.Tensor`
- `backward_student(loss, ctx, grad_accum_rounds=...)`（如果需要）

> Wan 侧可以复用现有 `TrainingBatch` 字段与 `normalize_dit_input / get_sigmas` 等工具，
> 目标是对齐 legacy `TrainingPipeline._transformer_forward_and_compute_loss()` 的 loss 语义。

---

## D) 风险点 / 需要你参与决策的地方（遇到会停下讨论）

1) **schema v1 兼容性**：Phase 3 是否需要继续支持 `distill:`？
   - 选项 A：完全切到 `recipe:`（更干净，但需要一次性改所有 YAML）
   - 选项 B：兼容 v1 → v2 conversion + warning（更平滑，但多一点维护负担）

2) **`method_config` 的形态**：是否需要做 method namespace？
   - 选项 A：flat mapping（推荐）：`method_config: {generator_update_interval: 5, ...}`
   - 选项 B：namespaced：`method_config: {dmd2: {...}}`（更显式但更啰嗦）

3) **finetune 的 validation**：Phase 3 是否要求 finetune 也能 video validation？
   - 最小交付可以先让 finetune 训练跑通，validation 可先关（`log_validation=false`）
   - 若要开启，需要确认 `WanValidator` 的 sampling pipeline 是否应随 recipe.method 切换

---

## E) YAML 小结：flow mapping vs block mapping

YAML 语法上：

```yaml
student: {family: wan, path: ..., trainable: true}
```

与

```yaml
student:
  family: wan
  path: ...
  trainable: true
```

在语义上是等价的（都是一个 mapping）。Phase 3 的 examples/docs 我们将统一采用 **block
style**（你偏好的后者），因为更可读、diff 更友好。
