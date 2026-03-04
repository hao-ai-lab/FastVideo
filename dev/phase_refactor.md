# Phase：Refactor（对齐 FastGen：Method 管“有哪些网络实例”，Student 持有 shared parts）

> 本阶段是一个 **较大重构**，目标是把目前 distillation 框架里“概念过多、职责漂移、读起来像一盘散沙”的问题，
> 收敛成一条更直观、FastGen 风格的主线：
>
> - **Method/Algorithm 决定自己需要哪些网络实例**（student/teacher/critic/fake_score/...），而不是框架层试图泛化所有 role。
> - **shared / expensive components**（VAE、encoder、scheduler、dataloader、validator 等）只构建一份，
>   默认由 `roles.shared_component_role`（通常是 student）负责加载并提供。
> - 每个 role 变成一个 **独立的模型实例（RoleModel）**：加载自己的 transformer（bidi 或 causal），并提供 forward primitives。
>
> 这一思路对齐 FastGen：
> - FastGen 里 `self.net`（student）初始化 preprocessors（`init_preprocessors()`），teacher 只是另一个 network 实例；
> - method/model class（如 `DMD2Model`/`SelfForcingModel`）硬编码构建 teacher/fake_score 等，而不是“传入 roles 列表自动生成”。

---

## 0) 目标 / 非目标

### ✅ 目标

1) **去掉 run-level “大 model plugin 内部 per-role if/else 分发 transformer”**
   - 例如现在 `wangame_causal` 里出现 `def _transformer_cls_name_for_role(...)` 这种逻辑。
   - 重构后：每个 role 自己对应一个独立 RoleModel（bidi / causal 各是一个 class），不在一个大类里分发。

2) **Method 明确声明“我需要哪些网络”**
   - `finetune`：只需要 student
   - `dmd2`：需要 student + teacher + critic（或 future：fake_score/discriminator）
   - `self_forcing`：需要 student（必须 causal）+ teacher/critic（可 bidi）

3) **shared parts 只加载一份（默认 student 持有）**
   - VAE / text encoder / image encoder / scheduler / dataloader / validator
   - 由 `roles.shared_component_role` 指定加载来源（默认 student），mismatch 就直接报错（从简、无 fallback）。

### ❌ 非目标（本阶段不做）

- 不追求兼容旧 YAML / legacy entrypoints（从简、只支持新语义）。
- 不做跨 family 的 teacher/student（例如 teacher=SDXL、student=Wan），遇到直接 error（后续再讨论）。
- 不在本阶段引入复杂的“shared parts 多套并存”（例如 teacher 也要自己的 VAE）。

---

## 1) 新的核心对象：`shared_context` + `RoleModel`

### 1.1 `shared_context`（按 run 只构建一次）

> 叫 `shared_context` 是为了避免和 backward 的 ctx 混淆。

它解决的问题：把“shared / expensive / 与 role 无关，但与 family/数据形态强相关”的东西集中管理。

以 **WanGame** 为例，`shared_context` 需要持有：

- `training_args`
- `dataloader`（parquet schema + workers 等）
- `validator`（可选；具体何时调用由 method 决定）
- shared modules：
  - `vae`
  - （可选）`text_encoder` / `image_encoder` / tokenizer / image_processor
  - `noise_scheduler`（以及 `num_train_timesteps`）
- batch 规范化：
  - `prepare_batch(raw_batch) -> TrainingBatch`
  - attention metadata builder（VSA/VMoBA）初始化与构建（如果属于 batch schema）
- RNG：
  - 训练噪声 RNG（用于 exact resume）

**代码落点（按你要求）**：直接放在 family 目录里，例如：

- `fastvideo/distillation/models/wangame/shared_context.py`

> FastGen 不需要显式 `shared_context.py` 的原因：它把 shared parts 隐式挂在 `self.net`（student）里；
> 我们这里显式拆出来，是因为 FastVideo 的 pipeline/validator/dataloader 语义更重、更适合集中持有（也更利于 checkpoint/resume）。

### 1.2 `RoleModel`（每个 role 一个独立实例）

RoleModel 只负责该 role 的 transformer 以及 forward primitives：

- `role: str`（用于日志/报错）
- `spec: RoleSpec`（family/path/trainable/extra…）
- `modules`（至少 `transformer`，可选 `transformer_2`）
- `trainable` 标记 + activation checkpointing（仅对 trainable role）

核心接口（operation-centric）：

- `predict_noise(shared_context, noisy_latents, timestep, batch, conditional, cfg_uncond, attn_kind)`
- `predict_x0(...)`
- （可选）`backward(loss, bwd_ctx, grad_accum_rounds)`：仅当 backward 需要恢复 forward_context 或处理 checkpoint wrapper 的特殊情况。

对 causal role（需要 KV cache / streaming rollout 的角色）：

- `CausalRoleModelBase` 扩展接口：
  - `clear_caches(cache_tag="pos")`
  - `predict_noise_streaming(shared_context, noisy_latents, timestep, batch, store_kv, cur_start_frame, ...)`
  - `predict_x0_streaming(...)`

> 对齐 FastGen：cache state 是 RoleModel 的 internal state，method 不传 KV tensor。

**代码落点建议**：仍放在 `models/<family>/` 里（和 shared_context 同一目录，读起来更高内聚）：

- `fastvideo/distillation/models/wangame/role_model.py`（bidirectional）
- `fastvideo/distillation/models/wangame/role_model_causal.py`（causal）
- `fastvideo/distillation/models/wangame/common.py`（共享 loader：load transformer / apply_trainable / activation checkpointing）

---

## 2) 配置语义（YAML）

### 2.1 `recipe.family`：选择 shared_context 的 family（不是“模型变体”）

重构后建议把 `recipe.family` 的职责收敛为：

- 选择 shared_context builder（数据 schema + shared parts 语义）

例如：

- `recipe.family: wangame` → build `WanGameSharedContext`
- `recipe.family: wan` → build `WanSharedContext`

**重要变化**：`recipe.family` 不再区分 `wangame` vs `wangame_causal`。
causal/bidi 的差异由 `roles.<role>.family` 决定（RoleModel 的选择）。

### 2.2 `roles.<role>.family`：选择 RoleModel 类型

例：

- `roles.student.family: wangame_causal` → `WanGameCausalRoleModel`
- `roles.teacher.family: wangame` → `WanGameRoleModel`

### 2.3 `roles.shared_component_role`：shared parts 的 owner（默认 student）

```yaml
roles:
  shared_component_role: student
```

约束（从简）：

- 必须是一个存在的 role name，否则直接报错。
- shared parts 的加载来源 = `roles[shared_component_role].path`
- `training.model_path` 必须与该 role 的 `path` 一致（不一致直接 error，防止 silent mismatch）。

---

## 3) 组装流程（dispatch/build）与调用链

### 3.1 build 的顺序（两段式）

1) build shared_context（由 `recipe.family` 选择）
2) build role_models（对每个 role，根据 `roles.<role>.family` 选择 RoleModel class）
3) build method（method 自己声明需要哪些 role，并 fail-fast 校验）

伪代码：

```py
shared_context = build_shared_context(cfg)  # recipe.family
role_models = build_role_models(cfg, shared_context)  # roles.<role>.family

method = Method.build(
  cfg=cfg,
  roles=RoleManager(role_models),
  shared_context=shared_context,
  validator=shared_context.validator,
)
```

### 3.2 示例调用链（Wangame：student causal self-forcing + teacher bidi）

假设你用 YAML 启动：

1) `fastvideo/training/distillation.py`
   - `cfg = load_distill_run_config(path)`
2) `fastvideo/distillation/dispatch.py: build_from_config(cfg)`
   - `shared_context = WangameSharedContext(cfg)`（从 `roles.shared_component_role.path` 加载 VAE 等）
   - `student = WanGameCausalRoleModel(role="student", spec=roles.student, shared_context=...)`
   - `teacher = WanGameRoleModel(role="teacher", spec=roles.teacher, shared_context=...)`
   - `critic  = WanGameRoleModel(role="critic",  spec=roles.critic,  shared_context=...)`
   - `method = SelfForcingMethod.build(cfg, roles=RoleManager(...), shared_context, validator=shared_context.validator)`
3) `fastvideo/distillation/trainer.py: DistillTrainer.run(...)`
   - 循环：
     - `loss_map, outputs, metrics = method.single_train_step(raw_batch, step, ...)`
     - `method.backward(...)`（若需要特殊 backward；否则走默认 `loss.backward()`）
     - `method.optimizers_schedulers_step(...)`
   - 验证：
     - `method.log_validation(step)` → `validator.log_validation(step, request=...)`

其中 Self-Forcing 的关键点：

- 强制 `student` 是 causal role model（否则 build 阶段直接 error）
- rollout 时只使用 `student.predict_*_streaming(...)`（内部 cache），teacher/critic 用并行 forward（无需 cache）

---

## 4) 需要修改/新增的文件（TODO 列表）

> 先以 wangame 落地，跑通后再推广到 wan。

### 4.1 shared_context

- [ ] `fastvideo/distillation/models/wangame/shared_context.py`
  - `WanGameSharedContext`

### 4.2 role models（每个 role 一个实例）

- [ ] `fastvideo/distillation/models/wangame/role_model.py`（bidi）
- [ ] `fastvideo/distillation/models/wangame/role_model_causal.py`（causal + cache/streaming）
- [ ] `fastvideo/distillation/models/wangame/common.py`（共享 loader 工具）

### 4.3 dispatch（从 “build model plugin” 改为 “build shared_context + role_models”）

- [ ] `fastvideo/distillation/dispatch.py`
  - 注册与构建：`@register_shared_context` / `@register_role_model`

### 4.4 methods（对齐 FastGen 的“继承表达差异点”）

- [ ] `fastvideo/distillation/methods/base.py`
  - `build(..., roles, shared_context, validator)`（替代 bundle+model plugin）
- [ ] `dmd2.py` 作为 DM 基类：self_forcing 只 override rollout
- [ ] `finetune.py` 作为 SFT 基类：dfsft 只 override timestep/chunk 采样（尽量复用 validation/optim/backward）

---

## 5) Refactor 后示例 YAML（Self-Forcing：student causal + teacher bidi）

> 下面仅展示关键结构；其余 training 超参按你现有 wangame YAML 填即可。

```yaml
recipe:
  family: wangame
  method: self_forcing

roles:
  shared_component_role: student

  student:
    family: wangame_causal
    path: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true

  teacher:
    family: wangame
    path: weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers
    trainable: false

  critic:
    family: wangame
    path: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true

training:
  # 必须：training.model_path 与 shared_component_role.path 一致（从简、fail-fast）
  model_path: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
  data_path: /path/to/wangame/parquet
  seed: 1000
  output_dir: outputs/wangame_self_forcing_refactor
  max_train_steps: 100000
  mixed_precision: bf16
  # ... 其余保持现有

  validation:
    enabled: true
    dataset_file: examples/training/finetune/WanGame2.1_1.3b_i2v/validation_random.json
    every_steps: 100
    sampling_steps: [4]
    sampler_kind: sde
    rollout_mode: streaming

method_config:
  rollout_mode: simulate
  dmd_denoising_steps: [4]  # 示例
  chunk_size: 3
  student_sample_type: sde
  context_noise: 0.0
```

---

## 6) 关键取舍（你 review 时最需要确认）

1) `recipe.family` 是否收敛为 “shared_context family”（我建议是）。
2) backward hook 的归属：默认 `loss.backward()`，只在必要时让 RoleModel 提供 `backward()`。
3) validator 的依赖：validator 通常更依赖 shared_context（vae/scheduler/pipeline_config），而不是 role_models。

