# Phase 3：3.1 Config schema v2 + 3.2 ODE/SDE sampler + 3.3 Finetuning

Phase 2.9 已完成三件关键事情（为 Phase 3 铺路）：
- operation-centric adapter（adapter 不看 role string，只收 `RoleHandle`）
- policy 回归 method（few-step rollout / step list 等在 method 里）
- families + registry + builder（优雅 dispatch：新增 family 或 method 是 N+M，不是 N×M）

因此 Phase 3 不再聚焦 dispatch；Phase 3 的新增工作按顺序拆成三个子阶段：

- **Phase 3.1：Config schema v2（`recipe` + `method_config`）**
- **Phase 3.2：ODE/SDE sampler 可插拔（淘汰 `<Model><Method>Pipeline`）**
- **Phase 3.3：Finetuning method 接入（only student + dataset）**

约束（延续前几个 phase）：
- 不新增 entry file：继续使用 `fastvideo/training/distillation.py`。
- forward/backward context 仍由 adapter 托管；method 只负责算法编排。

---

## Phase 3.1：Config schema v2（`recipe` + `method_config`）

### 目标（DoD）
- distillation 入口只接受 **schema v2** 的 YAML（顶层 `recipe:`），并作为 single source of truth。
- `method_config` 能被解析并传入 method；至少 DMD2 的关键超参从 `method_config` 生效。
- 将 “method knob” 从 `training_args/pipeline_config` 中剥离出来（逐步迁移），并修复 Phase 2.9
  残留的语义泄漏：
  - `WanAdapter.prepare_batch()` 不再读取 `training_args.simulate_generator_forward`。

### Schema v2（示意）
```yaml
recipe:
  family: wan
  method: dmd2

models:
  student:
    family: wan
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    family: wan
    path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    trainable: false
  critic:
    family: wan
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

training: {...}          # infra（映射到 TrainingArgs）
pipeline_config: {...}   # backbone/pipeline（模型侧）
method_config: {...}     # algorithm（方法侧）
```

### 文件 TODO（实现清单）
- [x] `fastvideo/distillation/specs.py`
  - 新增 `RecipeSpec(family: str, method: str)`
  - `DistillRunConfig` 增加 `recipe` 与 `method_config`
- [x] `fastvideo/distillation/yaml_config.py`
  - 解析 `recipe:` 与 `method_config:`（默认 `{}`）
  - 将 v1 的 `distill:` 视为不再支持（breaking change，直接推进 schema v2）
- [x] `fastvideo/distillation/builder.py`
  - 从 `cfg.recipe` 取 family/method（不再读 `cfg.distill`）
  - build method 时传入 `method_config`
- [x] `fastvideo/distillation/methods/distribution_matching/dmd2.py`
  - `DMD2Method(..., method_config=...)`
  - 关键参数读取优先级：`method_config` > `training_args`（迁移期平滑）
    - `generator_update_interval`
    - `real_score_guidance_scale`
    - `dmd_denoising_steps`（few-step step list）
    - `rollout_mode`（替代 `simulate_generator_forward`）
- [x] `fastvideo/distillation/adapters/wan.py`
  - 移除 `training_args.simulate_generator_forward` 的读取（这是 Phase 2.9 的残留耦合）
  - 把 batch 形态做成显式 API/参数，让 method 决定：
    - 选项 A（推荐）：拆分显式入口
      - `prepare_batch_from_data_latent(raw_batch, ...)`（必须有 `vae_latent`）
      - `prepare_batch_from_placeholder_latent(raw_batch, ...)`（不依赖 `vae_latent`）
    - 选项 B：保留单入口但显式参数化：`prepare_batch(..., latents_source=...)`（本阶段采用）
- [x] configs / docs
  - [x] `fastvideo/distillation/outside/fastvideo/configs/distillation/*.yaml` 全部升级到 schema v2
  - [x] 更新 `dev/config.md`（描述 schema v2 与迁移策略）

### 可运行产物
- Phase 3.1 YAML：`fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.1.yaml`
- One-shot 脚本：`examples/distillation/phase3_1/temp.sh`

---

## Phase 3.2：ODE/SDE sampler 可插拔（统一 sampling loop 语义）

### 背景（为什么需要）
Phase 2.9 已验证：即使统一 timesteps/scheduler，**只要 denoising loop 不同**，sampling 结果仍会 drift：
- ODE/solver 风格：`scheduler.step(noise_pred, t, latents)`（当前 `WanPipeline`）
- SDE 风格：`pred_x0 -> add_noise(next_t, eps)`（DMD2/legacy `WanDMDPipeline`）

如果继续靠 `<Wan><DMD2>Pipeline` 这类 pipeline 变体来选择 loop，会重新走向 N×M 组合爆炸。

### 目标（DoD）
- `WanPipeline` 支持通过参数/配置选择 sampler（`ode|sde`），默认 `ode`。
- distillation 的 validation 由 method/method_config 显式指定 sampler + step list；
  validator 回到 method-agnostic，不再 import `WanDMDPipeline`。
- `WanDMDPipeline` 保留为 legacy 兼容（可选），但新框架不依赖它。

### 文件 TODO（实现清单）
- [x] 抽象 sampler（中性命名，不出现 DMD）
  - `fastvideo/pipelines/samplers/`：`SamplerKind` + Wan sampler helpers
  - `pipeline_config.sampler_kind={ode|sde}`：`WanPipeline` 通过该参数选择 sampling loop
- [x] `fastvideo/pipelines/stages/denoising.py`
  - `SdeDenoisingStage`：SDE 风格 rollout（`pred_x0 -> add_noise(next_t, eps)`）
  - `SdeDenoisingStage` 接受显式 `batch.sampling_timesteps`（来自 ValidationRequest）
  - 继续使用 `batch.generator` 生成每一步注入的 `eps`（可复现）
  - 保留 `DmdDenoisingStage = SdeDenoisingStage` alias（legacy pipeline 兼容）
- [x] `fastvideo/pipelines/basic/wan/wan_pipeline.py`
  - `WanPipeline` 支持 `sampler_kind={ode|sde}`（单一 pipeline 覆盖两种 loop）
- [x] `fastvideo/distillation/validators/base.py`
  - `ValidationRequest` 新增：
    - `sampler_kind: Literal["ode", "sde"] | None`
    - `sampling_timesteps: list[int] | None`
- [x] `fastvideo/distillation/validators/wan.py`
  - 使用 `WanPipeline` + request 的 sampler/timesteps（不再 import `WanDMDPipeline`）
- [x] `fastvideo/distillation/methods/distribution_matching/dmd2.py`
  - validation request 指定 `sampler_kind="sde"` + `sampling_timesteps=<few-step list>`

### 可运行产物
- Phase 3.2 YAML：`fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.2.yaml`
- One-shot 脚本：`examples/distillation/phase3_2/temp.sh`

---

## Phase 3.3：Finetuning method（only student）

### 目标（DoD）
- 新增 `finetune` method，复用同一套 Trainer/Bundle/Adapter/Family/Validator 基础设施。
- 最小可运行：只需 `models.student` + dataset 即可训练。
- finetune 的 method 参数进入 `method_config`（与 Phase 3.1 schema 一致）。

### 文件 TODO（实现清单）
- [ ] `fastvideo/distillation/methods/fine_tuning/finetune.py`
  - `FineTuneMethod(DistillMethod)` + `@register_method("finetune")`
  - `bundle.require_roles(["student"])`
  - `single_train_step()` 只更新 student
- [ ] （如有必要）为 finetune 定义 adapter contract（类似 `_DMD2Adapter` 的做法）
  - 重点：**loss 仍由 method 计算**；adapter 只提供 operation-centric primitives
  - `_FineTuneAdapter(Protocol)` 推荐只包含：
    - `prepare_batch(...)`（产出 latents/noise/timesteps/sigmas/conditioning）
    - `predict_noise(handle, ...)`（以及可选 `predict_x0`）
    - `backward(loss, ctx, ...)`（forward-context/activation ckpt 相关）
- [ ] configs/examples
  - [ ] `fastvideo/distillation/outside/fastvideo/configs/distillation/finetune_*.yaml`
  - [ ] `examples/distillation/phase3/`（或更新现有 examples）

---

## 备注：关于 `simulate_generator_forward`（Phase 2.9 的残留）

该耦合已在 Phase 3.1 解决：
- `WanAdapter.prepare_batch()` 不再读取 `training_args.simulate_generator_forward`
- `DMD2Method` 通过 `method_config.rollout_mode` 决定 `latents_source={zeros|data}`，
  并把它作为参数传给 adapter（adapter 只处理 batch 形态，不解释 DMD2 语义）
