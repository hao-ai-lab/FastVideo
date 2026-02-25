# Phase 3：3.1 Config schema v2 + 3.2 ODE/SDE sampler + 3.3 Finetuning + 3.4 命名/结构整理

Phase 2.9 已完成三件关键事情（为 Phase 3 铺路）：
- operation-centric adapter（adapter 不看 role string，只收 `RoleHandle`）
- policy 回归 method（few-step rollout / step list 等在 method 里）
- families + registry + builder（优雅 dispatch：新增 family 或 method 是 N+M，不是 N×M）

因此 Phase 3 不再聚焦 dispatch；Phase 3 的新增工作按顺序拆成三个子阶段：

- **Phase 3.1：Config schema v2（`recipe` + `method_config`）**
- **Phase 3.2：ODE/SDE sampler 可插拔（淘汰 `<Model><Method>Pipeline`）**
- **Phase 3.3：Finetuning method 接入（only student + dataset）**
- **Phase 3.4：命名/结构整理（降低概念数量 + 更直觉的目录组织）**

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

roles:
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
- [x] `fastvideo/distillation/utils/config.py`
  - 新增 `RecipeSpec(family: str, method: str)`
  - `DistillRunConfig` 增加 `recipe` 与 `method_config`
- [x] `fastvideo/distillation/utils/config.py`（包含 YAML loader + schema/dataclass）
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
- 最小可运行：只需 `roles.student` + dataset 即可训练。
- finetune 的 method 参数进入 `method_config`（与 Phase 3.1 schema 一致）。

### 文件 TODO（实现清单）
- [x] `fastvideo/distillation/methods/fine_tuning/finetune.py`
  - `FineTuneMethod(DistillMethod)` + `@register_method("finetune")`
  - `bundle.require_roles(["student"])`
  - `single_train_step()` 只更新 student
- [ ] （如有必要）为 finetune 定义 adapter contract（类似 `_DMD2Adapter` 的做法）
  - 重点：**loss 仍由 method 计算**；adapter 只提供 operation-centric primitives
  - `_FineTuneAdapter(Protocol)` 推荐只包含：
    - `prepare_batch(...)`（产出 latents/noise/timesteps/sigmas/conditioning）
    - `predict_noise(handle, ...)`（以及可选 `predict_x0`）
    - `backward(loss, ctx, ...)`（forward-context/activation ckpt 相关）
- [x] configs/examples
  - [x] `fastvideo/distillation/outs
  ide/fastvideo/configs/distillation/finetun e_wan2.1_t2v_1.3B_phase3.3.yaml`
  - [x] `examples/distillation/phase3_3/temp.sh`  

---

## 备注：关于 `simulate_generator_forward`（Phase 2.9 的残留）

该耦合已在 Phase 3.1 解决：
- `WanAdapter.prepare_batch()` 不再读取 `training_args.simulate_generator_forward`
- `DMD2Method` 通过 `method_config.rollout_mode` 决定 `latents_source={zeros|data}`，
  并把它作为参数传给 adapter（adapter 只处理 batch 形态，不解释 DMD2 语义）

---

## Phase 3.4：命名/结构整理（降低概念数量 + 更直觉）

### 背景（为什么要做）

Phase 3.1~3.3 已经把训练端到端跑通；但目前 `fastvideo/distillation/` 的概念命名偏“框架内部术语”，对新 reviewer 不友好：
- `families/` 读起来像“人类家族”，但它实际承担的是 **model/pipeline contract 的集成/装配层**。
- `bundle.py` 读起来像“打包”，但它本质是 **roles 管理/索引容器**。
- `registry.py` / `builder.py` /（以及一些纯 dataclass 文件）分散在多个文件，阅读路径长，容易产生“概念过多”的感受。

我们希望把这些改成更直觉的命名，并把“infra”从“模型集成层”里抽出来。

> 备注：此阶段优先做 **低风险、可 review、行为不变（或可控变更）** 的整理。
> 若某些重排会牵动较大行为差异（例如数据加载完全抽象成独立 registry），可以拆成 3.4.x 逐步落地。

### 目标（DoD）

1) **更直觉的目录命名**
- `fastvideo/distillation/families/` → `fastvideo/distillation/models/`
  - 语义：这里的 “models” 指 **模型家族/管线 contract 的集成插件**（不是 YAML 的 `roles:`）。

2) **roles 容器命名统一**
- `fastvideo/distillation/bundle.py` → `fastvideo/distillation/roles.py`
- `ModelBundle` → `RoleManager`（或保持 `ModelBundle` 但在代码内逐步迁移到新名）

3) **把 infra 从 models(原 families) 中解耦合**
- dataloader 构建逻辑从 `models/*` 抽到 `fastvideo/distillation/utils/`（或 `infra/`）
- tracker 初始化从 `models/*` 抽到 `trainer/entrypoint`（更符合“infra 归 infra”）
- checkpointing 相关（目前 `fastvideo/distillation/checkpoint.py`）移动到 `utils/`（或 `infra/`）

4) **减少“文件级概念数量”**
- 已将纯 dataclass（原 `specs.py/runtime.py`）合并到 `utils/config.py`，减少“文件级概念数量”
- 已将 YAML loader（原 `yaml_config.py`）合并到 `utils/config.py`（schema+解析逻辑同处）
- `registry.py + builder.py` 可以合并/重命名为更直觉的 `dispatch.py`（保留注册表与 build_runtime 的入口）

5) **迁移策略：保证渐进、可回退**
- 保留兼容 import（re-export shim）一段时间，避免全 repo 级别大范围改动：
  - `fastvideo/distillation/families/__init__.py` re-export `fastvideo/distillation/models/*`
  - `fastvideo/distillation/bundle.py` re-export `fastvideo/distillation/roles.py` 的类型
- 更新 `fastvideo/distillation/doc/` 索引与各文件说明

### 具体设计：如何“解耦 dataloader/tracker”

#### Tracker
现状：tracker 在 `models/wan.py`（原 `families/wan.py`）里由 `_build_tracker()` 创建，并传给 validator。

Phase 3.4 目标：
- tracker 由 `fastvideo/training/distillation.py`（entrypoint）或 `DistillTrainer` 创建/持有；
- model plugin 只返回“是否需要 tracker config”（例如 raw config dict），validator 也由 method 触发调用；
- validator 构建可以延迟到 tracker 创建之后（factory/closure），避免 plugin 直接依赖 tracker。

#### Dataloader
现状：FastVideo 里 “数据 schema/预处理” 的差异主要来自 **任务/数据形态**，
并不严格等价于 model family（同一 family 内也可能有多种 schema）：

- parquet 族：`fastvideo/training/training_pipeline.py` 统一走
  `build_parquet_map_style_dataloader(..., parquet_schema=..., text_padding_length=...)`。
  - T2V：`fastvideo/dataset/dataloader/schema.py:pyarrow_schema_t2v`
  - I2V：`fastvideo/dataset/dataloader/schema.py:pyarrow_schema_i2v`
    （额外字段如 `clip_feature`/`first_frame_latent`/`pil_image`，见
    `fastvideo/training/wan_i2v_training_pipeline.py`）
  - MatrixGame：`fastvideo/dataset/dataloader/schema.py:pyarrow_schema_matrixgame`
    （额外 action cond，且不使用 text embedding，见
    `fastvideo/training/matrixgame_training_pipeline.py`）
  - ODE-init：`fastvideo/dataset/dataloader/schema.py:pyarrow_schema_ode_trajectory_text_only`
    （trajectory latents/timesteps，见 `fastvideo/training/ode_causal_pipeline.py`）
- 非 parquet：例如 LTX2 使用 `.precomputed/*.pt` 的数据形态（见
  `fastvideo/dataset/ltx2_precomputed_dataset.py`）。

因此 Phase 3.4 的目标应更准确表述为：**model plugin 不负责 data plumbing**；
dataloader 由通用层基于 `DataSpec`/`dataset_kind` 构建，而 family/adapter 只负责把
batch 转成 forward primitives 所需输入（若需要额外字段，由 `DataSpec` 显式声明）。

Phase 3.4 目标：
- model plugin **不直接构建 dataloader**，而是返回一个 `DataSpec`（或 `dataloader_factory`）描述：
  - dataset kind（parquet/webdataset/…）
  - schema/text padding length/cfg_rate 等必要参数
- `distillation/utils/data.py`（或 `infra/data.py`）统一执行 “根据 TrainingArgs + DataSpec 构建 dataloader”

这样做的收益：models(集成层) 文件更短、更聚焦在“加载模块 + 组装 adapter 需要的 shared context”。

### 文件 TODO（实现清单）

命名/结构（行为尽量不变）：
- [x] YAML schema：顶层 `models:` → `roles:`（与 `DistillRunConfig.roles` 对齐）
- [x] YAML loader：`fastvideo/distillation/yaml_config.py` → `fastvideo/distillation/utils/config.py`
- [ ] 新增 `fastvideo/distillation/models/`（拷贝/迁移原 `families/`）
- [ ] 保留 `fastvideo/distillation/families/` 作为兼容 re-export（短期）
- [ ] 新增 `fastvideo/distillation/roles.py` 并迁移 `RoleHandle/ModelBundle`
- [ ] `fastvideo/distillation/bundle.py` 变为兼容层（re-export）
- [x] `fastvideo/distillation/specs.py` + `fastvideo/distillation/runtime.py` 合并到 `fastvideo/distillation/utils/config.py`
- [ ] `fastvideo/distillation/registry.py` + `fastvideo/distillation/builder.py` 收敛为 `dispatch.py`（或最少改名）

infra 解耦：
- [ ] 新增 `fastvideo/distillation/utils/`（或 `infra/`）
  - [ ] `utils/tracking.py`：tracker 初始化（rank0 only）+ W&B run YAML 上传（如果需要）
  - [ ] `utils/data.py`：dataloader 构建（基于 `DataSpec`）
  - [ ] `utils/checkpoint.py`：checkpoint manager / config（从 `distillation/checkpoint.py` 迁移）
- [ ] `models/*`（原 families）移除 tracker/dataloader/checkpointing 的直接创建逻辑
- [ ] 更新 `utils/config.py` 的 artifacts 结构（必要时引入 factory/spec 而非直接对象）

docs：
- [ ] 更新 `fastvideo/distillation/doc/README.md` 与各文件说明（路径/命名变化）
