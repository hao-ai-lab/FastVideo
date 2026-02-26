# RFC-v1: FastVideo Distillation（截至 Phase 3）

本 RFC 记录：截至 Phase 3 结束时，FastVideo 新 distillation/finetuning 框架的**目录结构**、
**每一层的职责边界**、以及这样设计的原因与当前已达成的效果。

> 参考项目：`~/alex/pkgs/FastGen`（FastGen 的 “Trainer ↔ Method ↔ Network” 分层对我们影响很大）。

---

## 1. 背景与目标

### 1.1 旧代码的问题（动机）

历史上 FastVideo 的 distillation 主要以 `fastvideo/training/*distillation_pipeline*.py` 为核心：
- **model 类型（Wan）与 distill 算法（DMD2 / self-forcing / …）强耦合**；
- training loop、算法策略、validation/sampling、conditioning 初始化混在一起；
- 扩展一个新模型或新方法，往往意味着复制/改造一整条 pipeline；
- reviewer 很难在一个 PR 里把所有 coupling 看清楚。

### 1.2 新框架的目标

我们希望把 distillation/finetuning 做成可组合的三件事：
- **Model plugin（模型家族差异）**：负责“装配（build-time）”与共享组件。
- **Method（算法）**：只关心 loss / rollout / update policy / validation policy。
- **Trainer（基础设施）**：只负责 loop/accum/step/ckpt/log，不包含策略。

并额外引入：
- **Adapter（运行时 primitive）**：把 FastVideo 的 pipeline/forward 细节收敛成 method 可复用的操作接口；
- **RoleManager / RoleHandle（多角色容器）**：支持任意角色组合（student/teacher/critic/reward/...），避免“固定三件套”的硬编码。

最终目标是：**N 个模型插件 + M 个方法 = N+M 的扩展成本**，而不是 N×M 的构建爆炸。

---

## 2. 当前目录结构（Phase 3）

下面是 `fastvideo/distillation/` 的“可读结构图”（省略 `__pycache__/`）：

```text
fastvideo/distillation/
  __init__.py
  trainer.py
  dispatch.py
  roles.py

  adapters/
    base.py
    wan.py

  models/                       # (原 families) 现在叫 models = model plugins
    components.py
    wan.py

  methods/
    __init__.py
    base.py
    distribution_matching/
      __init__.py
      dmd2.py
    fine_tuning/
      __init__.py
      finetune.py
    knowledge_distillation/     # 预留目录（Phase 3 结束时可为空/占位）
      __init__.py
    consistency_model/          # 预留目录（Phase 3 结束时可为空/占位）
      __init__.py

  validators/
    base.py
    wan.py

  utils/
    __init__.py
    config.py                   # YAML 解析 + DistillRunConfig/Runtime 数据结构
    data.py                     # dataloader 构建（当前以 parquet T2V 为主）
    tracking.py                 # tracker 构建（W&B 等）
    checkpoint.py               # save/resume 管理

  doc/
    README.md                   # file-by-file 索引与设计原则
    RFC-v1.md                   # 本文件
    ...                         # 其它 file-by-file docs
```

此外，新框架的统一入口在：
- `fastvideo/training/distillation.py`：YAML-only entrypoint（不再兼容旧 CLI configs）。

---

## 3. 为什么这样分层（核心抽象与职责边界）

### 3.1 `trainer.py`：DistillTrainer（infra only）

**职责：**
- 迭代 dataloader + gradient accumulation
- 调用 `method.single_train_step()` 得到 loss/metrics
- backward + optimizer/scheduler step/zero_grad
- checkpoint save/resume（通过 utils/checkpoint）
- tracker 记录（统一在 trainer 侧 log）

**不做的事：**
- 不知道 “Wan / CogVideoX / …”
- 不知道 “DMD2 / finetune / …”
- 不知道 “teacher/student/critic 这些角色语义”

这样做的原因（FastGen 启发）：
> FastGen 的 Trainer 非常薄，算法更新策略由 method 决定；Trainer 只做 orchestration。
这能显著降低 reviewer 读 loop 的心智负担，并避免把 update policy 固化在 Trainer 中。

### 3.2 `methods/`：DistillMethod（算法层）

**职责：**
- 定义训练一步：`single_train_step(batch, iteration, current_vsa_sparsity=...)`
  - 返回：`loss_map`（tensor）、`outputs`（任意）、`metrics`（可 log 的标量）
- 定义 update policy：`get_optimizers()` / `get_lr_schedulers()`
  - multi-optimizer / 不同 update interval 的策略属于算法的一部分
- 定义 validation policy：method 构造 `ValidationRequest`，告诉 validator：
  - 用哪个 role 的 transformer sample（通常是 student）
  - sampler_kind（ode/sde）、steps、timesteps list、guidance_scale 等

**不做的事：**
- 不去关心具体模型加载（transformer/vae/text_encoder 的来源与细节）
- 不直接依赖 FastVideo pipeline 结构（通过 adapter/validator 间接使用）

这样做的原因：
- update cadence（比如 generator_update_interval、critic 的 ratio）是算法语义，放 Trainer 会导致 Trainer 越来越“懂算法”。
- method 作为算法实现者，天然需要决定“哪些 optimizer 该 step”。

### 3.3 `roles.py`：RoleManager / RoleHandle（多角色容器）

我们把训练参与者统一视为 **role**：
- role name 是字符串 key（例如 `"student"`, `"teacher"`, `"critic"`, `"reward"`…）
- `RoleHandle` 内部持有：
  - `modules: dict[str, nn.Module]`（例如 transformer / transformer_2 / …）
  - `optimizers / lr_schedulers`
  - `trainable` 标记

**关键点：role 是可扩展的，不存在“canonical role 更高贵”的区分。**
method 决定自己需要哪些 role（通过 `bundle.require_roles([...])`）。

这样做的原因：
- distillation 形态差异巨大，硬编码固定角色集合会快速失控；
- 用 role dict 让新方法/新角色可以 additive 地接入，不影响 Trainer。

### 3.4 `models/`：Model plugin（build-time 装配层）

`models/*` 的定位是：**把一个 family 的工程差异高内聚到一个地方**，并输出 method/Trainer 需要的“运行时组件包”：
- load modules（transformer / vae / …）
- 构建 `RoleManager`（把每个 role 的 modules 放进去）
- 构建 adapter / dataloader / validator / tracker
- 产出 `ModelComponents`（`models/components.py`）

为什么需要 model plugin，而不是让 method 直接 load？
- method 关心的是算法；如果 method 去处理 “Wan 的 loader、并行切分、offload、module 名称、schema”等，会把模型工程细节污染进算法层。
- model plugin 把 build-time 的“杂活”集中起来：更高内聚、更易替换/扩展。

### 3.5 `adapters/`：Adapter（运行时 primitive 层）

adapter 是 method 与 FastVideo 运行时之间的“接口层”，应当遵循：
- **operation-centric API（按操作抽象）**，而不是 role-centric API（避免 role 爆炸）：
  - 例如：`prepare_batch(...)`、`predict_x0(handle, ...)`、`predict_noise(handle, ...)`…
- adapter 不应该硬编码 DMD2/self-forcing 的 rollout 细节；
  - rollout 的 step list/重加噪等属于 method（或 method_config）的策略。

### 3.6 `validators/`：Validator（family-specific，method-controlled）

validator 是 **模型相关** 的（例如 Wan 的采样管线、shape/latents 约定等），因此放在 `validators/wan.py`。
但 validator 不应包含任何 “DMD2-specific” 逻辑：
- method 通过 `ValidationRequest` 指定 sampler_kind（ode/sde）、timesteps、steps 等；
- validator 只负责执行与记录（生成视频、写 mp4、通过 tracker 上传 artifacts）。

这能同时满足：
- validator 不被算法污染（保持复用性）
- 不同方法可以用同一个 validator，但采样策略由方法决定

### 3.7 `dispatch.py`：优雅 dispatch（避免 N×M builder）

`dispatch.py` 提供 registry：
- `@register_model("wan")`：注册 model plugin builder
- `@register_method("dmd2") / @register_method("finetune")`：注册方法 builder

统一入口只需做一次组合：
- `recipe.family` → 选择模型插件
- `recipe.method` → 选择算法方法

扩展一个新模型或新方法只需要新增一个 plugin 文件并注册，不需要写 25 个 “model×method 组合函数”。

---

## 4. YAML config 语义（Phase 3 的 schema）

Phase 3 采用 **YAML-only**（不兼容旧 CLI configs），由 `utils/config.py` 解析为：
- `RecipeSpec`（dataclass）：`recipe.family` + `recipe.method`
- `roles: dict[str, RoleSpec]`（dataclass）：每个 role 的 family/path/trainable…
- `training_args: TrainingArgs`（dataclass）：训练超参（直接映射 FastVideoArgs/TrainingArgs）
- `method_config: dict[str, Any]`：方法私有参数（保持灵活，便于快速迭代）

一个典型结构：

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
    path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    trainable: false
  critic:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

training:
  # 这里基本就是 TrainingArgs 的字段
  max_train_steps: 4000
  learning_rate: 1.0e-6
  VSA_sparsity: 0.7
  ...

pipeline_config:
  flow_shift: 3
  sampler_kind: ode

method_config:
  # 仅 method 关心的内容
  dmd_denoising_steps: [999, 750, 500, 250, 0]
  attn_kind: vsa
```

为何 `method_config` 是 dict，而不是强类型 dataclass？
- 方法参数变化频繁（研究迭代快），强类型会导致 schema/迁移成本高；
- 但 `recipe/roles/training_args` 是框架稳定边界，需要结构化来做 invariants 与错误提示。

---

## 5. 端到端执行路径（从命令到训练）

```
fastvideo/training/distillation.py
  -> utils.config.load_distill_run_config()
  -> dispatch.build_runtime_from_config()
       -> models/<family>.py: build_*_components()
       -> methods/<method>.py: build_*_method()
  -> DistillTrainer.run(method, dataloader, ...)
```

工程性改进：
- entrypoint 会把本次运行的 YAML 原封不动通过 tracker 上传（例如 W&B Files），便于复现与审阅。

---

## 6. Phase 3 已达成的效果（可验证的“产出”）

截至 Phase 3，框架已经能做到：

1) **完全摆脱 legacy distillation pipeline 的训练 loop**
- distill/finetune 都走统一入口 `fastvideo/training/distillation.py`
- Trainer/Method/Adapter/Validator 的边界清晰

2) **DMD2 few-step distillation 与 SFT/finetune 均可端到端跑通**
- DMD2：method 控制 rollout（SDE-style）、更新策略、validation 的采样步与 timesteps
- Finetune：只需要 student role，同框架内自然表达（“distillation 的特例”）

3) **VSA（Video Sparse Attention）可作为 backend 使用，并可 schedule/记录**
- `training.VSA_sparsity / VSA_decay_*` 可影响训练时 `current_vsa_sparsity`
- trainer 统一 log `vsa_sparsity`
- validator 确认会使用 `training_args.VSA_sparsity`（forward-time）

4) **扩展路径更清晰**
- 增加一个新方法：新增 `methods/<category>/<method>.py` + `register_method()`
- 增加一个新模型家族：新增 `models/<family>.py` + `adapters/<family>.py` + `validators/<family>.py` + `register_model()`

---

## 7. 已知限制与下一步（面向 Phase 4+）

Phase 3 结束时仍有一些“刻意未做”的点：
- 多 family 角色组合（例如 student=wan, teacher=sdxl）尚未正式支持；
  - 未来可能需要 method 持有多个 adapter/validator，或在 roles 层引入 cross-family 约束检查（例如 VAE/latent space 是否兼容）。
- 更多方法类别（KD/CM/self-forcing 等）与更多 model plugins 仍需逐步落地。
- dataloader schema 更系统的抽象（DataSpec）目前仍偏工程化，不影响核心分层但有优化空间。

---

## 8. 附：FastGen 对我们的直接启发点（总结）

我们从 FastGen 学到的最关键结构是：
- **Trainer 薄、Method 厚（策略在 method）**；
- config 选择 method/network，训练时 Trainer “只看 method”。

在 FastVideo 中，我们保留这条主线，同时增加了：
- **Adapter**（吸收 FastVideo pipeline/forward 的差异，让 method 可复用）
- **Model plugin（build-time 装配层）**（吸收 loading/dataloader/validator/tracker 等工程差异）
- **RoleManager**（支持任意角色组合）

这套结构让 FastVideo distillation 能以“增量 PR”方式逐步替换 legacy pipeline，而不是一次性大爆炸式重写。
