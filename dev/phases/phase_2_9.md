# Phase 2.9：A+B+Families 语义收敛（operation-centric adapter + policy 回归 method + 优雅 dispatch）

Phase 2 已经实现了“新框架可独立运行（不依赖 legacy distill pipeline）”。但从长期扩展的角度，
我们仍有三类结构性问题需要先解决，否则 Phase 3（配置语义升级 + finetune）会被迫在不稳的语义边界上继续堆功能。

本 Phase 2.9 的目标是先把语义边界收敛好（A+B+Families），并把 dispatch 做到真正优雅（N+M），
让 Phase 3 只需要在此基础上加 config schema 与新 method，而不需要再动 entrypoint/builder 的组合逻辑。

本文件最初用于 **Phase 2.9 的代码层面设计**；目前 Phase 2.9 已实现，下面 checklist 已同步打勾。
后续如果在实践过程中发现有小调整，会继续回填；遇到重大风险会停下讨论。

---

## 0) Phase 2.9 的 “A+B+Families” 是什么？

### A) operation-centric adapter API（避免 role 爆炸）

把 adapter API 从：
- `teacher_predict_x0(...) / critic_predict_x0(...) / backward_student(...) / backward_critic(...)`

收敛为：
- `predict_x0(handle, ...)`
- `backward(loss, ctx, ...)`（ctx 自带所需信息，不需要 role）

核心原则：
- adapter **不接触 role 字符串**，只接收 `RoleHandle`（handle）来取 module/optimizer 等资源。
- method 才持有 “role -> handle” 的语义映射（teacher/critic/student/reward/... 的意义只存在于 method）。

为什么要用 handle 而不是 role 字符串？
- 防止 adapter 内部出现 `if role == "teacher": ...` 这类语义泄漏，长期演化会再次耦合/role 爆炸。
- 让 adapter 更像 “family runtime primitives”，method 更像 “algorithm orchestration”。

### B) policy 回归 method（adapter 只保留 mechanics）

把 DMD2 的“策略”从 adapter 挪回 method，例如：
- timestep sampling strategy（uniform/课程/分段等）

adapter 只保留 scheduler 相关的 mechanics：
- `num_train_timesteps`
- `shift/clamp timestep` 的语义转换

### Families) build-time 语义收敛 + 优雅 dispatch（N+M）

引入 `families/` + registry 的目的：
- adapter 专注 runtime/step-time
- family 插件专注 build-time（加载 modules / shared components / dataloader / validator / tracker）
- builder/entrypoint 不写组合 if/else；新增 family 或 method 的成本为 N+M，而不是 N×M

---

## 0.5) 常见疑问（为什么这么拆？）

### backward 为什么不需要传 handle？

因为 backward 本质上只做两件事：
1) 让 autograd 通过 **已有的计算图** 反传（参数由计算图决定，不需要再“选择模块”）
2) 在 FastVideo/Wan 这种依赖全局 forward_context 的实现里，在 backward 期间 **恢复必要的上下文**

因此 adapter 的 backward 更像：
- `adapter.backward(loss, ctx)`：ctx 里带着 forward_context 所需的 timesteps/attn_metadata 等信息

handle 是为 **forward/select module** 服务的：比如选择哪个 transformer（以及 transformer_2/boundary）。而 backward 只需要“把当时 forward 用的上下文恢复回来”，不需要知道 handle。

如果未来真的出现 role/handle 相关的 backward 特殊逻辑（例如不同模块需要不同 backward context），也更推荐：
- 把 handle 放进 ctx（由 forward 返回），而不是让 adapter.backward(handle, ...) 成为常态 API。

### Families 和 Adapter 的关系是什么？

- `Family`（build-time）：负责“装配/构建”
  - 从 config 解析 roles -> 加载 modules -> 构建 `ModelBundle`
  - 构建 shared components（scheduler/vae/tokenizer/...）
  - 构建 dataloader/validator/tracker（可选）
  - **实例化 adapter**（把 bundle + shared components 注入进去）

- `Adapter`（runtime/step-time）：负责“操作/执行”
  - prepare_batch / forward_context / attention metadata
  - `predict_x0(handle, ...)` 这类 operation-centric primitives
  - backward 时恢复 forward_context

一句话：**Family 负责把一堆零件装成可运行的 runtime；Adapter 负责在训练 step 里把 FastVideo 的模型/管线差异变成可复用的操作。**

### 每个 family 都一定要有自己的 adapter 吗？能否复用？

通常会是 “一个模型家族（Wan/SDXL/Flux/...）至少有一个 adapter”，因为：
- raw batch schema 不同
- forward_context / attention metadata 的机制不同
- 预测目标（pred noise / pred x0 / flow-matching 等）的转换不同

但并不是强制 1:1：
- 一个 family 可能有多个 adapter（例如不同 pipeline 类型）
- 不同 family 也可能共享一个 adapter 基类或部分 mixin（当 runtime mechanics 很像时）

### 为什么不把 Family 和 Adapter 合并成一个东西？

可以合并，但会带来三个长期问题（这也是我们保留 Families 的核心原因）：
1) **高内聚被破坏**：build-time（下载/加载/优化器/dataloader）和 step-time（forward/backward/context）混在一起，类会膨胀成 God object。
2) **扩展成本回退到 N×M**：新增 method 时更容易被迫去改“合体的 adapter/builder”，最终又会出现各种 if/else 组合分支。
3) **语义边界变模糊**：method 想保持纯算法；adapter 想保持纯 runtime；family 想保持纯装配。合并后这些边界会互相污染，导致未来很难做 finetune/新 method 的接入。

---

## 1) Phase 2.9 交付目标（Definition of Done）

- `fastvideo/training/distillation.py` 不再硬编码 `wan + dmd2` 分支；
  改为调用通用 `build_runtime_from_config(cfg)`，并通过 registry resolve family/method。
- `WanAdapter` 对 DMD2 的暴露接口变为 operation-centric：
  - 不再暴露 `teacher_*` / `critic_*` / `student_*` 专用函数给 method 使用
  - DMD2Method 通过通用操作（如 `predict_x0(handle=...)`）完成 teacher/critic/student 的调用
- DMD2 的 timestep sampling policy 从 adapter 迁移到 method（最少把 `sample_dmd_timestep()` 挪走）。
- few-step rollout 的 step list / simulate 逻辑从 adapter 迁移到 method（未来应进一步移到 `method_config`）。
- `WanAdapter` 不应包含 method-specific 命名/概念（例如不应依赖 `*DMD*Pipeline` 这类算法命名）。
- Phase 2 的训练行为/结果应尽可能保持一致（同 config 下 loss 形态、validation 产物趋势不应漂移）。

---

## 2) 非目标（明确不做）

- 不做 YAML schema v2（`recipe` + `method_config`）升级（留到 Phase 3）。
- 不新增 finetune method（留到 Phase 3）。
- 不新增新模型家族（Phase 2.9 只整理 Wan）。
- 不追求把所有“validation / sample / prompt encode”的实现都完全脱离 pipeline（Phase 2.9 先保证训练路径独立可跑）；
  但 adapter 层需要避免 method-specific policy/rollout 泄漏，并避免 method-specific 命名耦合。

---

## 3) TODO List（Review Checklist）

### 3.1 Registry + Families（优雅 dispatch，N+M）

- [x] 新增 `fastvideo/distillation/registry.py`
  - `register_family(name)` / `register_method(name)` 装饰器
  - `get_family(name)` / `get_method(name)`（错误信息包含可用项）
  - `ensure_builtin_registrations()`：导入内置 family/method 以完成注册
- [x] 新增 `fastvideo/distillation/families/`
  - `fastvideo/distillation/families/__init__.py`
  - `fastvideo/distillation/families/wan.py`：`build_wan_family_artifacts`
    - 从 Phase 2 builder 迁移 Wan-specific build-time 逻辑：
      - 加载 role modules（transformer/transformer_2/vae 等）
      - shared components（scheduler/noise_scheduler）
      - dataloader（parquet + schema）
      - tracker
      - validator（WanValidator）
      - adapter 实例化（WanAdapter）
- [x] 改造 `fastvideo/distillation/builder.py`
  - 新增/收敛为 `build_runtime_from_config(cfg: DistillRunConfig) -> DistillRuntime`
  - `DistillRuntime.method` 类型改为 `DistillMethod`（而不是 `DMD2Method`）
  - builder 内部逻辑：
    - `family = registry.get_family(cfg.distill.model)`（Phase 2.9 暂用 `distill.model` 作为 family）
    - `method = registry.get_method(cfg.distill.method)`
    - 调用 family 构建 bundle/adapter/dataloader/tracker
    - 调用 method factory 构建 DistillMethod
- [x] 改造入口 `fastvideo/training/distillation.py`
  - 删除 `if cfg.distill.model == "wan" and cfg.distill.method == "dmd2": ...`
  - 统一走：`runtime = build_runtime_from_config(cfg)`

### 3.2 Adapter API：从 role-centric 收敛到 operation-centric（A）

- [x] 更新 `fastvideo/distillation/methods/distribution_matching/dmd2.py`
  - `_DMD2Adapter` Protocol 改为 operation-centric：
    - `predict_x0(handle, noisy_latents, timestep, batch, *, conditional, attn_kind)`
    - `backward(loss, ctx, ...)`
    - `select_module(handle, module_name, timestep)`（可选：用于 transformer_2/boundary）
    - `timestep_ops`（见 3.3）
  - 移除对 `teacher_predict_x0/critic_predict_x0/backward_student/backward_critic` 的直接依赖
- [x] 更新 `fastvideo/distillation/adapters/wan.py`
  - 把 `teacher_predict_x0/critic_predict_x0` 合并为 `predict_x0(handle=...)`
  - 把 `backward_student/backward_critic` 合并为 `backward(loss, ctx, ...)`
  - 将 `get_teacher_transformer/get_critic_transformer` 改为 `get_transformer(handle, timestep)`
    - handle 不包含“语义”
- [x] `fastvideo/distillation/adapters/wan.py` 不再出现 method-specific 命名
  - 移除 `WanDMDPipeline` 依赖，prompt encoding 改用 `WanPipeline`
  - adapter 内不再出现 `dmd/DMD` 字眼（避免“命名耦合”）

### 3.3 Timestep sampling policy 回归 method（B）

- [x] DMD2Method 内实现 timestep sampling policy：
  - `t = torch.randint(0, adapter.num_train_timesteps, ...)`（policy：uniform）
  - 然后调用 adapter 的 mechanics：
    - `t = adapter.shift_and_clamp_timestep(t)`（mechanics：shift/clamp 语义）
- [x] `WanAdapter` 去掉 `sample_dmd_timestep()`（改为提供 `shift_and_clamp_timestep()`）
- [x] few-step rollout policy 回归 method
  - denoising step list / warp mapping / multi-step simulate 全部由 `DMD2Method` 管理
  - adapter 只提供单步 primitives（如 `predict_x0` / `predict_noise` / `add_noise`）
  - TODO(Phase 3): 将 `pipeline_config.dmd_denoising_steps` 迁移到 `method_config`，避免“pipeline_config 承载算法语义”

### 3.4 兼容性与安全落地（降低风险）

- [x] （选择 direct cut）一次性迁移到 operation-centric API（不保留旧 role-centric API wrapper）
- [x] 明确哪些行为必须保持一致（不引入训练 drift）：
  - forward_context 的 ctx 捕获/恢复方式不改变
  - teacher 的 `transformer_2` boundary 逻辑不变
  - validation 路径不回退到 legacy

---

## 4) 关键接口草案（更具体一些）

### 4.1 `predict_x0(...)` 的建议签名

```text
predict_x0(
  handle: RoleHandle,
  noisy_latents: Tensor,
  timestep: Tensor,
  batch: TrainingBatch,
  *,
  conditional: bool,
  attn_kind: Literal["dense", "vsa"],
) -> Tensor
```

解释：
- `handle`：由 method 从 bundle 解析得到（`handle = bundle.role("teacher")` 等），adapter 只使用 handle 取 transformer（以及可选 transformer_2）
- `conditional`：选择 `batch.conditional_dict` 或 `batch.unconditional_dict`
- `attn_kind`：选择 `batch.attn_metadata` 或 `batch.attn_metadata_vsa`（以及对应 ctx）

### 4.2 `backward(...)` 的建议形态

```text
ctx = AdapterBackwardContext(timesteps, attn_metadata)
adapter.backward(loss, ctx, grad_accum_rounds=...)
```

ctx 不需要包含 role（role 语义由 method 管理；backward 只需要 forward_context 信息）。

### 4.3 timestep mechanics（adapter 提供）

```text
adapter.num_train_timesteps -> int
adapter.shift_and_clamp_timestep(t: Tensor) -> Tensor
```

method 决定如何 sample（policy），adapter 负责把 sample 结果转换为本模型家族的 scheduler 语义（mechanics）。

---

## 5) 风险点 / 需要你参与决策的地方

1) **一次性改动范围**：operation-centric API 迁移是否允许保留旧 API wrapper 一段时间？
   - 我建议允许，以降低重构风险；但最终目标是删除旧 API，避免长期双语义。

2) **timestep policy 的抽象边界**：
   - Phase 2.9 最小只迁 `sample_dmd_timestep`（uniform + shift/clamp）
   - 未来如果要更复杂的采样（课程学习），应该放在 `method_config`（Phase 3）

3) **registry 的注册方式**：使用显式 `ensure_builtin_registrations()` 还是 import side-effect？
   - 我建议显式 `ensure_builtin_registrations()`，避免 import 顺序导致“没注册”这类隐式 bug。

---

## 6) 备注：为什么 Phase 2.9 不做 `recipe/method_config`？

因为本 phase 的核心风险来自“语义边界调整”：
- adapter/method API 变更
- dispatch/build 结构变更

如果同时改 YAML schema（`distill` -> `recipe`）会叠加变量，出现问题时很难定位。
因此 Phase 2.9 先保证内部语义正确，再在 Phase 3 做 schema 升级与 finetune 接入。
