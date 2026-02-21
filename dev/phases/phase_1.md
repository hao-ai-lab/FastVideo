# Phase 1：把 DMD2 “抠出来”（算法与 Wan 解耦）+ Builder 雏形

Phase 0 已经证明：`Trainer/Method/Adapter/Bundle` 这套骨架是能跑通的（Wan DMD2 也已实测
能收敛/validation 变清晰）。但 Phase 0 的核心问题仍然存在：

- `WanDMD2Method` 仍在调用 legacy pipeline 的私有函数（`_generator_forward/_dmd_forward/
  faker_score_forward/...`），本质还是 **Wan+方法耦合**。
- `WanPipelineAdapter` 仍依赖 legacy pipeline 的 helper（normalize/prepare inputs/build metadata）。
- 启动路径仍然是“先起 legacy pipeline，再手工拼 bundle+method+trainer”，builder 还只是散落在
  entrypoint 里的一段 glue code。

Phase 1 的定位：**在不破坏 Phase 0 可跑性的前提下**，把算法层从 Wan/pipeline 中抽离出来，
建立稳定边界：`Method(算法)` 只依赖 `Adapter(模型 primitives)` + `ModelBundle(roles)`。

---

## Phase 1 目标（可交付）

1. 产出通用算法实现：`DMD2Method`（不再 import/调用 `fastvideo/training/distillation_pipeline.py`
   的任何私有函数）。
2. 产出可复用的 Wan 适配层：`WanAdapter`（逐步替代 `WanPipelineAdapter`，目标是不再依赖 legacy
   pipeline 的 helper；Phase 1 允许“过渡期双实现并存”）。
3. 落地 Builder 雏形：从 FastVideo 现有 args/config（`TrainingArgs/FastVideoArgs`，或新增一个
   `--models_json`）构建：
   - `ModelBundle(roles={...})`
   - `Adapter`
   - `Method`
   并让 entrypoint 变成“选择 + instantiate”，而不是手写胶水。
4. 保持 Phase 0 路径可用：`wan_distillation_v2.py` + `WanDMD2Method` 暂不删除（仅标注过渡/弃用）。

---

## Phase 1 非目标（明确不做 / 延后）

- Self-forcing v2（以及 ODE-init）—— Phase 2 再做
- role-based checkpoint/save/resume 协议—— Phase 2 再统一
- 多模型家族混搭（例如 student=Wan、teacher=SDXL）—— 先不承诺
- 完整的 validation 抽象（先保留 `method.log_validation` hook）

---

## Phase 1 TODO List（Review Checklist）

> Phase 1 的 checklist 会在实施过程中持续更新、打钩（像 Phase 0 一样）。

### A. 方法目录结构（对齐 FastGen 的“分层 catalog”思路）

- [x] 建立 methods 分层目录（至少有目录与 `__init__.py`）
  - `fastvideo/distillation/methods/distribution_matching/`
  - `fastvideo/distillation/methods/consistency_model/`
  - `fastvideo/distillation/methods/knowledge_distillation/`
  - `fastvideo/distillation/methods/fine_tuning/`
- [x] 把 Phase 1 的 `DMD2Method` 放到
  `fastvideo/distillation/methods/distribution_matching/dmd2.py`

### B. 通用 `DMD2Method`（算法层）

- [x] 新增 `DMD2Method`（算法实现），并显式 `bundle.require_roles([...])`
  - 最小要求：`student/teacher/critic`（未来可扩展 role，但 Phase 1 先按 DMD2 固定需求）
- [x] `DMD2Method` 不持有 pipeline，不调用 legacy pipeline 私有函数
- [x] 保留 Phase 0 的关键语义：
  - generator update interval（`generator_update_interval`）
  - backward 期 forward recompute 的 `forward_context` 约束（要么由 adapter 管，要么 method 管）

### C. Adapter 接口升级（让算法真正可复用）

- [x] 定义“DMD2 需要的 adapter primitives”（以 Protocol 的方式定义在 `DMD2Method` 内部）
  - `prepare_batch(...)`
  - `student_predict_x0(...)`（或“generator_pred_video”等等价语义）
  - `teacher_predict_x0(cond/uncond, guidance_scale, ...)`
  - `critic_flow_matching_loss(...)`（或拆成 critic forward + target 构造）
  - `add_noise(...) / sample_timestep(...) / shift+clamp timestep(...)`
  - `forward_context(...)`（如果决定由 adapter 托管 forward_context）
- [x] 让 `DMD2Method` 只依赖这些 primitives（而不是 Wan 细节）

### D. `WanAdapter`（模型层，逐步摆脱 pipeline helper）

- [x] 在 `fastvideo/distillation/adapters/wan.py` 中新增 `WanAdapter`
  - 输入：`training_args` + `noise_scheduler` + `ModelBundle`（或 bundle role handles）
  - 输出：实现 Phase 1 定义的 DMD2 primitives
- [x] 把以下逻辑从 legacy pipeline/helper 迁出到 adapter（Phase 1 做到“可跑通”即可）
  - `_build_distill_input_kwargs`（transformer forward 的输入组装）
  - `_get_real_score_transformer/_get_fake_score_transformer`（Phase 1 先保证 teacher 侧可选 transformer_2；critic MoE/optimizer 选择后续再补齐）
  - `denoising_step_list` 构造、warp 逻辑、`min/max_timestep`、`timestep_shift`
  - `ensure_negative_conditioning()`（Phase 0 已有，Phase 1 要确保能被复用）
- [x] `WanPipelineAdapter` 继续保留（Phase 0 兜底），并在文档/代码里标注为“legacy-backed”

### E. Builder 雏形（config -> instantiate）

- [x] 新增 builder/registry（先落 `fastvideo/distillation/builder.py`，Phase 1 仅实现 Wan）
  - `build_models(args) -> ModelBundle + Adapter`（Phase 1 先实现 Wan）
  - `build_method(args, bundle, adapter) -> DistillMethod`（DMD2 先实现）
  - 让 entrypoint 不再手写 `_build_bundle_from_wan_pipeline(...)`
- [x] 新增一个“通用 distill entrypoint”（`fastvideo/training/distill.py`）
  - CLI 通过 `--distill_model/--distill_method` 选择并运行
  - Phase 1 先支持：`wan + dmd2`
- [x] 新增一个“Phase 1 entrypoint”（`fastvideo/training/wan_distillation_v3.py`，后续会变为 wrapper）
  - Phase 1 先支持：Wan + DMD2（student/teacher/critic）
  - Phase 0 入口 `wan_distillation_v2.py` 不动（便于 A/B）

### F. 示例脚本（Phase 1）

- [x] 新增 `examples/distillation/phase1/` 的脚本
  - 目标：用户只改路径就能启动 “Wan 1.3B 学 14B，8 steps distill” 的 DMD2
  - 如果 Phase 1 引入 `--models_json`：提供一个最小 JSON 模板（写在脚本注释里）

### G. 最小单测（CPU；可选但推荐）

> 你说过 GPU test 你会自己跑；CPU test 主要用于锁定“调度/接口”不回归。

- [ ] 保留 Phase 0 的 schedule test（不改语义）
- [ ] 为 Phase 1 新增 1 个最小单测（不需要真实模型）：
  - builder 能按 role 组装 optimizers/schedulers（或 method 能正确选择 opt/sched）

---

## 关键设计决策点（出现风险就停下问你）

### 决策点 1：`forward_context` 由谁管理？

背景：FastVideo 的 backward 可能触发 forward 重算，重算必须处于正确的
`set_forward_context(current_timestep, attn_metadata)` 中。

**决定：Adapter 管理（你拍板）。**

落地方式（Phase 1 采用）：

- Method 不 import `fastvideo.forward_context`，也不直接调用 `set_forward_context(...)`
- Adapter 提供显式的上下文 API（例如 `adapter.student_context(...) / adapter.critic_context(...)`）
  以及必要时的 `adapter.backward_*()` 封装，确保 activation checkpointing 触发的 forward 重算也在
  正确的 context 中执行

如果后续实现发现 adapter 侧会导致接口/实现“非常不优雅”（比如需要过多特殊 case），我会停下并
汇报尝试与失败原因，再一起讨论是否回退到“Method 管理”或引入更好的抽象。

### 决策点 2：Builder 是否在 Phase 1 “彻底摆脱 legacy pipeline”？

Phase 1 的最小目标是“把胶水集中起来”，不必一次性重写所有加载/optimizer/dataloader 逻辑。
如果我们发现不依赖 pipeline 会导致需要大规模复制 loader/optimizer 初始化逻辑，
我会建议 Phase 1 先做：

- builder 内部仍可调用 `WanDistillationPipeline.from_pretrained(...)` 完成加载
- 但 **method/adapter 不再依赖 pipeline 私有算法函数**

是否要把“加载也完全重写”强行塞进 Phase 1，是一个风险点，需要你决定优先级。

---

## 代码落地点（具体到文件；Phase 1 预计会改/新增这些）

- 新增：
  - `fastvideo/distillation/methods/distribution_matching/dmd2.py`
  - `fastvideo/distillation/methods/distribution_matching/__init__.py`
  - `fastvideo/distillation/methods/consistency_model/__init__.py`（空壳）
  - `fastvideo/distillation/methods/knowledge_distillation/__init__.py`（空壳）
  - `fastvideo/distillation/methods/fine_tuning/__init__.py`（空壳）
  - `fastvideo/distillation/builder.py`（或 `fastvideo/distillation/builders/*`）
  - `fastvideo/training/distill_v3.py`（Phase 1 新入口，名字可再讨论）
  - `examples/distillation/phase1/*`
- 修改：
  - `fastvideo/distillation/adapters/base.py`（扩展/引入 protocol）
  - `fastvideo/distillation/adapters/wan.py`（新增 `WanAdapter`，保留 `WanPipelineAdapter`）
  - `fastvideo/distillation/methods/__init__.py`（导出新 method）
  - （可选）`fastvideo/distillation/methods/wan_dmd2.py`（标注 deprecated + 逐步迁移）

---

## Phase 1 完成标准（Definition of Done）

- `DMD2Method` 存在且可运行，且不依赖 `fastvideo/training/distillation_pipeline.py` 的私有函数
- Wan DMD2 的训练可以走 Phase 1 新入口（至少 smoke 跑通；GPU A/B 由你后续验证）
- Phase 0 入口仍可用（便于对齐/回滚）
