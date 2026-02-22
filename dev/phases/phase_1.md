# Phase 1：算法/模型解耦落地（DMD2Method + WanAdapter）+ 通用入口

Phase 1 的定位：把 distillation 从 “pipeline god object” 拆成稳定边界，让 **算法** 与
**模型家族** 解耦合，并且让训练入口开始具备 “选择 + instantiate” 的结构（对齐 FastGen
的 catalog 思路）。

目标边界（Phase 1 确认有效）：

- `DistillTrainer`：训练基础设施（accum/step/log hooks），不懂具体模型/算法细节
- `DistillMethod`：算法编排（DMD2 / Self-Forcing / CM / ...），只依赖 adapter primitives
- `DistillAdapter`：模型家族适配（Wan / CogVideoX / ...），负责 forward/backward context 与
  pipeline 细节
- `ModelBundle`：`roles -> handles`（modules/optimizers/schedulers），method 通过 role 取用

现状：Wan DMD2 的 Phase 1 训练结果已与旧 baseline 对齐，因此 Phase 0 的脚手架代码路径已移除
（见 TODO H）。

---

## Phase 1 非目标（明确延后）

- Self-forcing v2（以及 ODE-init）
- role-based checkpoint/save/resume 协议统一
- 完整的 validation 抽象（Phase 1 先保留 `method.log_validation()` hook）
- 多模型家族混搭（例如 student=Wan、teacher=SDXL）

---

## Phase 1 TODO List（Review Checklist）

### A. 方法目录结构（catalog：多层级 methods）

- [x] 建立 methods 分层目录（至少有目录与 `__init__.py`）
  - `fastvideo/distillation/methods/distribution_matching/`
  - `fastvideo/distillation/methods/consistency_model/`
  - `fastvideo/distillation/methods/knowledge_distillation/`
  - `fastvideo/distillation/methods/fine_tuning/`
- [x] `DMD2Method` 放到 `fastvideo/distillation/methods/distribution_matching/dmd2.py`

### B. 通用 `DMD2Method`（算法层）

- [x] 新增 `DMD2Method`（算法实现），并显式 `bundle.require_roles(["student","teacher","critic"])`
- [x] `DMD2Method` 不持有 legacy pipeline，不调用 legacy pipeline 私有算法函数
- [x] 保留关键语义：`generator_update_interval`（只在该 step 更新 student）

### C. Adapter primitives 契约（让算法真正可复用）

- [x] 在 `DMD2Method` 内通过 `Protocol` 定义 DMD2 所需 adapter surface（`_DMD2Adapter`）
- [x] `DMD2Method` 仅依赖 primitives，而不是 Wan 细节

### D. `WanAdapter`（模型层：forward/backward/context 全部在 adapter）

- [x] 新增 `fastvideo/distillation/adapters/wan.py::WanAdapter`
- [x] Wan 侧实现 DMD2 所需 primitives（batch prepare / teacher cfg / critic loss / backward 封装）
- [x] `forward_context` 由 adapter 托管（method 不直接触碰 `set_forward_context`）
- [x] `ensure_negative_conditioning()` 显式化（不依赖 validation 的副作用）

### E. Builder + 通用入口（config -> instantiate）

- [x] 新增 `fastvideo/distillation/builder.py::build_wan_dmd2_method`
  - Phase 1 先实现：`wan + dmd2`
- [x] 新增通用 distill 入口：`fastvideo/training/distillation.py`
  - CLI：`--distill-model` / `--distill-method`
- [x] 保留一个 Wan wrapper：`fastvideo/training/wan_distillation_v3.py`

### F. 示例脚本（Phase 1）

- [x] `examples/distillation/phase1/distill_wan2.1_t2v_1.3B_dmd2_8steps.sh`
- [x] `examples/distillation/phase1/temp.sh`（可直接改路径启动训练）

### G. 最小单测（CPU；锁定调度语义）

- [x] 保留并重命名 optimizer/scheduler 对齐测试：
  - `fastvideo/tests/distillation/test_optimizer_scheduler_alignment.py`
- [ ] （可选）为 builder 增加 1 个最小单测（不需要真实模型）

### H. 移除 Phase 0 脚手架（去除旧路径依赖）

- [x] 移除 legacy-backed `WanPipelineAdapter`
- [x] 移除旧的强耦合方法 `WanDMD2Method`
- [x] 移除旧入口 `fastvideo/training/wan_distillation_v2.py`
- [ ] 保留 `examples/distillation/phase0/`（暂存脚本用于对照；最终会统一清理）
- [x] 清理代码里残留的 `Phase 0/phase0` 调用与命名

---

## 仍然依赖 legacy code 的部分（尚未解耦干净）

Phase 1 已经把 **算法层** 从 legacy pipeline 私有函数中剥离出来，但当前仍保留 “复用 legacy
pipeline 做加载/数据/日志” 的过渡策略，主要耦合点是：

1. **加载/数据/优化器仍依赖 legacy pipeline**
   - `fastvideo/training/distillation.py` 仍通过 `WanDistillationPipeline.from_pretrained(...)`
     完成：模型加载、dataloader 构建、optimizer/scheduler 初始化、tracker 等。
   - `fastvideo/distillation/builder.py` 目前接受的输入还是 `WanDistillationPipeline`。

2. **validation 仍复用 legacy pipeline 的实现**
   - `fastvideo/distillation/adapters/wan.py::WanAdapter.log_validation()` 仍调用
     legacy 的 `pipeline._log_validation(...)`（并需要 legacy pipeline 的 validation init/RNG）。

这些耦合点将是 Phase 2 的主要清理对象（尤其是 validation + builder 输入从 “pipeline” 变为
“roles/specs”）。

---

## Definition of Done（Phase 1）

- `DMD2Method` 存在且可运行，且不依赖 legacy pipeline 私有算法函数
- Wan DMD2 训练可走 `fastvideo/training/distillation.py --distill-model wan --distill-method dmd2`
- Phase 0 代码路径已移除，避免后续继续扩散耦合
