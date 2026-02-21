# Phase 0：新 Distill 框架落地 + Wan(DMD2) 跑通（实践记录/执行清单）

目标（Phase 0 的“可交付”）：

1. 在 FastVideo 内落地一套 **Method/Trainer/Adapter/Bundle** 的 distill 框架骨架，
   代码可 import，可扩展，不影响现有 `fastvideo/training/*distillation_pipeline.py`。
2. 用新框架跑通 **Wan 的 DMD2**（student+teacher+critic），并提供一个独立入口脚本，
   便于 A/B 对齐旧实现。
3. 在 Phase 0 就消除一个当前硬耦合：**DMD2 所需的 uncond/neg_condition 不再依赖 validation 副作用**。

非目标（Phase 0 暂不强求）：

- 完整复刻旧 pipeline 的所有日志/可视化/validation 产物（可以逐步补）
- role-based 通用 checkpoint 协议（Phase 0 先复用现有 distill ckpt utils，后续再迁移）
- 支持除 Wan 以外的模型（Phase 0 只做 `WanAdapter`）

---

## 0. 关键风险与应对

### 风险 A：`negative_prompt_embeds` 目前只在 validation 中被初始化

现状：`fastvideo/training/distillation_pipeline.py` 里，`_prepare_dit_inputs()` 只有在
`self.negative_prompt_embeds` 已经存在时才会构造 `unconditional_dict`；
而 `self.negative_prompt_embeds` 是在 `_log_validation()` 里通过
`validation_pipeline.prompt_encoding_stage(...)` 赋值的。

如果不跑 validation，DMD2 会在 real_score 的 uncond forward 直接报错
（`text_dict cannot be None`）。

Phase 0 方案：引入一个最小的 `ConditioningProvider`（Wan 版本），在训练开始前：

- 从 `SamplingParam.from_pretrained(model_path).negative_prompt` 取 negative prompt 字符串
- 用一个轻量的 prompt encoder（优先复用 `WanDMDPipeline` 的 `prompt_encoding_stage`）
  计算 `negative_prompt_embeds / negative_prompt_attention_mask`
- 缓存到 adapter/method，并确保每个 step 都能显式提供 uncond conditioning

这一步是 Phase 0 必做，不然新 Trainer/Method 没法脱离 validation。

### 风险 B：Phase 0 会短期“仍然 Wan 耦合”

为了尽快跑通 + 降低风险，Phase 0 允许 `WanAdapter` 通过 **wrap 现有 pipeline 的 helper
methods**（normalize/noise/timestep/attention metadata/build_input_kwargs 等）实现。

后续 Phase 1/2 再把这些 helper 从 pipeline 迁移/重写进 adapter，彻底摆脱旧实现。

---

## 1. 代码落地点（具体到文件）

> 约定：Phase 0 把新框架放到 `fastvideo/distillation/`（目前该目录为空）。

### 1.1 新增 distill 框架骨架

- `fastvideo/distillation/__init__.py`
  - 导出 Phase 0 需要的核心类（Trainer/Method/Bundle）
- `fastvideo/distillation/bundle.py`
  - `RoleHandle` / `ModelBundle`：`roles: dict[str, RoleHandle]`
- `fastvideo/distillation/trainer.py`
  - `DistillTrainer`：通用训练循环（grad accum + step/zero_grad），不认识 roles
- `fastvideo/distillation/methods/base.py`
  - `DistillMethod` 抽象：`single_train_step()`、`get_optimizers()` 等
- `fastvideo/distillation/adapters/base.py`
  - `DistillAdapter` 抽象：`prepare_batch()`、`forward_*()`、conditioning provider hook

### 1.2 Phase 0 的 Wan 实现（pipeline-backed，先跑通）

- `fastvideo/distillation/adapters/wan.py`
  - `WanPipelineAdapter`：
    - 复用 `fastvideo/training/distillation_pipeline.py` 的 helper 方法做数据准备/forward
    - 提供 `ensure_negative_conditioning()`，不依赖 validation
- `fastvideo/distillation/methods/wan_dmd2.py`
  - `WanDMD2Method`：
    - 实现 DMD2 的 loss 计算（generator_loss + fake_score_loss）
    - 实现 update schedule（`generator_update_interval`）与 optimizer/scheduler step 对齐

### 1.3 独立入口（不影响旧脚本）

- `fastvideo/training/wan_distillation_v2.py`
  - 行为与 `wan_distillation_pipeline.py` 类似，但走新框架：
    - 构建 `WanDistillationPipeline.from_pretrained(...)`（仅用于复用现有加载/优化器/dataloader）
    - 构建 `WanPipelineAdapter` + `WanDMD2Method`
    - 用 `DistillTrainer.run(...)` 启动训练

### 1.4 最小单测（CPU 可跑）

- `fastvideo/tests/distillation/test_phase0_schedule.py`
  - 只测 method 的 optimizer/scheduler 选择逻辑是否与 update ratio 对齐
  - 不依赖 GPU/模型权重

---

## 2. Phase 0 训练循环的行为约定（便于 A/B）

为了尽量可对齐旧实现，Phase 0 的新 Trainer 约定：

- global `step` 仍然按旧 pipeline 的语义：从 `init_steps+1` 开始，到 `max_train_steps`
- grad accumulation 由 Trainer 处理（每个 microbatch 调一次 `single_train_step`，最后 step）
- generator optimizer/scheduler 只在 `step % generator_update_interval == 0` 时 step
  （这是对旧实现的一个显式修正：旧实现会每步 step generator scheduler）
- fake_score optimizer/scheduler 每步 step

关于 backward 的一个 Phase 0 现实约束：

- 由于 FastVideo 的 attention/kernel 依赖 `set_forward_context(...)`，并且训练里常开
  activation checkpointing，**backward 可能触发 forward 重算**，重算时也必须处于正确的
  forward_context 里。
- 旧实现通过在 backward 前重新 `set_forward_context` 来保证这一点（且 generator/critic
  的 context 可能不同）。
- 因此 Phase 0 的接口在 `DistillMethod` 里增加 `backward(loss_map, outputs, grad_accum_rounds)`
  这个 hook：Trainer 调用它，但不关心里面怎么拆分 loss/怎么设置 context。
  默认实现仍然是对 `total_loss` 做 backward；Wan(DMD2) method 会覆写为
  “generator_loss 在 vsa context 下 backward + fake_score_loss 在 normal context 下 backward”。

> 如果后续发现这个 scheduler 行为变化会影响 A/B 对齐，我们可以在 Phase 0
> 加一个 “legacy 模式开关”；但默认先按“optimizer step 对齐 scheduler step”的正确语义实现。

---

## 3. 开始实践（本次提交会先做到什么程度）

本次实现优先级：

1. 新框架骨架文件可 import（`fastvideo.distillation.*`）
2. `WanPipelineAdapter.ensure_negative_conditioning()` 可在无 validation 的情况下生成 neg embeds
3. `WanDMD2Method.single_train_step()` 能产出 `loss_map["total_loss"]`
4. `DistillTrainer.run()` 能跑若干 step（最小 smoke）并 step optimizers
5. 加一个 schedule 单测，确保 `get_optimizers/get_lr_schedulers` 与 update ratio 对齐

后续增量（Phase 0 内可迭代）：

- checkpoint/resume 接入（优先复用 `save_distillation_checkpoint/load_distillation_checkpoint`）
- validation 接入：已通过 `DistillTrainer` -> `method.log_validation(step)` hook
  接入旧 pipeline 的 `_log_validation`（见 `WanDMD2Method.log_validation()`）

---

## 4. “大设计硬伤”停工汇报标准

如果在 Phase 0 实践过程中出现以下情况，我会暂停继续写代码并直接汇报你：

- `models={...}` + adapter 的抽象无法覆盖 Wan 的关键差异（例如 conditioning/CFG 方式根本不一致）
- DMD2 的计算图要求导致 Method/Trainer 的边界必须反转（Trainer 不可算法无关）
- 现有 pipeline 的 helper 复用导致强耦合无法逐步迁移（必须一次性大重构才可跑通）

---

## 5. Phase 0 的“耦合债务”与命名说明（非常重要，避免未来遗忘）

### 5.1 为什么现在会有 `WanDMD2Method` 这种名字？

结论：这是 **Phase 0 的过渡实现**，名字里带 `Wan` 是“刻意暴露耦合”，防止误用。

原因：当前 `fastvideo/distillation/methods/wan_dmd2.py` 并不是一个纯算法层的 DMD2。
它直接复用/依赖了旧实现 `fastvideo/training/distillation_pipeline.py` 的 Wan-only 私有逻辑：

- DMD2 的关键计算来自旧 pipeline 的内部函数：`_dmd_forward(...)`、`faker_score_forward(...)`、
  `_generator_forward(...)` 等（它们隐含了 layout/normalize/CFG/uncond 等具体假设）
- `fastvideo/distillation/adapters/wan.py` 也在复用旧 pipeline 的 helper：
  `_normalize_dit_input/_prepare_dit_inputs/_build_attention_metadata`

因此它在语义上等价于：**“把旧 Wan distill pipeline 包了一层 Method/Trainer 外壳”**，
而不是一个可对接任意 adapter 的“通用 DMD2Method”。

### 5.2 FastGen 有没有类似的做法？

FastGen 的命名与分层更“干净”：

- 算法层叫 `DMD2Model`（算法名），不会叫 `WanDMD2Model`
- 网络/架构差异在 `networks/*` + config 里选择（网络与算法解耦）

所以我们现在的 `WanDMD2Method` 更像是 Phase 0 的迁移脚手架，而不是最终形态。

### 5.3 TODO（必须做）：把 `WanDMD2Method` 演进为 **算法名 method + 模型名 adapter**

为了避免“又一次耦合到 Wan”，必须把 Phase 0 的耦合逐步清掉，目标对齐 FastGen：

1) **把算法从旧 pipeline 里抠出来**
   - 新增：`fastvideo/distillation/methods/dmd2.py`（`DMD2Method`，不依赖任何具体模型）
   - `DMD2Method` 只依赖 adapter 提供的 primitives（noise/pred_to_x0/teacher_cfg/critic_loss 等）

2) **把模型差异收敛到 adapter（WanAdapter）**
   - 演进：`WanPipelineAdapter` -> `WanAdapter`
   - `WanAdapter` 不再调用 `DistillationPipeline` 的私有 helper 方法，
     而是自己实现 normalize/layout/attention metadata/输入 kwargs 组装等

3) **最终命名与入口应变成**
   - `DMD2Method + WanAdapter`（method 不带模型名）
   - `fastvideo/training/wan_distillation_v2.py` 里只选择 adapter，不再选择“WanDMD2Method”

4) **迁移后应删除/冻结 Phase 0 的 pipeline-backed 版本**
   - 避免未来复制粘贴 `WanDMD2Method` 去做其它模型（那会把耦合扩散）

> 备注：Phase 0 用 `WanDMD2Method` 的意义是“先把训练循环与多 optimizer 调度结构稳定下来”，
> 但我们必须把它当成临时脚手架，Phase 1/2 逐步替换为真正解耦的 method+adapter。
