# Distill 重构设计（吸收 FastGen 架构）：`models={...}` + Method/Trainer/Adapter 解耦

本文是基于：

- FastVideo 当前 distill 实现：`dev/distill_structure.md`
- FastGen distillation pipeline 的优秀结构：`dev/fastgen_structure.md`

做出的“面向落地”的重构设计草案。重点是把 **算法**（DMD2/Self-forcing/…）
与 **模型/管线**（Wan/其它架构）彻底解耦，并让训练循环（Trainer）保持长期稳定。

---

## 0. TL;DR（推荐最终形态）

把 FastGen 的四层结构迁移到 FastVideo，并显式引入 `models={...}`：

- `DistillTrainer`：只做训练基础设施（循环、分布式、grad accum、logging、ckpt、validate）
- `DistillMethod`：一个“可训练对象”，封装 distill 算法 + 多角色模型 + 多优化器/交替更新
- `DistillAdapter`：把具体 pipeline/network 适配成统一的 noise/forward/CFG/cache 接口
- `ModelBundle`：`models={student, teacher, critic, ...}` 的统一容器（含 optim/ema/fsdp 策略）
- `ConditioningProvider`（或 dataset 常量注入）：显式提供 `neg_condition` 等 conditioning 常量

关键原则：**Trainer 不认识 teacher/critic，也不写 DMD/SF 的 if/else。**

---

## 1. 现状与痛点（FastVideo）

（细节见 `dev/distill_structure.md`，这里仅列架构层面的痛点）

- **Wan 耦合**：normalize/layout/transformer override 等散落在训练代码里，换模型不可能仅靠配置
- **算法分叉**：DMD2 与 Self-forcing 各自维护训练 loop，扩展新算法/新模型成本高且容易 drift
- **conditioning 隐式依赖**：`neg_condition/uncond` 可能依赖 validation 初始化等副作用
- **多网络调度不稳**：交替更新时，scheduler step 与 optimizer step 不严格对齐（会引入 lr 偏差）
- **专家/双 transformer 逻辑分散**：MoE/expert 选择与“哪个 timestep 更新哪个 expert”缺乏单点抽象

---

## 2. FastGen 架构要点（我们要吸收什么）

### 2.1 FastGen 的四层分离（核心）

FastGen 把 distillation 的复杂度拆成：

1. `Trainer`：训练循环与分布式/accum/validate/ckpt/callbacks
2. `Method(FastGenModel)`：算法 + 多网络容器 + 多优化器调度（交替更新在这里）
3. `Network`：统一 forward contract（pred_type/features/cache），并提供少量 hook
4. `Dataset/Preprocess`：提供 `real/condition/neg_condition`，并支持常量注入

这种结构的长期价值是：**训练循环不随算法增长而膨胀**，算法复用与组合能力强。

### 2.2 FastGen → FastVideo 对照表（建议直接照搬）

| FastGen 概念 | FastVideo 目标概念 | 说明 |
|---|---|---|
| `Trainer.run(model)` | `DistillTrainer.run(method)` | Trainer 只依赖统一接口 |
| `FastGenModel.single_train_step()` | `DistillMethod.single_train_step()` | 返回 `loss_map["total_loss"]` |
| `get_optimizers()/get_lr_schedulers()` | 同名接口 | 交替更新/多 optim 的唯一入口 |
| `model_dict/optimizer_dict/scheduler_dict` | 同名映射 | checkpoint 以 role 命名空间泛化 |
| `DDPWrapper.single_train_step()` | `DDPTrainStepWrapper`（可选） | 让训练 step 吃到 DDP hooks |
| dataset `files_map/presets_map` | dataset 常量注入（推荐） | `neg_condition` 不再隐式依赖 |
| callbacks + checkpointer + autoresume | 回调 + checkpoint + resume | 基础设施通用化，算法不介入 |
| meta-init + rank0 broadcast（FSDP2） | 可选的大模型加载策略 | teacher/critic ≥10B 时显著收益 |

### 2.3 我们希望复制的“具体模式”（不止抽象名词）

- **统一训练入口**：Trainer 每个（micro）step 都只做：
  - `loss_map, outputs = method_ddp.single_train_step(batch, iter)`
  - backward 只对 `loss_map["total_loss"]`
  - 在 accum 最后一步调用：
    - `method.optimizers_schedulers_step(iter)`
    - `method.optimizers_zero_grad(iter)`
- **交替更新收敛到 method**：更新 student/critic 的比例完全由
  `get_optimizers(iter)` 决定，Trainer 永不写 role-aware 的分支。
- **conditioning 显式化**：`neg_condition` 最好是 dataset 常量（或启动时一次性缓存/广播），
  绝不依赖 validation 副作用。
- **role 命名空间 checkpoint**：把保存/加载做成 “按 role key 的映射”，未来加模型不会改协议。

---

## 3. 总体架构（FastVideo 版本）

### 3.1 一次训练的总数据流（推荐）

```text
CLI/YAML config
  -> build ModelBundle(models={student, teacher, critic?, ...})
  -> build DistillAdapter.from_pipelines(bundle)  # pipeline/network 适配
  -> build DistillMethod(adapter, bundle, method_cfg)
  -> DistillTrainer(trainer_cfg, callbacks, checkpointer).run(method)
```

### 3.2 分层职责（把边界画清楚）

1. **Data/Conditioning 层**
   - dataloader 输出：`real`、`condition`、`neg_condition`（可选）以及 I2V/V2V 的额外条件
   - `ConditioningProvider`：若 dataloader 不提供 `neg_condition`，则构建并缓存/广播

2. **Adapter/Network 层（模型相关）**
   - `DistillAdapter`：layout/normalize/noise schedule/CFG/forward/(可选)cache
   - 每个架构一个 adapter：`WanAdapter`、`HunyuanAdapter`、`LTX2Adapter`…

3. **Method 层（算法相关 + 多网络训练）**
   - `DistillMethod` 基类（FastGenModel analog）
   - `DMD2Method` / `SelfForcingMethod` /（未来）`TeacherOnlyMethod` 等

4. **Trainer/Engine 层（基础设施）**
   - `DistillTrainer.run(method)`：DDP/FSDP、grad accum、日志、验证、断点、回调
   - Trainer 永不写 DMD/SF 专有逻辑

---

## 4. 核心对象与接口（建议 API）

### 4.1 `ModelBundle`：角色显式化（外部输入）

目标：让入口层显式传入 `models={student, teacher, critic, ...}`，并把所有
“训练态（optim/ema/fsdp 策略）”结构化地挂在 role 下。

```text
ModelBundle
  roles: dict[str, RoleHandle]  # key == "student"/"teacher"/"critic"/...

RoleHandle
  modules: dict[str, nn.Module]      # e.g. {"transformer": ..., "transformer_2": ...}
  frozen: bool
  precision: optional               # bf16/fp16/fp32
  fsdp_policy: optional             # shard strategy / ignored modules
  ema: optional
  optimizers/schedulers: optional
```

约定：

- `role` 只是一个字符串 key；Trainer/Checkpoint 对所有 role **一视同仁**（不做“主次”区分）。
- 为了可读性，推荐使用一些常见命名（非强制）：
  `student`, `teacher`, `critic`, `discriminator`, `reward`, `refiner`, `aux_teacher`, ...
- 每个 `DistillMethod` 应显式声明并在初始化时校验自己需要的 roles
  （例如 DMD2 需要 `student+teacher+critic`，teacher-only 只需要 `student+teacher`）。

### 4.2 `DistillAdapter`：把 pipeline/network 适配成算法可消费接口

adapter 的职责是“怎么调用模型”，而不是“什么时候更新谁”。建议接口包含：

- noise & target：
  - `add_noise(x0, noise, t) -> x_t`
  - `pred_to_x0(pred, x_t, t)`（或统一为 `pred_to_target`）
- forward：
  - `forward(role, x_t, t, cond, *, fwd_pred_type=..., neg_cond=None, cfg=None, caches=None)`
- layout & normalize（按模型需要）：
  - `to_model_layout(x)` / `from_model_layout(x)`
  - `normalize_latents` / `denormalize_latents`
- conditioning：
  - `encode_condition(raw_cond) -> cond`
  - `encode_neg_condition(raw_neg_cond) -> neg_cond`（或由 dataset 提供 embedding）
- cache（可选，Self-forcing 用）：
  - `supports_kv_cache`
  - `clear_caches(role=...)`

此外建议 adapter 暴露 capabilities，避免 method 靠 if/else 猜：

```text
adapter.capabilities = {
  "supports_cfg": True/False,
  "supports_kv_cache": True/False,
  "supported_pred_types": {...},
  "supports_features": True/False,
  "supports_expert_routing": ...,
}
```

### 4.3 `DistillMethod`：算法 + 多网络 + 多优化器调度（核心）

这是 FastGen 最值得抄的点：把 distill 的关键复杂度集中在 method。

**最小接口（强制）**

- `single_train_step(batch, iteration) -> (loss_map, outputs)`
  - `loss_map` 必须包含 `total_loss`
  - `outputs` 仅用于日志/验证/可视化
- `get_optimizers(iteration)` / `get_lr_schedulers(iteration)`
  - 返回本次 iteration 应该 step 的 optimizer/scheduler 列表（交替更新就在这里实现）
- `optimizers_schedulers_step(iteration)` / `optimizers_zero_grad(iteration)`
  - Trainer 只调用它们，不关心内部有哪些 optimizer
- `model_dict/optimizer_dict/scheduler_dict`
  - 给 CheckpointManager 使用（key == role 或 role 内模块）

**建议能力（可选但推荐）**

- `autocast()` / `grad_scaler`：统一 AMP 管理，Trainer 不关心精度细节
- `sample_for_logging(...)`：返回可调用的采样函数或采样结果，Trainer 不写采样逻辑
- `set_trainable(role, enabled)`：method 内部统一 `requires_grad_` 切换（Self-forcing/critic alternation）

> 直接收益：scheduler step 的粒度天然与 optimizer step 对齐，避免 update ratio 引入 lr 偏差。

### 4.4 `DistillTrainer`：完全算法无关的训练循环

Trainer 只依赖 method 的统一接口，推荐对齐 FastGen 的关键形态：

- grad accumulation：Trainer 计算 `sync_grads`，并在 DDP/FSDP 下用 context 禁止同步
- forward/backward：只围绕 `loss_map["total_loss"]`
- step/zero_grad：只在 accum 最后一次调用 method 接口
- validate：可复用 `single_train_step`（no_grad + autocast），并允许 method 扩展额外 eval
- callbacks：把 EMA / grad clip / logger / profiler 等都做成回调（可保存状态）

**DDP 的一个关键实现点（强烈建议照 FastGen）**

如果 `single_train_step` 不是 `forward()`，DDP 的隐式 hooks 可能不生效。
FastGen 用 `DDPWrapper` 临时把 `module.forward` 指到 `single_train_step`，
然后通过 `ddp_model(*args)` 触发 hooks。

在 FastVideo 落地时建议二选一：

1) `DistillMethod.forward = single_train_step`（简单，但 forward 被占用）
2) 实现一个 `DDPTrainStepWrapper.single_train_step()`（推荐，行为更明确）

### 4.5 `CheckpointManager`：围绕 role 命名空间泛化

建议统一协议：

- 输入：`model_dict/optimizer_dict/scheduler_dict/grad_scaler/callback_state/iteration`
- 输出：按 role key 保存（尤其是 FSDP sharded state）

并显式支持：

- **只导出 student** 的 inference 权重（teacher/critic 不随推理包分发）
- **兼容旧 distill ckpt**：至少提供一次性迁移脚本或兼容 loader

### 4.6 `Callback` 系统：把“训练周边复杂度”解耦出去

把这些都做成 callbacks（并进入 checkpoint），Trainer/Method 都不硬编码：

- EMA 更新
- grad clipping
- logging（wandb/tensorboard/本地）
- profiler/step timer
- param count / debug dumps
- autoresume（从 ckpt 恢复 callback 状态）

---

## 5. 关键设计决策（每条含原因）

### 设计 1：引入 `DistillMethod`（Method 中心，而非 Algorithm/Trainer 中心）

**设计**

- DMD2/Self-forcing/未来算法都实现为 `DistillMethod` 子类
- Method 负责多角色模型、交替更新、optim/sched/EMA、缓存生命周期等

**原因**

- distill 的“本质复杂度”就是多网络 + 多优化器调度；放在 Method 最自然
- Trainer 只需要稳定地做基础设施，长期维护成本最低

### 设计 2：`models={...}` 显式输入 + `ModelBundle` 结构化承载训练态

**设计**

- 配置/CLI 显式给出 `student/teacher/critic?`
- `ModelBundle` 统一挂载冻结策略、precision、FSDP 策略、EMA、optim/sched

**原因**

- 角色显式化是解耦的前提，且天然支持未来扩展更多角色
- checkpoint 也可以自然以 role 命名空间组织

### 设计 3：Trainer 固化成“只认一个接口”的稳定循环

**设计**

- Trainer 仅调用：
  - `loss_map, outputs = method_ddp.single_train_step(...)`
  - backward `total_loss`
  - `method.optimizers_schedulers_step()` / `optimizers_zero_grad()`
- validate 也尽可能复用 `single_train_step`

**原因**

- 彻底杜绝算法越写越多导致 trainer 分叉、难以测试和维护
- validate 复用训练逻辑能减少“训练/验证 drift”

### 设计 4：交替更新/多 optimizer 调度统一走 `get_optimizers()`（FastGen 关键模式）

**设计**

- DMD2：
  - `iter % student_update_freq == 0`：更新 student
  - 否则更新 critic（+ 可选 discriminator）
- Self-forcing：
  - 复用相同调度，只替换 student rollout

**原因**

- update ratio 的复杂度从 trainer 中消失，且扩展更多 role 不改 trainer
- scheduler step 与 optimizer step 自动对齐（减少 lr 偏差）

### 设计 5：adapter 明确 capability，而不是靠 if/else 猜

**设计**

- adapter 暴露 `capabilities`，method 启动时检查依赖
- 自适应训练：method 根据 capability 选择路径或报错/降级

**原因**

- 新增模型架构时，差异集中在 adapter；method 保持稳定
- 失败要早失败（init-time），避免跑到中途才出形状/feature 错

### 设计 6：`neg_condition` 变成“数据常量”或“启动一次性缓存/广播”

**设计**

两条路径（二选一或同时支持）：

1) dataset 常量注入：dataloader 直接输出 `neg_condition` embedding
2) provider 缓存：训练开始时用 adapter 编码 negative prompt，并缓存/广播

**原因**

- 消除 “依赖 validation 初始化 uncond embedding” 这类隐式耦合
- negative prompt embedding 通常是常量，适合缓存（性能更稳）

### 设计 7：用 role 命名空间做 checkpoint 协议（对齐 `model_dict`）

**设计**

- `model_dict/optimizer_dict/scheduler_dict` 的 key 直接是 role
- FSDP 情况下按 role key 分 shard 保存；非 FSDP rank0 写单文件

**原因**

- 多网络 distill 的保存/加载本质就是 “多个命名空间的 state”
- 未来新增/删减 role 不改变 checkpoint 顶层协议

### 设计 8：DDP 训练 step 触发 hooks（借鉴 FastGen 的 wrapper 技巧）

**设计**

- 为 DDP 场景提供 `DDPTrainStepWrapper.single_train_step()`：
  临时重定向 `forward` 到 `single_train_step`，再调用 `ddp_model(...)`

**原因**

- 让 “训练 step != forward” 的结构仍能享受 DDP 的正确行为与 hooks
- 避免强行把算法逻辑写进 `forward` 导致语义混乱

### 设计 9（可选）：meta-init + rank0 broadcast 的大模型加载

**设计**

- teacher/critic ≥10B 时，非 rank0 以 `torch.device("meta")` 构建空权重
- rank0 加载权重，FSDP wrap 后 broadcast

**原因**

- 显著减少多机启动 I/O contention 与峰值内存

---

## 6. 配置与 CLI 形态（渐进式）

### 6.1 最小可用（建议先落地）

- `--models.student <path>`
- `--models.teacher <path>`
- `--models.critic <path>`（可选）
- `--distill.method dmd2|self_forcing|teacher_only`

distill 专有参数建议用 namespace：

- `--distill.dmd2.student_update_freq 5`
- `--distill.dmd2.guidance_scale 3.5`
- `--distill.sf.num_frame_per_block 3`

### 6.2 复杂配置（建议支持）

- `--models_json path/to/models.json`
  - per-role precision/offload/trainable/fsdp_policy/ckpt_path 等

### 6.3 配置系统演进（可选吸收 FastGen 的优点）

FastGen 的 python config + instantiate + override 很优秀，但 FastVideo 现阶段可以先：

- 保留现有 YAML/argparse
- 内部把配置整理成结构化 dataclass：
  - `TrainerConfig / DataConfig / MethodConfig / ModelsConfig`
- 后续再逐步引入“延迟实例化/override”能力（不阻塞 distill 重构）

---

## 7. 目录落地建议（FastVideo 内）

建议新增 `fastvideo/distill/`（或 `fastvideo/training/distill/`），结构对齐 FastGen：

- `fastvideo/distill/bundle.py`：`ModelBundle/RoleHandle/RoleSpec`
- `fastvideo/distill/adapters/`：`WanAdapter`（先支持 Wan）、后续扩展更多模型
- `fastvideo/distill/methods/`：`base.py`、`dmd2.py`、`self_forcing.py`
- `fastvideo/distill/trainer.py`：`DistillTrainer`
- `fastvideo/distill/checkpoint.py`：`CheckpointManager`（先兼容旧格式）
- `fastvideo/distill/callbacks/`：EMA/clip/log/profiler 等

旧入口（如 `fastvideo/training/*distillation_pipeline.py`）先保留，
通过 flag 切新旧框架做 A/B 对齐。

---

## 8. 迁移计划（低风险）

### Phase 0：框架落地 + Wan(DMD2) 跑通

- 新增 `DistillTrainer/DistillMethod/ModelBundle/WanAdapter`
- `DMD2Method` 覆盖现有 Wan distill 训练（student+teacher+critic）
- checkpoint 至少能：
  - 保存/加载新格式
  - 从旧格式加载 student 初始化（兼容迁移）

### Phase 1：Self-forcing 迁移（复用 DMD2 框架）

- `SelfForcingMethod(DMD2Method)`：只覆写 student rollout / cache 生命周期
- 对齐现有 self-forcing 输出与 loss（允许数值差异但要解释）

### Phase 2：清理旧实现 + 扩展新模型/新算法

- 逐步冻结或移除旧 distill pipeline（保留兼容入口亦可）
- 新增更多 adapter（Hunyuan/LTX2/LongCat…）
- 新增更多 method（teacher-only、多 teacher、KD 轨迹蒸馏等）

---

## 9. Guardrails / 测试建议（避免重构“跑得通但不可维护”）

- **scheduler step 对齐测试**：交替更新下，未 step 的 optimizer 对应 scheduler 不应 step
- **batch_size > 1**：消除所有隐式 `B==1` 的 reshape/unflatten 假设
- **role 可选性**：critic 可选时应有清晰报错/降级路径（teacher-only）
- **conditioning 显式性**：训练开始前必须具备 `neg_condition`（来自数据或 provider）
- **checkpoint roundtrip**：save → load → loss 不发散（最小 smoke test）
