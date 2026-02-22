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
  trainable: bool
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

> Phase 2 开始，新的 distillation 入口 **不再兼容旧式 CLI 传参**。
> 我们只接受新的结构化配置（YAML），让一次运行可读、可复现、可审查。

### 6.1 最小可用（建议先落地）

**Phase 2+ 目标**：一个 YAML 配置文件描述一次运行（distill/finetune/…），入口只需要：

- `fastvideo/training/distillation.py --config path/to/run.yaml`

除此之外的训练参数/模型选择/方法选择，都写入 YAML。

### 6.2 复杂配置（建议支持）

- `--models_json path/to/models.json`
  - per-role precision/offload/trainable/fsdp_policy/ckpt_path 等

### 6.3 YAML schema v2（Phase 3）：`recipe` + `method_config`

说明：
- Phase 2 的 YAML schema v1 使用 `distill:` 顶层（历史原因）
- Phase 3 将升级为 schema v2：用 `recipe:` 顶层，并引入 `method_config:`（语义更通用）

schema v2 的 “单次运行” 配置示意（字段可迭代）：

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

training:
  output_dir: outputs/...
  max_train_steps: 4000
  seed: 1000
  # ... (TrainingArgs/FastVideoArgs 的字段)

pipeline_config:
  # 支持直接内联覆盖，也支持只给 pipeline_config_path
  # pipeline_config_path: fastvideo/configs/wan_1.3B_t2v_pipeline.json
  flow_shift: 8

method_config:
  # method-specific 超参（不进入 TrainingArgs；由 method/adapter 自行解析）
  generator_update_interval: 5
  real_score_guidance_scale: 3.5
```

**解析策略（最优雅且低风险）**

- 新入口的 parser 只保留 `--config run.yaml`（以及少量 meta flags，如 `--dry-run`）。
- 训练相关的所有参数（TrainingArgs/FastVideoArgs/pipeline_config/method/models）都来自 YAML。
- 解析流程：
  1) `yaml.safe_load` 得到 dict
  2) 规范化/校验 schema（recipe/models/training/pipeline_config/method_config/...）
  3) 将 `training:` 与 `pipeline_config:` 合成 kwargs，调用 `TrainingArgs.from_kwargs(**kwargs)`
     （由现有 PipelineConfig/PreprocessConfig 负责子配置实例化与校验）

这样不需要推翻现有 TrainingArgs/FastVideoArgs 体系，但从入口层面彻底摒弃旧式 CLI 传参方式。

### 6.4 `outside/` overlay（Phase 2 约束下的 workaround）

我们不能直接修改大项目里的 `fastvideo/configs/`（避免冲突/合并成本）。
因此 Phase 2 建议在 distillation 侧新增一个 overlay 根目录：

- `fastvideo/distillation/outside/`

并约定：

- 把“本应在外部 repo 存在的新增/改版配置”放进：
  - `fastvideo/distillation/outside/fastvideo/configs/...`
- distillation 入口 **不做任何自动补全/overlay 重写**：
  - 用户传入的 `--config` 必须是一个真实存在的文件路径（通常位于 `outside/` 下）
  - config 内引用的其它路径（如 `pipeline_config_path`）也必须是 **真实路径**

这让我们可以在不侵入主仓库配置的情况下，迭代 YAML/JSON config、做实验性变更，
同时不影响 legacy 代码路径。

**实现注意**

- 不建议把 `outside/` 直接插入 `sys.path` 去 shadow 整个 `fastvideo` 包（风险太高、调试困难）。
- 推荐把 `outside/` 仅作为 **外部配置存放目录**（YAML/JSON），避免运行时“魔法寻路”。
- 如果确实需要覆盖 Python config（`.py`）：
  - 用 `importlib` 的“按文件路径加载模块”方式加载为独立 module name，避免影响全局 import。

### 6.5 配置系统演进（可选吸收 FastGen 的优点）

FastGen 的 python config + instantiate + override 很优秀，但 FastVideo 现阶段可以先：

- 保留现有 YAML/argparse
- 内部把配置整理成结构化 dataclass：
  - `TrainerConfig / DataConfig / MethodConfig / ModelsConfig`
- 后续再逐步引入“延迟实例化/override”能力（不阻塞 distill 重构）

---

## 7. 目录落地建议（FastVideo 内）

Phase 0 的实践表明：先把新框架以 **additive** 方式落地到一个独立目录最稳妥。
目前已选择并落地在 `fastvideo/distillation/`（建议继续沿用该路径，避免再迁一次目录）。

建议结构（已部分实现）：

- `fastvideo/distillation/bundle.py`：`ModelBundle/RoleHandle`
- `fastvideo/distillation/adapters/`：`WanAdapter`（Phase 1 已落地；后续新增更多 adapter）
- `fastvideo/distillation/methods/`：`base.py`、`distribution_matching/dmd2.py`、（目标）`self_forcing.py`
- `fastvideo/distillation/trainer.py`：`DistillTrainer`
- `fastvideo/distillation/builder.py`：把 “config -> roles -> bundle/adapter/method” 的胶水集中起来
- `fastvideo/training/distillation.py`：通用入口（YAML-only：`--config path/to/run.yaml`）
- （后续）`fastvideo/distillation/checkpoint.py`：role-based `CheckpointManager`（先兼容旧格式）
- （后续）`fastvideo/distillation/callbacks/`：EMA/clip/log/profiler 等

旧入口（如 `fastvideo/training/*distillation_pipeline.py`）先保留，
通过 flag 切新旧框架做 A/B 对齐。

---

## 8. 迁移计划（低风险）

### Phase 0（已完成）：框架落地 + Wan(DMD2) 跑通（过渡实现）

Phase 0 的定位在实践中更明确了：它是“**把旧 Wan distill pipeline 包一层新框架壳**”，
先把训练循环/多 optimizer 调度/validation hook 等基础设施固定下来，再逐步解耦。

- ✅ 新增 `DistillTrainer/DistillMethod/ModelBundle` 的骨架，并跑通 WAN distill
- ✅ 用单测锁定关键语义：scheduler step 与 optimizer step 对齐
  - `generator_update_interval > 1` 时不会“空 step scheduler”
- ✅ 为后续解耦铺路：把 “roles={student,teacher,critic}” 显式化到 bundle

Phase 0 明确没有做（刻意延期）：

- ❌ v2 path 的 checkpoint/save/resume（role-based）
- ❌ `DMD2Method` 的真正算法解耦（目前仍调用旧 pipeline 内部函数）
- ❌ Self-forcing v2 迁移

### Phase 1（已完成）：算法与模型真正解耦（先把 DMD2 “抠出来”）

Phase 1 的核心目标：把 Phase 0 的“脚手架耦合”逐步替换为 **Method(算法) + Adapter(模型)**
的稳定边界，让其它模型/其它方法可以复用 Trainer。

Phase 1 的“辉煌”（落地与收益）：

- ✅ 通用算法 method：`fastvideo/distillation/methods/distribution_matching/dmd2.py::DMD2Method`
  - 算法层不再调用 legacy pipeline 私有算法函数
  - 依赖面缩到 adapter primitives（通过 `Protocol` 约束 surface）
- ✅ 真正的 WAN 适配层：`fastvideo/distillation/adapters/wan.py::WanAdapter`
  - `forward_context` 与 backward 重算约束收敛到 adapter（method 只实现算法）
  - `ensure_negative_conditioning()` 显式化（不再依赖 validation 的隐式副作用）
- ✅ Builder 雏形：`fastvideo/distillation/builder.py`
  - 把 “roles -> bundle -> method” 的胶水集中在一处，便于扩展新 method/new model
- ✅ 通用入口：`fastvideo/training/distillation.py`
  - Phase 1 仍是 CLI 选择：`--distill-model` + `--distill-method`
  - Phase 2 起将切换为 **YAML-only**（见第 6 节），并逐步废弃这套 CLI
- ✅ 训练效果对齐：Phase 1 跑出来的 WAN DMD2 与 Phase 0/baseline 行为一致（已实测）

### Phase 2（已完成）：彻底脱离 legacy distill pipeline（让新框架可独立存在）

你提的建议我同意：Phase 2 应该把 Phase 1 仍然残留的 legacy 依赖清干净，让新的 distill
代码路径可以 **不依赖** `fastvideo/training/*distillation_pipeline.py` 和
`WanDistillationPipeline` 仍可运行训练与验证。

为了降低风险，建议 Phase 2 按 “先 validation、再 builder/runtime、最后清理入口” 的顺序推进。

#### Phase 2.1：Validation 独立化（优先级最高，收益最大）

- 目标：`WanAdapter.log_validation()` 不再调用 legacy `pipeline._log_validation(...)`
- 建议实现：
  - 新增 `fastvideo/distillation/validation/`（或 `fastvideo/distillation/validators/`）
  - 由 adapter 提供 `build_validator(...)` 或直接实现 `adapter.sample(...)`
  - 复用模块化 inference pipeline（例如 `fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py`）
    来生成视频并交给 tracker 记录
- 收益：彻底消除 “validation 初始化副作用/属性缺失” 这类隐式耦合与脆弱点

#### Phase 2.2：Builder/Runtime 脱离 pipeline（roles/spec -> instantiate）

- 目标：`fastvideo/training/distillation.py` 不再先 instantiate `WanDistillationPipeline`
- 建议实现：
  - 定义结构化 spec：`RoleSpec/ModelSpec`（role -> {family, path, precision, trainable,...}）
  - 配置形态落地（Phase 2 必做）：
    - `--config path/to/run.yaml`（YAML 为 single source of truth；CLI 仅指定配置路径）
    - `outside/` workaround：把新增/实验性 configs 放在 `outside/`，入口只接受真实路径（不做 overlay 寻路）
    - （可选）保留 `--models_json` 作为“程序生成配置”的接口
  - builder 根据 spec：
    - 加载 modules（student/teacher/critic）
    - 构建 role-based optimizers/schedulers
    - 组装 `ModelBundle + Adapter + Method`
    - 构建 dataloader（直接复用 dataset 代码，不经由 legacy pipeline class）
  - 不新增入口文件：直接增强 `fastvideo/training/distillation.py`，并把它定义为 **YAML-only distill entrypoint**
    - 仅支持 `--config run.yaml`（以及少量 meta flags），不再兼容旧式 CLI configs
    - legacy distill 继续通过原有 `fastvideo/training/*distillation_pipeline.py` 入口运行（两套路径并存）
- 收益：distill 路径具备真正的“模型/算法 catalog + instantiate”，开始能支持更多模型家族

#### Phase 2.3：role-based checkpoint/save/resume（新框架自洽）

- 目标：新框架训练可 save/resume，且协议围绕 role 命名空间（不再绑死 WAN pipeline）
- 建议实现：
  - `fastvideo/distillation/checkpoint.py`：保存/加载 modules + optimizers + schedulers + RNG states
  - 明确兼容策略：兼容旧格式（若必要）或提供一次性转换脚本

#### Phase 2.4（Deferred）：收敛与清理（暂不做；完全解耦后手动处理）

本轮 Phase 2 采用 **非侵入式** 策略：只新增新路径所需的代码，不做 legacy 代码搬家/清理。
当 Phase 2.1/2.2/2.3 全部完成、并且新框架可以独立运行后，再由你手动清理旧入口/旧实现。

在 Phase 1 的稳定边界之上，Phase 2 再做“功能扩展 + 旧实现收敛”：

- Self-forcing v2：`SelfForcingMethod(DMD2Method)`（只覆写 student rollout / cache 生命周期）
  - 并把 ODE-init（若需要）归类为 **student 初始化策略**（builder/config 层），而不是 Trainer 特例
- role-based checkpoint/save/resume（v2 path）
- 新增更多 adapter（Hunyuan/LTX2/LongCat…）
- 新增更多 method（teacher-only、多 teacher、KD 轨迹蒸馏等）
- 逐步冻结或移除旧 distill pipeline（保留兼容入口亦可）

### Phase 3（计划）：优雅 dispatch + Recipe config + Finetuning（统一到同一框架）

Phase 3 的定位：在 Phase 2 已经证明“新 distill 框架可独立运行”的基础上，解决两个长期
扩展的核心问题：

1) **真正优雅的 dispatch（避免 N×M builder 组合爆炸）**  
2) **配置语义升级（`distill` -> `recipe`，引入 `method_config`）**  
3) **把 finetuning 作为一种 method 接入框架**（只需要 `student` + dataset）

#### Phase 3.1：真正优雅的 dispatch（N+M，而不是 N×M）

目标：新增第 5 个模型家族 + 第 5 个算法时，不需要写 25 个 `build_<model>_<method>()`。

核心思路：把 “可组合的变化” 拆成两类 registry，然后用 adapter capability/protocol 做约束：

- **Model family registry**（按 `recipe.family` 注册）
  - 负责：按 role 加载 modules、构建 adapter、构建 validator、构建 dataloader（或 data hooks）
- **Method registry**（按 `recipe.method` 注册）
  - 负责：构建 method（算法）；声明 `required_roles`；声明需要的 adapter primitives（Protocol 或 capability）

入口层只做组合（伪代码）：

```text
cfg = load_run_config(...)
family = FAMILY_REGISTRY[cfg.recipe.family]
method = METHOD_REGISTRY[cfg.recipe.method]

bundle = family.build_bundle(cfg.models, cfg.training, cfg.pipeline_config)
adapter = family.build_adapter(bundle, cfg.training, cfg.pipeline_config, cfg.method_config)
validator = family.build_validator(...)  # optional
dataloader = family.build_dataloader(cfg.training, cfg.data?)  # optional

distill_method = method.build(bundle=bundle, adapter=adapter, method_config=cfg.method_config)
trainer.run(distill_method, dataloader, ...)
```

这样新增扩展的成本是：
- 新模型家族：新增 1 个 family plugin（N）
- 新算法：新增 1 个 method plugin（M）
- 组合不需要额外代码（不再写 N×M）

实现落点（建议，Phase 3 落地到代码时再细化）：
- `fastvideo/distillation/registry.py`
  - `register_family(name)(cls)` / `register_method(name)(cls)` 装饰器
  - `get_family(name)` / `get_method(name)` + “可用项”错误提示
- `fastvideo/distillation/builder.py`
  - 收敛为 `build_runtime_from_config(cfg)`（通用），内部查 registry
  - Wan 的加载逻辑迁移为 `WanFamily` plugin（保留当前 Phase2 的 loader 复用）

#### Phase 3.2：配置语义升级（`distill` -> `recipe`，引入 `method_config`）

动机：
- `distill.method=finetune` 语义别扭，因为 finetune 是一种训练 recipe，不一定是“蒸馏”。
- method-specific 参数长期塞进 `training:`（TrainingArgs）会让配置语义越来越混杂。

Phase 3 计划把 YAML schema 升级为：

```yaml
recipe: {family: wan, method: dmd2}   # 只负责 “选什么”
models: {student: ..., teacher: ...}  # 参与者
training: {...}                       # infra 参数（映射到 TrainingArgs）
pipeline_config: {...}                # pipeline/backbone config（模型侧）
method_config: {...}                  # algorithm/method 超参（方法侧）
```

同时保持与 FastVideo 现有语义对齐：
- 入口层会根据 `recipe.method` 推导 `TrainingArgs.mode`
  - `finetune` -> `ExecutionMode.FINETUNING`
  - 其它 distillation methods -> `ExecutionMode.DISTILLATION`

迁移策略（建议）：
- Phase 3 先把 `method_config` 作为新增字段引入，并逐步把以下参数从 `training:` 挪过去：
  - DMD2：`generator_update_interval`, `real_score_guidance_scale`, `simulate_generator_forward`, ...
  - Self-forcing：ODE-init / cache / rollout 策略相关参数
  - Finetune：loss/target/pred_type 等
- `training:` 保持 “trainer/infra” 语义（分布式、优化器、ckpt、logging、数据路径等）。

#### Phase 3.3：Finetuning 作为一种 method 接入（only student）

目标：让 finetuning 跟 distillation 一样走同一套：
`ModelBundle + Adapter + Method + Trainer + (Validator/Checkpoint)`。

建议落地形态（Phase 3 落地到代码时）：
- 新增 method：`fastvideo/distillation/methods/fine_tuning/finetune.py::FineTuneMethod`
  - `bundle.require_roles(["student"])`
  - 复用 trainer 的 step/ckpt/validation
  - 通过 adapter 提供的 primitives 完成 forward/loss/backward（避免 method 管 forward_context）
- 为 finetune 定义 adapter contract（类似 `_DMD2Adapter` 的做法）：
  - `_FineTuneAdapter(Protocol)`：`prepare_batch()` + `sample_train_timestep()` + `student_predict()` + `training_loss()` 等
  - Wan 侧由 `WanAdapter` 实现该 contract（或拆出 `WanAdapterBase + WanFineTuneOps` 以避免 adapter 过度膨胀）

Finetune 的 config（示意）：
```yaml
recipe: {family: wan, method: finetune}
models:
  student: {family: wan, path: ..., trainable: true}
training: {...}
pipeline_config: {...}
method_config:
  pred_type: x0
  loss: flow_matching
```

---

## 9. Guardrails / 测试建议（避免重构“跑得通但不可维护”）

- **scheduler step 对齐测试**：交替更新下，未 step 的 optimizer 对应 scheduler 不应 step
- **batch_size > 1**：消除所有隐式 `B==1` 的 reshape/unflatten 假设
- **role 可选性**：critic 可选时应有清晰报错/降级路径（teacher-only）
- **conditioning 显式性**：训练开始前必须具备 `neg_condition`（来自数据或 provider）
- **checkpoint roundtrip**：save → load → loss 不发散（最小 smoke test）
