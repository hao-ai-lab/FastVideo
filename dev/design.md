# Distill 重构设计草案：`models={teacher, student, critic, ...}` + 算法/模型解耦

这份文档的目标是：在不牺牲当前训练基础设施（分布式、tracker、checkpoint、
VSA/VMoBA 等）的前提下，把 distill 训练部分重构成：

1. **模型输入是显式的 `models` 映射**（teacher/student/critic/...），critic 可选
2. **distill 算法与具体模型解耦**：算法通过 “pipeline adapter/capability” 来
   调用模型，而不是写死 Wan 的张量布局/归一化/输入 key
3. **易扩展**：新增算法或新增模型架构，只需要新增一个 strategy 或 adapter，
   不需要到处改训练 loop

下面所有 “设计” 都会同时写 “原因”，方便取舍。

## 一句话总结（推荐的最终形态）

- `DistillTrainer`（统一训练 loop） + `DistillAlgorithm`（DMD2/SelfForcing/…）
  + `DistillAdapter`（把具体 pipeline 适配成统一接口） + `ModelBundle`
  （按角色组织的模型对象/optimizer/EMA）。

## 现状痛点（为什么要重构）

> 这里只列与架构强相关的点；代码细节见 `dev/distill_structure.md`。

- distill 逻辑 **Wan 耦合严重**（normalize / override cls name / input layout）
- DMD 与 Self-forcing 两套 pipeline 各自维护一份训练 loop，重复且容易 drift
- “需要 teacher/critic/CFG/uncond embedding”等约束不显式，依赖隐含流程
- MoE/dual-transformer（`transformer_2`）的选择、更新在不同 pipeline 里不统一
- 想支持更多模型（Hunyuan/LTX2/LongCat/…）或更多 distill 算法会越来越难

## 设计目标与非目标

### 目标

- 用 `models: dict[str, ...]` 显式描述参与训练的所有模型角色
- 算法（DMD2/Self-forcing/未来更多）只依赖统一接口，不依赖 Wan 特有细节
- 训练 loop 统一（分布式/日志/checkpoint/validation 只实现一次）
- 在 adapter 层支持多种输入来源：
  - 已 preprocess 的 parquet（含 `vae_latent`/`text_embedding`）
  - 或者（可选）原始数据在线 encode（由 pipeline 决定）

### 非目标（第一阶段先不做）

- 不强求“一套 adapter 自动适配所有模型”，允许为不同架构写显式 adapter
- 不强求把所有训练 pipeline（finetune、matrixgame、ltx2 等）一起重构掉
- 不追求立刻改变 checkpoint 格式（可以先兼容旧格式再演进）

## 总体架构（模块分层）

建议把 distill 拆成四层，从下到上：

1. **Adapter 层**（模型相关）
   - 把某个 pipeline/transformer 的 forward 细节封装起来
2. **Algorithm/Strategy 层**（算法相关）
   - DMD2 / Self-forcing / 其它 distill 算法
3. **Trainer/Engine 层**（基础设施）
   - 分布式、seed、grad accumulation、优化器 step、日志、checkpoint、验证
4. **CLI/Entrypoint 层**
   - 解析参数、加载 models、选择算法、启动 trainer

## 关键设计决策（每项含原因）

### 设计 1：用 `ModelBundle` 统一承载多角色模型（核心）

**设计**

- 引入一个只做 “容器” 的数据结构（建议 dataclass）：

  - `models: dict[str, ModelHandle]`
  - 推荐 canonical keys：`student`, `teacher`, `critic`
  - 允许扩展：`reward`, `refiner`, `aux_teacher`, `student_ema` 等

- `ModelHandle` 不是裸 `nn.Module`，而是：

  - `module`: 主要网络（如 transformer）
  - `extra_modules`: 可选模块（如 `transformer_2`、image encoder 等）
  - `optimizers`, `schedulers`, `ema`（可选）
  - `trainable: bool` + `param_groups`（用于决定哪些参数会更新）
  - `capabilities`: 模型端能力声明（见设计 3）

**原因**

- 角色显式化后，算法只需要声明 “我需要哪些 role”，无需硬编码一堆参数名
- `ModelHandle` 把 optimizer/EMA 等训练态绑定到 role，checkpoint/save/resume
  逻辑才能自然泛化
- 对 MoE/dual-transformer 这类 “一个 role 内部有多个可训练模块” 的情况，
  `extra_modules` 能承载而不污染最上层 `models` 命名空间

### 设计 2：把“加载模型”从“训练算法”中剥离成 ModelFactory/Loader

**设计**

- 单独实现 `ModelFactory`：
  - 输入：`ModelSpec`（路径/是否冻结/precision/offload/fsdp 等）
  - 输出：`ModelHandle`
- teacher/critic 只加载算法需要的最小组件（通常只要 transformer），其余组件
  （VAE/scheduler/text encoder）优先复用 student pipeline 的 shared 部分，
  由 adapter 决定是否允许共享

**原因**

- 让算法代码完全不关心 “从哪里 load”、“如何 CPU offload”、“如何 FSDP 包装”
- 有利于做 memory/throughput 优化（teacher/critic 可以走不同策略）
- 避免现状里 `load_modules` 里混杂大量 Wan 特判/override 的情况

### 设计 3：用 `DistillAdapter` 做“pipeline 适配层”，并显式声明 capability

**设计**

- 定义一个 adapter 接口（Protocol/ABC 均可），把下列能力抽象出来：

  1. **latent 规范化/布局**
     - `normalize_latents(x)` / `denormalize_latents(x)`（如需）
     - `to_model_layout(x)` / `from_model_layout(x)`
  2. **conditioning 规范化**
     - 把 dataset/pipeline 产生的 conditioning 变成统一结构
       （例如 `Conditioning` = dict[str, Tensor]）
  3. **噪声与 parameterization**
     - `add_noise(x0, noise, t)`
     - `pred_to_x0(pred, x_t, t)` 或 `pred_to_video_latent(...)`
     - 声明 `prediction_type`（eps/v/x0/flow），由 adapter 负责转换
  4. **模型 forward**
     - `forward(role, x_t, t, cond, *, caches=None, return_dict=False)`
     - 允许 adapter 在内部处理 `set_forward_context` / attention metadata
  5. **CFG 支持（可选）**
     - `supports_cfg: bool`
     - `build_uncond(cond, negative_prompt=...)` 或直接提供
       `uncond_conditioning_cache`

- adapter 要返回一个 `Capabilities` 对象（dataclass）：
  - 是否支持 CFG/uncond
  - 是否支持 KV cache（self-forcing 需要）
  - 是否存在/如何选择 `transformer_2`（MoE）
  - 输入 key 要求（`encoder_attention_mask` 是否必须等）

**原因**

- “算法与模型解耦”只能靠明确的接口边界实现；adapter 是最合适的边界
- capability 显式化后，算法可以做：
  - `if not supports_cfg: raise/降级`
  - `if supports_kv_cache: enable self-forcing` 否则 fallback
- 避免用脆弱的反射去“猜 pipeline 有哪些 stage/属性”，可维护性更高

### 设计 4：用 Strategy 模式承载 distill 算法（DMD2/SF/未来更多）

**设计**

- `DistillAlgorithm` 负责：
  - 声明需要的 roles：`required_roles`, `optional_roles`
  - 声明每一步要更新哪些 role（以及 update ratio/交替策略）
  - 定义 loss：
    - `compute_losses(batch, ctx) -> LossBundle`
    - 或者更细：`losses_for(role)` + `metrics`
  - （可选）维护算法内部状态（例如 self-forcing 的 cache 管理/exit flag 采样）

- 例子：
  - `DMD2Algorithm`：
    - `required_roles = {student, teacher, critic}`
    - loss = 当前 `_dmd_forward` + critic flow-matching
  - `SelfForcingAlgorithm`：
    - 基于 DMD2，但 student forward 换成 causal/self-forcing 的 trajectory
    - 需要 `supports_kv_cache` + `is_causal` 等 capability
  - `TeacherOnlyAlgorithm`（未来可选）：
    - `required_roles = {student, teacher}`
    - 不依赖 critic（满足 “critic optional” 的场景）

**原因**

- 把 “训练 loop” 从 pipeline class 里抽出来后，DMD/SF 不需要两份 train()
- 新增算法不会影响 adapter/trainer，只需要加一个 strategy 类和少量配置
- 更容易做 unit test：给一个 dummy adapter + dummy models 就能测 loss 逻辑

### 设计 5：`DistillTrainer` 统一训练基础设施，并用“更新计划”驱动 optimizer

**设计**

- trainer 只做基础设施：
  - 数据迭代（dataloader / batch builder）
  - grad accumulation + autocast + clip grad
  - 按 strategy 给出的 “更新计划” 去 step 不同 optimizer
  - all-reduce loss/metrics
  - tracker log + checkpoint + validation hook
- trainer 不写任何 DMD/SF 专有逻辑（最多提供 hook 点）
- “更新计划”可以是：
  - `UpdatePlan = list[Update(role, loss_key, optimizer_key, scheduler_key)]`
  - 或者简单：`roles_to_update = {...}` + `losses`

**原因**

- 训练 loop 只实现一次，避免 DMD/SF drift
- update ratio（generator_update_interval / dfake_gen_update_ratio）变成算法参数，
  不再散落在不同 pipeline 里
- 支持更多角色/更多 optimizer 组合时不会爆炸

### 设计 6：把 distill 专有参数从 `TrainingArgs` 里拆成 `DistillConfig`

**设计**

- `TrainingArgs` 保持偏 “训练基础设施”：
  - 数据、输出目录、分布式、optimizer 基本参数、logging/checkpoint
- distill 算法专有参数放到 `DistillConfig`：
  - DMD2：timestep ratio、guidance scale、denoising steps、update interval 等
  - Self-forcing：num_frame_per_block、gradient masking 等
  - Critic：fake_score lr/betas/scheduler 等
- CLI 层把 config 做成 namespace：
  - `--distill.algo dmd2`
  - `--distill.dmd2.generator_update_interval 5`
  - `--distill.sf.num_frame_per_block 3`

**原因**

- 现在 `TrainingArgs` 已经非常大；继续塞 distill 参数会让其它训练模式更难维护
- 分离后能清晰表达：某些参数只对某个算法生效
- 便于做默认值管理（不同 pipeline/算法可提供不同 defaults）

### 设计 7：checkpoint 以 role 为单位泛化（并提供推理导出）

**设计**

- 引入 `CheckpointManager`：
  - `save(role_states, shared_states, step)`
  - `load(...)`
- role_states 来自 `ModelHandle`：
  - trainable role 保存 optimizer/scheduler/ema
  - frozen role（teacher）通常只保存 path 或权重 hash（可选）
- “推理导出”是一个独立通道：
  - 例如只导出 `student`（和可选 `student.transformer_2`）到 diffusers 格式

**原因**

- 现有 `save_distillation_checkpoint` 已经在 role 粒度上开始泛化
  （generator/critic/generator_2/critic_2/real_score_2），继续泛化会更自然
- 未来支持更多角色时（reward/refiner）不需要再复制粘贴一套 save/load

### 设计 8：把 “uncond conditioning” 做成显式的 ConditioningProvider

**设计**

- 对需要 CFG 的算法，提供一个 `ConditioningProvider`：
  - 在训练开始时就构建/缓存 negative prompt embedding（或从 dataset 读取）
  - 不依赖 “是否开启 validation logging”
- provider 与 adapter 配合：
  - adapter 负责怎么 encode negative prompt（模型不同，编码方式不同）
  - provider 负责生命周期与 cache（rank0 广播、避免重复算）

**原因**

- 现状里 uncond embedding 依赖 `_log_validation`，属于隐式耦合，容易踩坑
- provider 显式后，算法与 trainer 的依赖更清晰，validation 也可以变成可选

## CLI 形态建议（与 `models={...}` 对齐）

### 推荐参数形式

- `--models.student <path>`
- `--models.teacher <path>`
- `--models.critic <path>`（可选）

如果需要支持更复杂的配置（比如每个 role 的 precision/offload）：

- `--models_json path/to/models.json`

示例 JSON（建议）：

```json
{
  "student": {"path": "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers", "trainable": true},
  "teacher": {"path": "Wan-AI/Wan2.1-T2V-14B-Diffusers", "trainable": false},
  "critic":  {"path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "trainable": true}
}
```

**原因**

- 直接映射到 `models: dict[str, ModelSpec]`，减少 “某个 role 的 path 到底对应哪个
  CLI 参数” 的歧义
- JSON 形式在复杂场景下更可扩展（多模块、MoE、不同 offload 策略等）

## 迁移计划（推荐分阶段）

### Phase 0：只加新框架，不动现有入口

- 新增 `fastvideo/distill/`（或 `fastvideo/training/distill/`）：
  - `models.py`：`ModelSpec/ModelHandle/ModelBundle`
  - `adapters/`：`WanDistillAdapter`（先只支持 wan）
  - `algorithms/`：`DMD2Algorithm`、`SelfForcingAlgorithm`
  - `trainer.py`：`DistillTrainer`
- 先用单元测试覆盖核心 loss（不用真模型，dummy adapter 即可）

**原因**

- 风险最小：不影响现有脚本/训练
- 先把边界（adapter/strategy/trainer）跑通，后面迁移才不痛

### Phase 1：把现有 Wan DMD / Wan Self-forcing 迁移到新框架（行为对齐）

- 新建一个新的入口脚本（或训练文件）：
  - `fastvideo/training/distill.py`（仅示例）
- 让旧入口（`wan_distillation_pipeline.py` 等）可以选用新 trainer（通过 flag）
- 对齐：
  - loss 数值
  - checkpoint 目录结构
  - validation 输出

**原因**

- 迁移时可 A/B 对比，减少“重构引入质量回归”的概率

### Phase 2：清理旧实现 + 扩展更多模型/算法

- 删除或冻结旧 distill pipeline（保留兼容入口也行）
- 为其它模型实现 adapter（Hunyuan/LTX2/LongCat…）
- 引入更多算法（teacher-only、multi-teacher、RLHF-style…）

**原因**

- 把新增工作限制在 “写 adapter / 写 strategy”，让扩展成本线性增长

## 额外建议（踩坑预防）

- 明确 role 的训练/冻结策略：teacher 永远 `no_grad + eval`，critic/trainable
  role 的 `requires_grad` 与 optimizer param group 必须绑定在一起
- MoE/dual-transformer：建议把 “按 timestep 选择哪个 expert 更新” 的逻辑放到
  adapter 或 strategy 的单一位置，避免像现状一样分散在多处
- lr scheduler 的 step 粒度：建议按 optimizer step，而不是按 global step
  （否则 update ratio 会改变 effective schedule）
