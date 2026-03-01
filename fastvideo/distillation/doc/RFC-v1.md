
## 1) 文件结构（带注释）

```text
fastvideo/distillation/
  trainer.py # 构建training loop 调用method提供的train_one_step接口
  dispatch.py # 根据@register_method和@register_model自动识别类型，根据config构建DistillRuntime
  roles.py # RoleHandle模型外面包一层role的字段，用于区分teacher/student/critic。

  models/
    components.py # ModelComponent dispatch构建模型过程中的中间查无 记录了模型的各个组成部分
    wan.py # 加载Wan模型。不同模型的加载逻辑有所不同（例如ltx就是double stream）。
    ...

  adapters/
    base.py # Adapter本质是一个把已经load好的模型转变为运行时可用api的框架 把Model转变为predict_x0/add_noise/backward/...
    wan.py # 针对Wan的Adapter。
    ...

  methods/
    base.py # DistillMethod基类，需要Method提供必要的例如train one step的接口
    distribution_matching/
      dmd2.py # DMD2Method：DMD2 distillation（student/teacher/critic）
    fine_tuning/
      finetune.py # FineTuneMethod：SFT/finetuning（只有 student）
    knowledge_distillation/
    consistency_model/

  validators/
    base.py # 不同模型推理方式不同，需要根据模型类型写不同的validator。
    wan.py # 调用WanPipeline的validator

  utils/
    config.py # yaml parser
    dataloader.py
    moduleloader.py
    module_state.py # apply_trainable(...)：统一 requires_grad + train/eval
    tracking.py # wandb tracker，由trainer管理
    checkpoint.py# save/resume
```

统一入口（YAML-only）：
- `fastvideo/training/distillation.py`

---

## 2) 关键接口（contracts，注释版）

### 2.1 `roles.py`：RoleManager / RoleHandle

```py
# RoleHandle：一个 role 的“资源包”（method 只通过 RoleHandle 操作 modules/optimizers）
RoleHandle:
  modules: dict[str, nn.Module]           # e.g. {"transformer": ..., "transformer_2": ...}
  optimizers: dict[str, Optimizer]        # method 创建并写回
  lr_schedulers: dict[str, Any]           # method 创建并写回
  trainable: bool                         # 来自 roles.<role>.trainable（只影响 module 状态）

# RoleManager：roles 的容器（role key 不限于 student/teacher/critic）
RoleManager:
  roles: dict[str, RoleHandle]
  require_roles([...])                    # method 用它声明依赖（早失败、错误信息清晰）
```

### 2.2 `dispatch.py`：registry + DistillRuntime

```py
# 目标：新增一个 model plugin 或 method 的成本是 O(1)，而不是写 N×M 组合函数

@register_model("wan")
def build_wan_components(cfg) -> ModelComponents: ...

@register_method("dmd2")
class DMD2Method(DistillMethod): ...

build_runtime_from_config(cfg):
  components = model_builder(cfg)               # -> ModelComponents
  method = method_cls.build(                    # -> DistillMethod instance
    cfg=cfg,
    bundle=components.bundle,
    adapter=components.adapter,
    validator=components.validator,
  )
  return DistillRuntime(training_args, method, dataloader, start_step)
```

### 2.3 `trainer.py`：DistillTrainer（infra only）

```py
# Trainer 只看见 method（算法对象），不看见 roles 的语义，也不看见模型细节。

DistillTrainer(training_args, config=raw_yaml):
  tracker = build_tracker(training_args, config=raw_yaml)

run(method, dataloader, max_steps, start_step, checkpoint_manager?):
  method.set_tracker(tracker)         # 给 method/validator 注入 tracker（artifact logging）
  method.on_train_start()?            # 可选：让 method/adapter 做一次性初始化
  for step in steps:
    loss_map, outputs, metrics = method.single_train_step(...)
    method.backward(loss_map, outputs)?  # 可选覆写（forward_context / ctx-aware backward）
    method.optimizers_schedulers_step(step)?
    method.optimizers_zero_grad(step)?
    method.log_validation(step)?       # 可选：method-managed validation
    checkpoint_manager.maybe_save(step)?
    tracker.log(...)
```

### 2.4 `adapters/`：Adapter 应提供哪些运行时 primitive？

> 说明：`DistillAdapter` 基类只约束最小接口；具体 method 通过自己的 `Protocol`
> 显式声明需要哪些 primitive（duck typing）。这样避免把 DMD2 的需求硬塞进所有 adapter。

当前方法族常用的 primitives（operation-centric）：

```py
# batch/conditioning
prepare_batch(raw_batch, current_vsa_sparsity=..., latents_source={"data"|"zeros"}) -> TrainingBatch
on_train_start()?                      # seed/RNG/negative conditioning/cache（可选）
get_rng_generators()?                  # ckpt 时保存 RNG（可选）

# timestep/noise
num_train_timesteps -> int
shift_and_clamp_timestep(t) -> t
add_noise(clean_latents, noise, t) -> noisy_latents

# forward primitives（不区分 role，只吃 handle）
predict_x0(handle, noisy_latents, t, batch, conditional, attn_kind=...) -> x0
predict_noise(handle, noisy_latents, t, batch, conditional, attn_kind=...) -> noise_like

# backward（为了适配 forward_context / activation ckpt 等约束）
backward(loss, ctx, grad_accum_rounds=...) -> None
```

### 2.5 `methods/base.py`：一个 Method 应有哪些接口？

```py
class DistillMethod(nn.Module):
  # dispatch 用 build(...) 统一装配实例（避免每个 method 写 build_*_method boilerplate）
  @classmethod
  def build(cfg, bundle, adapter, validator) -> DistillMethod

  # 必需：训练一步
  def single_train_step(batch, iteration, current_vsa_sparsity=...) -> (loss_map, outputs, metrics)
  # - loss_map 必须包含 total_loss（或 method 覆写 backward 自己处理）
  # - metrics 用于额外标量日志（trainer 会统一 log）

  # 必需：update policy（算法语义）
  def get_optimizers(iteration) -> Sequence[Optimizer]
  def get_lr_schedulers(iteration) -> Sequence[Any]

  # 可选：更复杂的 backward / 多 ctx / forward_context 约束
  def backward(loss_map, outputs, grad_accum_rounds=...) -> None

  # 可选：validation policy（method-managed）
  def log_validation(step) -> None
```

### 2.6 `validators/`：Validator（family-specific，method-controlled）

```py
ValidationRequest:
  sample_handle: RoleHandle            # method 指定要采样哪个 role（通常 student）
  sampler_kind: {"ode"|"sde"}?         # method 指定采样 loop 类型
  sampling_steps: list[int]?           # 展示用：要跑多少步（可能有多个）
  sampling_timesteps: list[int]?       # few-step schedule（与 sampling_steps 一致时更可控）
  guidance_scale: float?
  output_dir: str?

DistillValidator:
  log_validation(step, request=ValidationRequest?) -> None
```

---

## 3) 当前接受的 YAML config 格式（schema v2）

> 入口：`fastvideo/training/distillation.py --config /abs/path/to/run.yaml`
>
> 特性：
> - **YAML-only**：不与 legacy CLI configs merge
> - `training:` 大部分字段直接映射 `TrainingArgs.from_kwargs(...)`
> - `method_config:` 保持 dict（研究迭代快；由 method 自己解释/校验）

```yaml
# ---- 必需：选择 model plugin + method ----
recipe:
  family: wan        # dispatch key：models/wan.py
  method: dmd2       # dispatch key：methods/**/dmd2.py

# ---- 必需：定义 roles（role key 是任意字符串）----
roles:
  student:
    # family 可省略：默认继承 recipe.family
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    trainable: false
  critic:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

# ---- 必需：训练参数（大部分字段直接走 TrainingArgs）----
training:
  seed: 0
  output_dir: /path/to/out
  data_path: /path/to/parquet
  max_train_steps: 4000
  train_batch_size: 1
  dataloader_num_workers: 4
  num_gpus: 8

  # validation（由 method 调用 validator；validator 具体做采样与记录）
  log_validation: true
  validation_steps: 50
  validation_dataset_file: /path/to/validation.json
  validation_sampling_steps: "8"     # legacy-style string；method 可选择覆写
  validation_guidance_scale: "5.0"   # legacy-style string；method 可选择覆写

  # tracker（由 trainer 构建并持有；不建议在 model plugin 里构建）
  trackers: ["wandb"]
  tracker_project_name: my_project
  wandb_run_name: my_run

# ---- 可选：pipeline config（传入 TrainingArgs.pipeline_config）----
# 只能提供一个：pipeline_config 或 pipeline_config_path
pipeline_config:
  flow_shift: 3
  sampler_kind: ode
# pipeline_config_path: /abs/path/to/pipeline_config.yaml

# ---- 可选：method 私有参数（dict，method 自己解析/校验）----
method_config:
  # DMD2 示例
  rollout_mode: simulate
  dmd_denoising_steps: [999, 750, 500, 250, 0]
  generator_update_interval: 1
  real_score_guidance_scale: 1.0
  attn_kind: dense
```

---

## 4) 为什么这样设计（取舍逻辑）

### 4.1 “很多东西都能写进 config”——为什么还要 model plugin？

可以把“模块名/类名/参数”写进 YAML，但最终仍需要一段代码来：
- 解释这些配置（动态 import / 默认值 / 校验 / 失败时给出清晰错误信息）；
- 做 build-time 的工程装配（模块是否存在、可选模块、shared components 复用、role 组合约束等）；
- 把 FastVideo 现实代码的差异（schema、parallel/offload、module packing 约定）收敛起来。

因此我们把这层解释器命名为 **model plugin（`models/<family>.py`）**：
- config 负责“选择 + 超参”
- model plugin 负责“把选择落地成可运行组件（ModelComponents）”

### 4.2 为什么 adapter 基类很薄（`DistillAdapter` 只有 prepare_batch）？

如果把 `predict_x0/add_noise/backward/...` 全塞进一个巨大的 adapter 基类：
- 你会把某个算法（例如 DMD2）的需求固化成“全体 adapter 必须实现”的硬约束；
- 未来新增方法会被迫实现一堆不需要的接口（耦合上升、可维护性变差）。

当前策略：
- adapter 的稳定边界保持最小
- 每个 method 用 `Protocol` 显式声明自己需要哪些 primitives（代码可读、依赖清晰）

### 4.3 为什么 optimizer/scheduler 由 method 创建？

optimizer cadence / 多优化器更新比例 / critic 的超参等都属于算法语义。  
如果放在 model plugin，会导致：
- model plugin 需要理解 DMD2/finetune/... 的算法细节（污染 build-time 层）
- 同一个 family 随着方法增多出现“把算法 if/else 塞进 models/”的风险

### 4.4 为什么 tracker 由 trainer 构建并持有？

tracker 是 infra 资源（日志/文件/媒体记录），生命周期属于训练 loop：  
- trainer 负责创建 tracker，并统一 `tracker.log(...)`
- method 只产出 metrics；若 method-managed validation 需要 log 视频，则通过 `method.set_tracker(...)`
  把 tracker 注入到 validator（而不是让 model plugin 构建并传递 tracker）

### 4.5 为什么 `method_config` 是 dict？

研究/工程迭代中，方法参数变化频繁；强类型 schema 会带来迁移成本。  
我们把稳定边界结构化（`recipe/roles/training`），把快速变化的部分留给 dict：
- method 自己解析/校验（并给出明确错误提示）
- 解析 helper（int/float/betas）放在 `utils/config.py` 复用，减少重复代码

### 4.6 为什么需要 `dispatch.py` 的 registry？

目标是避免“模型×方法”的组合爆炸：
- 新增一个 family → 加一个 model plugin 并注册
- 新增一个 method → 加一个 method 文件并注册
- 不需要写 25 个 build 函数或 if/else 分支

