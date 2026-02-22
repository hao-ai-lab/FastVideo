# Phase 2 Distillation YAML Config

本文档描述当前 **Phase 2 distillation** 入口所使用的 YAML 配置结构、字段含义，以及为什么采用这种设计。

相关实现：
- YAML loader：`fastvideo/distillation/yaml_config.py`
- Phase 2 入口：`fastvideo/training/distillation.py`
- Phase 2 示例 YAML：`fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`

## 1) 入口与约束（非常重要）

Phase 2 distillation **只接受一个真实存在的 YAML 文件路径**，不会 fallback 到 legacy CLI/configs，也不会做 “overlay/补全 outside 路径” 的魔法。

运行方式（示意）：
```bash
python -m fastvideo.training.distillation \
  --config /abs/path/to/fastvideo/distillation/outside/fastvideo/configs/distillation/xxx.yaml
```

CLI 只保留少量 **runtime override**（不属于“实验定义”的内容）：
- `--resume-from-checkpoint`：从 checkpoint 恢复
- `--override-output-dir`：临时覆盖输出目录（方便重复跑实验）
- `--dry-run`：只 parse + build runtime，不启动训练

## 2) YAML 顶层结构

目前 YAML 顶层包含 4 个部分：

```yaml
distill:        # 选择 “模型家族” + “蒸馏方法”
models:         # 以 role 为 key 的模型参与者配置
training:       # 训练参数（直接映射到 TrainingArgs）
pipeline_config:        # pipeline_config 的内联配置（dict）
# 或者 pipeline_config_path: /path/to/pipeline_config.json|yaml
```

loader 规则：
- `pipeline_config` 与 `pipeline_config_path` **二选一**，不能同时提供。
- `training` 会被传入 `TrainingArgs.from_kwargs(**training_kwargs)`；Phase 2 不重复造一套训练参数体系。
- Phase 2 会强制一些 invariants（见第 5 节）。

## 3) `distill`: 选择模型家族与蒸馏方法

```yaml
distill:
  model: wan
  method: dmd2
```

用途：
- 让入口决定用哪个 **runtime builder**（当前仅支持 `wan + dmd2`）。
- 未来扩展时，用同样的 schema 接入更多 `model/method` 组合，而不需要为每个组合写一个新的 training entry file。

设计原因：
- 把 “选择什么（model/method）” 与 “如何训练（training）/谁参与（models）” 分开，结构更稳定。
- 更接近 FastGen：config 选择 `method` + `network`，训练逻辑由 method/adapter 决定。

## 4) `models`: role-based 模型参与者

```yaml
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
```

字段含义（见 `fastvideo/distillation/specs.py`）：
- `family`：模型家族（可省略，默认等于 `distill.model`）
- `path`：模型路径/Hub 名称（由 builder/loader 负责加载）
- `trainable`：该 role 的参数是否参与训练（默认 `true`）

设计原因（为什么要 role-based）：
- distillation setup 千差万别，不应 hard-code “只有 student/teacher/critic 才是核心角色”。**role 只是 key**，method 决定它需要哪些 role。
- role-based bundle 让 Method 可以泛化：例如 CM/KD/SFT 可能需要 reward/refiner/aux_teacher 等，都可以用同一个结构表达。

关于 `trainable` 应由谁决定：
- YAML 里的 `trainable` 表示 **“训练配置的意图/策略”**（policy）。
- Method 仍然可以施加 **算法不变量（invariants）**。例如 `DMD2Method` 会强制要求
  `models.teacher.trainable=false`（否则直接报错），因为 DMD2 默认 teacher 作为固定 reference。

## 5) `training`: 直接映射到 `TrainingArgs`

`training:` 下的 key 基本上就是 `TrainingArgs` 的字段（`fastvideo/fastvideo_args.py`），例如：
- 分布式：`num_gpus`, `sp_size`, `tp_size`（以及内部需要的 hsdp 维度默认值）
- 数据：`data_path`, `dataloader_num_workers`, batch/shape 相关字段
- 输出：`output_dir`, `max_train_steps`, `seed`, `checkpoints_total_limit`
- 优化器/调度器：`learning_rate`, `betas`, `lr_scheduler`, `fake_score_learning_rate`, ...
- distill 相关：`generator_update_interval`, `real_score_guidance_scale`, ...
- tracking/validation：`log_validation`, `validation_*`, `tracker_project_name`, ...

Phase 2 loader 会强制/补全的关键 invariants（见 `fastvideo/distillation/yaml_config.py`）：
- `mode = ExecutionMode.DISTILLATION`
- `inference_mode = False`
- `dit_cpu_offload = False`
- `dit_precision` 默认 `fp32`（保持 training-mode loader 语义：fp32 master weights）
- 若未显式提供，补默认分布式尺寸（例如 `num_gpus`、`sp_size/tp_size`、hsdp 维度）
- 若 `training.model_path` 未提供，则默认取 `models.student.path`
  - 这是为了让 FastVideo 的 pipeline/pipeline_config registry 能正常工作（以 student 为 base）。

设计原因（为什么 training 直接复用 TrainingArgs）：
- FastVideo 的大量组件（loader、pipeline config、distributed init、各种 utils）都以 `TrainingArgs` 为中心。
- Phase 2 的目标是 **解耦 legacy pipeline**，但不等于要立刻重造整个 training 参数系统；复用 TrainingArgs 能显著降低迁移成本与风险。

## 6) `pipeline_config` / `pipeline_config_path`

两种等价写法（二选一）：

1) 直接内联（推荐用于少量关键字段）：
```yaml
pipeline_config:
  flow_shift: 8
  dmd_denoising_steps: [1000, 850, 700, 550, 350, 275, 200, 125]
```

2) 使用外部文件路径（适合复用现有 pipeline config 文件）：
```yaml
pipeline_config_path: /abs/path/to/wan_1.3B_t2v_pipeline.json
```

设计原因：
- 把 “运行一个 distillation 实验需要的最小 pipeline 配置” 放在 distill YAML 附近，便于复现实验。
- 但也允许用 path 复用大型 config 文件，避免在 YAML 中塞进过多模型细节。
- 同时与我们 “outside/ 非侵入式新增 config” 的策略兼容：不必修改上游 `fastvideo/configs/*`。

## 7) 最小可运行示例（Wan few-step DMD2）

完整示例参考：
`fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`

其核心要点是：
- `distill: {model: wan, method: dmd2}`
- `models` 至少包含 `student/teacher/critic`
- `training.data_path / output_dir / max_train_steps / seed` 等训练必须项
- `pipeline_config.flow_shift` + `pipeline_config.dmd_denoising_steps`（8 steps）用于 few-step schedule

## 8) Phase 3 计划：`recipe` 顶层 + `method_config`

Phase 2 的 YAML schema 使用 `distill:` 作为顶层选择（历史原因：当时入口只跑 distillation）。
但随着我们计划把 **finetuning 也纳入同一框架**，`distill.method=finetune` 的语义会显得别扭。

因此 Phase 3 计划升级 schema：

```yaml
recipe: {family: wan, method: dmd2}   # 只负责选择（更通用）
models: {...}                         # role -> {family/path/trainable}
training: {...}                       # infra 参数（映射到 TrainingArgs）
pipeline_config: {...}                # pipeline/backbone config（模型侧）
method_config: {...}                  # method-specific 超参（方法侧）
```

同时保持与 FastVideo 的 `ExecutionMode` 语义对齐（Phase 3 计划）：
- `recipe.method=finetune` 时：入口层设置 `training.mode=FINETUNING`
- 其它 distillation methods：入口层设置 `training.mode=DISTILLATION`

### 8.1 为什么需要 `method_config`？

动机是把语义分清楚：
- `training:`（TrainingArgs）应该尽量只承载 **基础设施**：分布式、优化器、ckpt、logging、数据路径等
- `method_config:` 承载 **算法/recipe** 的超参：DMD2 / Self-Forcing / Finetune 各自不同

这样未来 method 变多时，不会出现所有参数都混在 `training:` 里，导致配置难读、难 review、难复现。

### 8.2 示例：DMD2（新 schema）

```yaml
recipe:
  family: wan
  method: dmd2

models:
  student: {family: wan, path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, trainable: true}
  teacher: {family: wan, path: Wan-AI/Wan2.1-T2V-14B-Diffusers, trainable: false}
  critic:  {family: wan, path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, trainable: true}

training:
  # ... TrainingArgs fields ...
  output_dir: outputs/...
  max_train_steps: 4000
  seed: 1000

pipeline_config:
  flow_shift: 8
  dmd_denoising_steps: [1000, 850, 700, 550, 350, 275, 200, 125]

method_config:
  generator_update_interval: 5
  real_score_guidance_scale: 3.5
  simulate_generator_forward: true
```

### 8.3 示例：Finetuning（only student）

```yaml
recipe:
  family: wan
  method: finetune

models:
  student: {family: wan, path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, trainable: true}

training:
  # ... TrainingArgs fields ...
  data_path: ...
  output_dir: outputs/...
  max_train_steps: 4000
  seed: 1000

pipeline_config:
  flow_shift: 8

method_config:
  pred_type: x0
  loss: flow_matching
```
