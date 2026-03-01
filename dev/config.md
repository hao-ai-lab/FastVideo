# Distillation YAML config (schema v2)

本文档描述当前 distillation 入口所使用的 **YAML schema v2**、字段含义与设计取舍。

相关实现：
- YAML loader：`fastvideo/distillation/utils/config.py::load_distill_run_config`
- Entrypoint：`fastvideo/training/distillation.py`
- Schema/类型定义：`fastvideo/distillation/utils/config.py`
- 示例 YAML（examples）：`examples/distillation/`

## 1) 入口与约束（非常重要）

distillation 入口 **只接受一个真实存在的 YAML 文件路径**（不 merge legacy CLI/configs，
也不做路径补全/overlay）。YAML 是 single source of truth。

运行方式（示意）：
```bash
python fastvideo/training/distillation.py \
  --config /abs/path/to/examples/distillation/<phase>/xxx.yaml
```

CLI 仅保留少量 **runtime override**（不属于“实验定义”的内容）：
- `--resume-from-checkpoint`：从 checkpoint 恢复
- `--override-output-dir`：临时覆盖输出目录（方便重复跑实验）
- `--dry-run`：只 parse + build runtime，不启动训练

## 2) YAML 顶层结构（schema v2）

```yaml
recipe:          # 选择 family + method（只负责“选什么”）
roles:           # role -> role spec（谁参与）
training:        # infra 参数（直接映射到 TrainingArgs）
pipeline_config: # 模型/pipeline 侧 config（可 inline）
method_config:   # method/algorithm 超参（方法侧 single source of truth）
# 或者 pipeline_config_path: /abs/path/to/pipeline_config.json|yaml
```

loader 规则：
- `pipeline_config` 与 `pipeline_config_path` **二选一**，不能同时提供。
- `training` 会被传入 `TrainingArgs.from_kwargs(**training_kwargs)`；我们不重造一套训练参数体系。
- 缺少 `recipe:` 会直接报错（schema v1 的 `distill:` 不再支持）。

## 3) `recipe`: 选择 family 与 method

```yaml
recipe:
  family: wan
  method: dmd2
```

用途：
- registry dispatch：选择 `models/<family>.py` + `methods/<method>.py` 的组合（N+M，而非 N×M）。
- 语义更通用：未来把 finetuning 也纳入时不会出现 `distill.method=finetune` 的别扭表达。

## 4) `roles`: role-based 参与者

```yaml
roles:
  student:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    trainable: false
    disable_custom_init_weights: true
  critic:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
    disable_custom_init_weights: true
```

字段含义（见 `fastvideo/distillation/utils/config.py`）：
- `family`：可选；默认继承 `recipe.family`
- `path`：模型路径 / hub 名称（由 family 负责加载）
- `trainable`：该 role 的参数是否参与训练（影响 `requires_grad`/train/eval）
- `disable_custom_init_weights`：可选；禁用 family 的 “加载时自定义 init weights 逻辑”

设计原因：
- role 只是 key；framework 不强行规定 “canonical roles”。method 决定它需要哪些 roles。
- `trainable` 表示训练意图；method 仍可施加算法不变量（例如 DMD2 强制 teacher frozen）。

## 5) `training`: 直接映射到 `TrainingArgs`

`training:` 下的 key 基本上就是 `TrainingArgs` 字段（`fastvideo/fastvideo_args.py`），例如：
- 分布式：`num_gpus`, `sp_size`, `tp_size`
- 数据：`data_path`, `dataloader_num_workers`, shape/batch 相关字段
- 输出：`output_dir`, `max_train_steps`, `seed`, `checkpoints_total_limit`
- 优化器默认值：`learning_rate`, `betas`, `lr_scheduler`, ...
- tracking/validation：`log_validation`, `validation_*`, `tracker_project_name`, ...

loader 会注入/补全的 invariants（见 `fastvideo/distillation/utils/config.py`）：
- `mode = ExecutionMode.DISTILLATION`
- `inference_mode = False`
- `dit_precision` 默认 `fp32`（master weights）
- `dit_cpu_offload = False`
- 分布式尺寸默认值（`num_gpus/tp_size/sp_size/hsdp_*`）
- `training.model_path` 若缺失，默认使用 `roles.student.path`（供 pipeline_config registry 使用）

## 6) `pipeline_config` / `pipeline_config_path`

两种写法（二选一）：

1) inline（适合少量 override）：
```yaml
pipeline_config:
  flow_shift: 8
```

2) path（适合复用大型 config 文件）：
```yaml
pipeline_config_path: /abs/path/to/wan_1.3B_t2v_pipeline.json
```

常见字段（非穷举）：
- `flow_shift`：Wan 的 flow-matching shift（影响 noise schedule）。
- `sampler_kind`：`ode|sde`，选择 sampling loop 语义（`WanPipeline` 内部切换）。

备注（重要）：
- 从语义上讲，`dmd_denoising_steps` 是 algorithm knob，应当只存在于 `method_config`。
- Phase 3.2 已将 sampling loop 语义显式化：
  - method 通过 `ValidationRequest(sampler_kind=..., sampling_timesteps=...)` 指定采样方式与 few-step schedule
  - `WanValidator` 将 timesteps 写入 `ForwardBatch.sampling_timesteps`，并使用 `WanPipeline` 执行采样
  - `pipeline_config.dmd_denoising_steps` 不再是 distillation 的必需字段（仅保留为 inference/legacy 兼容）

## 7) `method_config`: method/algorithm 专属超参

`method_config` 由 method 自己解释。以 DMD2 为例：
```yaml
method_config:
  rollout_mode: simulate            # {simulate|data_latent}
  generator_update_interval: 5
  real_score_guidance_scale: 3.5
  dmd_denoising_steps: [1000, 850, 700, 550, 350, 275, 200, 125]
```

其中：
- `rollout_mode` 替代 legacy 的 `training.simulate_generator_forward`：
  - `simulate`：adapter 用零 latents 构造 batch（不依赖 `vae_latent`）
  - `data_latent`：dataset batch 必须提供 `vae_latent`
- `dmd_denoising_steps` 是 method 的 few-step schedule single source of truth。

## 8) 最小可运行示例（Wan few-step DMD2）

参考 `examples/distillation/` 下的可运行 YAML：
- `examples/distillation/phase2/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml`
- `examples/distillation/phase2_9/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase2.9.yaml`
- `examples/distillation/phase3_1/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.1.yaml`
- `examples/distillation/phase3_2/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.2.yaml`
