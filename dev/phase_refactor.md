# Phase：Refactor（抛弃 “Distill” 命名：这是通用 Training 框架；YAML=instantiate-first）

## 0) 最终 YAML 示例（Self-Forcing：Student=Causal Wangame，Teacher=Bidirectional Wangame）

> 你提的方向我同意：YAML 里不要再出现 `wangame/dmd2` 这种“缩写字符串”。
> 直接写 **类名/类路径**，用 FastGen 同款 `instantiate` 思路（Hydra 风格 `_target_`），避免 registry/if-else。

```yaml
log:
  project: fastvideo
  group: wangame
  name: self_forcing_causal_student_bidi_teacher
  wandb_mode: online

trainer:
  # trainer 本身也可以不 instantiate（只当作纯参数 dict）
  # 但为了彻底对齐 FastGen，我们允许它也走 _target_。
  _target_: fastvideo.distillation.trainer.Trainer
  output_dir: outputs/wangame_self_forcing_refactor
  max_train_steps: 100000
  seed: 1000
  mixed_precision: bf16
  grad_accum_rounds: 1

data:
  _target_: fastvideo.distillation.utils.dataloader.ParquetDataConfig
  data_path: /path/to/wangame/parquet
  dataloader_num_workers: 4

validation:
  enabled: true
  dataset_file: examples/training/finetune/WanGame2.1_1.3b_i2v/validation_random.json
  every_steps: 100
  sampling_steps: [4]
  rollout_mode: streaming
  # 不固定字段：由 method/validator 自己按需读取（从简）
  pipeline:
    sampler_kind: sde
    scheduler:
      _target_: fastvideo.models.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler
      flow_shift: 3

shared_context:
  # 替代旧的 recipe.family：shared_context 是一个“可实例化对象”
  _target_: fastvideo.distillation.models.wangame.shared_context.WanGameSharedContext
  # shared parts（VAE/encoders/schedulers/validator 等）只构建一份，来源由该路径决定
  model_path: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000

models:
  # 每个 role 一个独立 Model 实例（method 决定需要哪些 role）
  student:
    _target_: fastvideo.distillation.models.wangame.wangame_causal.WanGameCausalModel
    init_from: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true

  teacher:
    _target_: fastvideo.distillation.models.wangame.wangame.WanGameModel
    init_from: weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers
    trainable: false

  critic:
    _target_: fastvideo.distillation.models.wangame.wangame.WanGameModel
    init_from: outputs/wangame_dfsft_causal_4n8g/persistent/checkpoint-22000
    trainable: true

method:
  _target_: fastvideo.distillation.methods.distribution_matching.self_forcing.SelfForcingMethod
  # method-specific config（随 method 变化，不强行统一 schema）
  rollout_mode: simulate
  chunk_size: 3
  student_sample_type: sde
  context_noise: 0.0
```

---

## 1) 为什么要从 “Distill” 改成 “Training”

现状里我们已经同时支持：

- few-step distillation（DMD2 / Self-Forcing）
- finetuning / SFT
- DFSFT（causal + streaming rollout）

所以 “distill” 只是其中一种训练策略；框架本质更像：

> **Trainer（loop） + Method（algorithm） + Model（per-role instance） + SharedContext（shared parts）**

本次 refactor 的目标是把命名与 config 语义对齐，让人第一眼就能理解“这是训练框架”，而不是“蒸馏脚手架”。

---

## 2) 参考 FastGen：我们要“照搬”的点是什么

FastGen 的几个关键点（对应到我们要做的事）：

1) **instantiate-first**
   - FastGen 用 `LazyCall/_target_ + instantiate()`：config 里写“我要哪个类、给它什么参数”，代码只负责实例化与编排。
   - 我们的 YAML 也应如此：不再让 `dispatch.py` 维护字符串 registry（`wangame`/`dmd2`/...）。

2) **继承 + `super().build_*()` 复用装配逻辑**
   - 通用的：建 shared parts、建 networks、DDP/FSDP wrapper、optim/scheduler、checkpoint、logging。
   - method 子类只 override：需要哪些 models、rollout/损失、更新策略。

3) **Student 持有 shared parts（或显式 shared_context）**
   - FastGen 把 preprocessors 隐式挂到 `self.net`（student）；
   - 我们这边 shared parts 语义更重（pipeline/validator/dataloader），更适合显式 `shared_context`（但 config 同样 instantiate）。

---

## 3) 新 YAML 的核心语义（草案）

### 3.1 `_target_`：一切都可 instantiate

- `method._target_`：算法/训练策略（SelfForcing/DMD2/Finetune/DFSFT）
- `models.<role>._target_`：每个 role 的 Model class（bidi/causal）
- `shared_context._target_`：shared parts 的构建者（按任务/数据形态）
- （可选）`trainer._target_`：训练 loop（我们也可以先只传纯参数，避免 trainer 过度可插拔）

### 3.2 去掉 `family/method` 缩写

不再支持：

- `recipe.family: wangame`
- `models.student.family: wangame_causal`
- `recipe.method: self_forcing`

只保留 `_target_`（或 `class_path` 同义字段），避免隐式映射导致概念膨胀。

### 3.3 “shared_component_model” 的位置

我建议 **直接去掉 `shared_component_model`**：

- shared parts 的来源已经由 `shared_context.model_path` 显式指定；
- 再用一个 role 去“引用/指代”来源，属于重复表达，反而会让人误以为 role 会影响 shared parts 的语义。

从简后的约束：

- 只要求 `shared_context.model_path` 必填；
- 是否与 `models.student.init_from` 一致由用户自行保证（如果未来要 strict 校验，再引入显式的 `shared_context.strict_match` 之类字段）。

---

## 4) 组装流程（调用链）

伪代码（对齐 FastGen：method 主导装配）：

```py
cfg = load_yaml(path)

method = instantiate(cfg["method"], cfg=cfg)  # method class

shared_context = instantiate(cfg["shared_context"], cfg=cfg)  # build shared parts once
models = {
  role: instantiate(model_cfg, role=role, shared_context=shared_context)
  for role, model_cfg in cfg["models"].items()
}

run_objects = method.build_run(
  cfg=cfg,
  shared_context=shared_context,
  models=models,
)

trainer = instantiate(cfg["trainer"], cfg=cfg, **run_objects.trainer_inputs)
trainer.run(**run_objects.run_inputs)
```

Self-Forcing 额外 fail-fast：

- `student` 必须是 `CausalModelBase`（不允许 cache-free 分支）

---

## 5) TODO（本阶段要改哪些地方）

> 先以 wangame 跑通；再推广到 wan。

- [ ] 新增 `fastvideo/distillation/utils/instantiate.py`
  - 支持 `_target_`（Hydra 语义）+ 纯 dict 参数
  - 支持把 `cfg` 作为通用参数注入（FastGen 常用 pattern）

- [ ] `fastvideo/distillation/utils/config.py`
  - 更新 yaml schema：以 `_target_` 为主（不再解析 `recipe.family/method`）

- [ ] `fastvideo/distillation/dispatch.py`
  - 从 “registry 分发 build_*” → “instantiate + method.build_run()”

- [ ] 命名清理（不要求一次性全改完，但新接口上应去掉 Distill）
  - `DistillTrainer` → `Trainer`（内部文件夹可暂时保留）
  - `DistillMethod` → `Method`
  - entrypoint：`distillation.py` → `training.py`（可选；看 PR 策略）
