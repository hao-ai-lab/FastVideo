# Phase 2：让新 Distill 框架 **独立运行**（摆脱 legacy distill pipeline）

Phase 2 的定位：在 Phase 1 已经验证 “Wan DMD2 训练行为对齐 baseline” 的前提下，
把当前仍然依赖 legacy pipeline 的部分逐步替换掉，使 **新 distill 代码路径可以独立运行**
（训练 / validation / checkpoint-resume），同时 **旧代码仍然可跑**（通过保留旧入口文件）。

> 约束：本 phase 采用 **非侵入式** 策略 —— 优先新增代码路径，不强行重构/迁移 legacy 文件。
> 等到完全解耦之后，旧代码的清理由你手动完成（不在 Phase 2 做“删除/搬家”）。

> 额外约束（你拍板）：**不新增任何入口文件**。
> 新 distill 入口直接落在 `fastvideo/training/distillation.py`，并且 **仅接受新的 YAML configs**
>（不兼容旧式 CLI configs）。legacy distill 继续通过现有
> `fastvideo/training/*distillation_pipeline.py` 入口运行（两套路径并存）。

---

## Phase 2 目标（可交付）

1. **Validation 独立化**：不再调用 legacy `pipeline._log_validation(...)`，避免隐式依赖与脆弱属性。
2. **Builder/Runtime 脱离 pipeline**：不再依赖 `WanDistillationPipeline.from_pretrained(...)` 来启动训练；
   改为从 `models={role -> spec}` 直接构建 `ModelBundle + Adapter + Method + DataLoader + Tracker`。
3. **role-based checkpoint/save/resume**：新框架自洽地保存/恢复：
   - per-role modules / optimizers / schedulers
   - RNG states（含用于噪声采样的 generator）
   - StatefulDataLoader（若使用）
4. **YAML 驱动的训练参数解析**：用 `distill.yaml` 描述一次运行；入口只接受新 config（不做 legacy CLI merge）。
5. **`outside/` overlay workaround**：不修改主仓库 `fastvideo/configs/`，在 distillation 内提供可覆盖的“外部配置根”。

---

## Phase 2 非目标（明确不做）

- 清理/删除 legacy distill code（你会在完全解耦后手动清理）
- 将 `fastvideo/training/*` 的通用函数迁移到更中立目录（设计里叫 Phase 2.4，先不做）
- 迁移 Self-forcing v2 / ODE-init（等 distill runtime 自洽后再做）

---

## 当前“仍依赖 legacy”的点（Phase 2 要消灭）

以 Phase 1 的 Wan DMD2 为例，当前仍有两类耦合：

1. **启动/加载耦合**：
   - `fastvideo/training/distillation.py` 仍通过 `WanDistillationPipeline.from_pretrained(...)`
     完成模型加载、optimizer/scheduler、dataloader、tracker。
2. **validation 耦合**：
   - `WanAdapter.log_validation()` 仍复用 legacy `pipeline._log_validation(...)`

Phase 2 的 deliverable 就是把这两类耦合点都替换掉。

---

## Phase 2 TODO List（Review Checklist）

### A. Validation 独立化（Phase 2.1）

- [ ] 定义通用接口：`DistillValidator`
  - 输入：`bundle + adapter + training_args + tracker`
  - 输出：`log_validation(step)`（只在 rank0 真正写 artifact）
- [ ] 实现 `WanValidator`（Wan + T2V 的最小版本）
  - 复用 `fastvideo/dataset/validation_dataset.py::ValidationDataset`
  - 复用模块化 **inference pipeline**（建议：`fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py::WanDMDPipeline`）
  - 支持关键参数：
    - `validation_sampling_steps`（few-step）
    - `validation_guidance_scale`
    - seed / RNG（不依赖 legacy pipeline 初始化副作用）
- [ ] 将 Phase 1 的 `WanAdapter.log_validation()` 切到新 validator（不再依赖 legacy pipeline）

### B. Builder/Runtime 脱离 pipeline（Phase 2.2）

- [ ] 定义结构化 spec（角色驱动）：`ModelSpec / RoleSpec / DistillSpec`
  - 目标：`models={role -> spec}` 成为唯一真相
  - method 自己声明需要哪些 roles（缺失则报错）
- [ ] 新增 YAML 配置解析（Phase 2 必做）
  - [ ] `fastvideo/training/distillation.py` 支持 `--config path/to/distill.yaml`
  - [ ] 解析策略：`yaml.safe_load` + schema 校验（不做 legacy CLI merge）
  - [ ] YAML schema 最小包含：`distill.{model,method}` + `models{role->spec}` + `training{...}` + `pipeline_config{...}`
- [ ] 支持 `outside/` overlay（Phase 2 workaround）
  - [ ] 新增目录：`fastvideo/distillation/outside/`（视作外部 repo root）
  - [ ] 约定覆盖路径：`fastvideo/distillation/outside/fastvideo/configs/...`
  - [ ] config loader 在解析 `pipeline_config_path`/yaml include 时：outside 优先、repo fallback
  - [ ] 不通过 `sys.path` shadow `fastvideo` 包；outside 仅做文件 overlay（必要时用 `importlib` 按路径加载 .py config）
- [ ] 实现 “standalone runtime builder”
  - 直接加载 modules（student/teacher/critic）并构建 `ModelBundle`
  - 构建 per-role optimizers/schedulers（复用现有 TrainingArgs 超参）
  - 构建 dataloader（直接调用 `fastvideo/dataset/parquet_dataset_map_style.py::build_parquet_map_style_dataloader`）
  - 初始化 tracker（复用 `fastvideo/training/trackers/`）
- [ ] 修改现有入口 `fastvideo/training/distillation.py`（不新增入口文件）
  - `--config` 为必需参数：只跑 standalone builder 路径（YAML-only）
  - legacy distill 通过旧入口文件继续可跑（不在该入口做 fallback）

### C. role-based checkpoint/save/resume（Phase 2.3）

- [ ] 新增 `DistillCheckpointManager`
  - 保存内容：
    - `bundle.roles[*].modules`（仅 trainable params 或全量可配置）
    - `bundle.roles[*].optimizers / lr_schedulers`
    - `StatefulDataLoader`（如果使用）
    - `RandomStateWrapper`（torch/numpy/python/cuda + noise generators）
  - 恢复内容：
    - `start_step`、dataloader iterator state、optimizer/scheduler state、RNG
- [ ] 将 checkpoint manager 接入 `DistillTrainer`
  - `--resume_from_checkpoint`
  - `--checkpointing_steps`（或复用现有 args）
  - `--checkpoints_total_limit`

### D. 示例脚本（Phase 2）

- [ ] `examples/distillation/phase2/`：
  - 最小 smoke：Wan 1.3B 学 14B（DMD2）+ few-step validation + save/resume
  - 提供 `distill.yaml` 模板（role-based）

### E. 最小单测（可选但建议）

- [ ] `models_json` schema 解析 + role 校验
- [ ] checkpoint roundtrip（mock modules + optimizer）不 crash

---

## 关键设计（具体到代码）

### 2.1 Validation 独立化：`DistillValidator` / `WanValidator`

**核心原则**：validation 只是一段 “inference + decode + log”，它应当：

- 不依赖训练 pipeline 的内部属性（例如 `validation_random_generator`）
- 不要求 `pipeline.train()` 被调用才“初始化齐全”

建议落地：

- 新增目录：`fastvideo/distillation/validators/`
  - `base.py`：`DistillValidator` 抽象
  - `wan.py`：`WanValidator`

`WanValidator` 的实现思路（最小版本）：

1. 从 `training_args.validation_dataset_file` 构建 `ValidationDataset`
2. 取若干条 prompt（rank0）
3. 构建/复用一个 inference pipeline：
   - `WanDMDPipeline.from_pretrained(student_model_path, loaded_modules=...)`
   - 或者直接用 loader 加载 `text_encoder/tokenizer/vae/scheduler`，并注入 student transformer
4. 生成视频并交给 tracker（wandb）记录

**风险点（如遇到需要你决策会停下讨论）**

- validation 要不要严格对齐 legacy `_log_validation` 的所有输出格式（latent vis dict 等）？
  - 建议 Phase 2 先只对齐“视频 artifact + caption”，其余可后续补齐。

### 2.2 Builder/Runtime 独立化：roles/spec -> instantiate

**目标**：从 “先构建 legacy pipeline，再拆 roles” 转成 “roles/spec 直接构建 bundle”。

建议新增：

- `fastvideo/distillation/specs.py`
  - `ModelSpec`：`family/path/revision/precision/...`
  - `RoleSpec`：`role/trainable/optimizer/scheduler/...`
  - `DistillSpec`：`method + models{role->RoleSpec} + adapter_family(optional)`

一个最小的 `distill.yaml` 示例（Wan 1.3B 学 14B + critic=1.3B；示意）：

```yaml
distill:
  model: wan
  method: dmd2

models:
  student: {family: wan, path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, trainable: true}
  teacher: {family: wan, path: Wan-AI/Wan2.1-T2V-14B-Diffusers, trainable: false}
  critic:  {family: wan, path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, trainable: true}

training:
  output_dir: outputs/phase2_wan_dmd2
  max_train_steps: 4000
  seed: 1000
```

**加载实现建议（Wan）**

尽量复用已经存在的、与 distill training pipeline 无关的 loader 工具链：

- `fastvideo/utils.py::maybe_download_model / verify_model_config_and_directory`
- `fastvideo/models/loader/component_loader.py::PipelineComponentLoader`

这样我们无需实例化 `WanDistillationPipeline`，也能加载：

- `transformer`（student/teacher/critic）
- （可选）`transformer_2`（MoE 支持，Phase 2 先保持 optional）
- `vae/text_encoder/tokenizer/scheduler`（用于 adapter + validation）

**构建 optimizer/scheduler**

- 在 builder 内按 role 创建 optimizer/scheduler，并放进 `RoleHandle.optimizers/lr_schedulers`
- method 继续用 `get_optimizers/get_lr_schedulers` 做 update policy（Trainer 不关心 role）

**入口文件**

- 不新增入口文件；直接增强 `fastvideo/training/distillation.py`：
  - 仅支持 `--config distill.yaml`（YAML-only），不再兼容旧式 CLI configs
  - legacy pipeline 继续通过现有 `fastvideo/training/*distillation_pipeline.py` 入口运行

### 2.3 role-based checkpoint：`DistillCheckpointManager`

建议新增：

- `fastvideo/distillation/checkpoint.py`
  - `DistillCheckpointManager`
  - 内部复用 `fastvideo/training/checkpointing_utils.py` 的 wrappers：
    - `ModelWrapper/OptimizerWrapper/SchedulerWrapper/RandomStateWrapper`

建议 checkpoint 的 “state dict key” 命名空间：

- `models/{role}/{module_name}`
- `optimizers/{role}/{name}`
- `schedulers/{role}/{name}`
- `random/{name}`（全局 RNG + per-role noise generator）
