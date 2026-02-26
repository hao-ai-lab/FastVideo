# `fastvideo/distillation/utils/config.py`

**目的**
- 把 distillation 的 **YAML loader + schema/types + runtime 装配类型** 集中在一个更直觉的位置，
  减少文件级概念数量。

备注：
- model plugin 的 build-time 产物结构体 `ModelComponents` 在
  `fastvideo/distillation/models/components.py`（更贴近语义归属）。

这里包含：

## 1) YAML loader（schema v2；YAML-only）

**核心 API**
- `load_distill_run_config(path) -> DistillRunConfig`

**核心产物**
- `DistillRunConfig`
  - `recipe: RecipeSpec`（选择 family + method）
  - `roles: dict[str, RoleSpec]`（来自 YAML 的 `roles:`）
  - `training_args: TrainingArgs`（来自 YAML 的 `training:`，并注入 entrypoint invariants）
  - `method_config: dict`（来自 YAML 的 `method_config:`，传给 method 解释）
  - `raw: dict`（原始 YAML，便于 tracker 记录/复现）

**YAML 结构（schema v2）**
- `recipe: {family: ..., method: ...}`
- `roles: {<role>: {family?, path, trainable?}, ...}`
- `training: {...}`（大部分字段复用 `TrainingArgs.from_kwargs()`）
- `pipeline_config` 或 `pipeline_config_path`
- `method_config: {...}`（算法/recipe 专属超参）

**实现要点**
- `_resolve_existing_file()`：要求传入真实存在的路径（不做 overlay/fallback）
- 默认分布式 size：
  - `num_gpus` 默认 1
  - `tp_size` 默认 1
  - `sp_size` 默认 `num_gpus`（保持与现有 pipeline 的期望一致）
- 训练模式 invariants（由入口强制注入）：
  - `mode = DISTILLATION`
  - `inference_mode = False`
  - `dit_precision` 默认 `fp32`（master weights）
  - `dit_cpu_offload = False`

## 2) Schema / run config 相关（轻量选择项）
- `RecipeSpec`
  - `family`: family 名称（例如 `"wan"`）
  - `method`: method 名称（例如 `"dmd2"`）
- `RoleSpec`
  - `family`: 该 role 的 family（默认可继承 `recipe.family`）
  - `path`: 模型权重路径（HF repo 或本地目录）
  - `trainable`: 是否训练该 role（只影响 `requires_grad`/模式；具体 optimizer 由 method 决定）
  - `disable_custom_init_weights`: 是否禁用 family 的“加载时自定义 init weights 逻辑”

## 3) Builder 装配相关（build-time / run-time 边界）
- `ModelComponents`
  - model 插件 build-time 的产物集合：
    - `training_args`
    - `bundle`
    - `adapter`
    - `dataloader`
    - `validator`（可选；model-specific）
    - `start_step`（用于 resume / warm-start）
备注：
- `DistillRuntime` 由 `dispatch.build_runtime_from_config()` 创建并定义在
  `fastvideo/distillation/dispatch.py`（谁创建谁声明）。
- tracker 由 `DistillTrainer` 构建并持有（避免 model plugin 变成 infra owner）。

## 4) 通用解析 helpers（method_config / optimizer 等）
- `get_optional_int(mapping, key, where=...)`
- `get_optional_float(mapping, key, where=...)`
- `parse_betas(raw, where=...)`
