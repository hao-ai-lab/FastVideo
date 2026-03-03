# `fastvideo/distillation/utils/config.py`

**目的**
- 把 distillation 的 **YAML loader + schema/types + runtime 装配类型** 集中在一个更直觉的位置，
  减少文件级概念数量。

备注：
- `dispatch.build_from_config()` 负责把 YAML 装配成 `(training_args, method, dataloader, start_step)`。
- tracker 由 `DistillTrainer` 构建并持有（避免 model plugin 变成 infra owner）。

这里包含：

## 1) YAML loader（schema v2；YAML-only）

**核心 API**
- `load_distill_run_config(path) -> DistillRunConfig`

**核心产物**
- `DistillRunConfig`
  - `recipe: RecipeSpec`（选择 family + method）
  - `roles: dict[str, RoleSpec]`（来自 YAML 的 `roles:`）
  - `training_args: TrainingArgs`（来自 YAML 的 `training:`，并注入 entrypoint invariants）
  - `validation: dict`（来自 `training.validation:`，由 method 解释并驱动验证）
  - `method_config: dict`（来自 YAML 的 `method_config:`，传给 method 解释）
  - `raw: dict`（原始 YAML，便于 tracker 记录/复现）

**YAML 结构（schema v2）**
- `recipe: {family: ..., method: ...}`
- `roles: {<role>: {family?, path, trainable?}, ...}`
- `training: {...}`（大部分字段复用 `TrainingArgs.from_kwargs()`）
- `training.validation: {...}`（validation 配置；method 也会读取）
- `default_pipeline_config` 或 `default_pipeline_config_path`（默认 pipeline config）
- `method_config: {...}`（算法/recipe 专属超参）

**Validation 参数建议归属**
- 统一放在 `training.validation:`（框架层固定字段 + method 按需字段）。
- trainer 每步调用 `method.log_validation(step)`；是否真正执行由 method 基于
  `training.validation.every_steps` 决定。

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
  - `extra: dict`：保留 `roles.<role>` 下除上述字段外的所有 key/value，
    交给 model plugin 解释（例如 `roles.student.variant: causal`）

## 3) Builder 装配相关（build-time / run-time 边界）
- model plugin（`@register_model`）直接构建并返回一个 `ModelBase` 实例：
  - `training_args`
  - `bundle`
  - `dataloader`
  - `validator`（可选；model-specific）
  - `start_step`（用于 resume / warm-start）
- `dispatch.build_from_config()` 选择 model/method 并返回 `(training_args, method, dataloader, start_step)`。

## 4) 通用解析 helpers（method_config / optimizer 等）
- `get_optional_int(mapping, key, where=...)`
- `get_optional_float(mapping, key, where=...)`
- `parse_betas(raw, where=...)`
