# `fastvideo/distillation/yaml_config.py`

**目的**
- distillation 的 YAML-only 配置加载器（schema v2；不兼容/不 merge legacy CLI）。

**核心产物**
- `DistillRunConfig`
  - `recipe: RecipeSpec`（选择 family + method）
  - `roles: dict[str, RoleSpec]`（来自 YAML 的 `models:`）
  - `training_args: TrainingArgs`（来自 YAML 的 `training:`，并注入 entrypoint invariants）
  - `method_config: dict`（来自 YAML 的 `method_config:`，传给 method 解释）
  - `raw: dict`（原始 YAML，便于 tracker 记录）
    - `wan` family 默认会把 `raw` 作为 W&B config 传给 `wandb.init(config=...)`
    - 入口还会把 YAML 文件本身以 `run.yaml` 的形式上传到 tracker（如 W&B Files）

**YAML 结构（schema v2）**
- `recipe: {family: ..., method: ...}`
- `models: {<role>: {family?, path, trainable?}, ...}`
- `training: {...}`（大部分字段复用 `TrainingArgs.from_kwargs()`）
- `pipeline_config` 或 `pipeline_config_path`
- `method_config: {...}`（算法/recipe 专属超参）

**实现要点**
- `_resolve_existing_file()`：要求传入真实存在的路径（Phase 2 不做 overlay/fallback）
- 默认分布式 size：
  - `num_gpus` 默认 1
  - `tp_size` 默认 1
  - `sp_size` 默认 `num_gpus`（保持与现有 pipeline 的期望一致）
- 训练模式 invariants（由入口强制注入）：
  - `mode = DISTILLATION`
  - `inference_mode = False`
  - `dit_precision` 默认 `fp32`（master weights）
  - `dit_cpu_offload = False`

**兼容性**
- loader 仅接受 schema v2：缺少 `recipe:` 会直接报错。
