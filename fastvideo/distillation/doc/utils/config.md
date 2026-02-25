# `fastvideo/distillation/utils/config.py`

**目的**
- 把 distillation 里常用的“结构化类型”集中在一个更直觉的位置，减少文件级概念数量。

这里包含两类 dataclass：

## 1) Schema / run config 相关（轻量选择项）
- `RecipeSpec`
  - `family`: family 名称（例如 `"wan"`）
  - `method`: method 名称（例如 `"dmd2"`）
- `RoleSpec`
  - `family`: 该 role 的 family（默认可继承 `recipe.family`）
  - `path`: 模型权重路径（HF repo 或本地目录）
  - `trainable`: 是否训练该 role（只影响 `requires_grad`/模式；具体 optimizer 由 method 决定）
  - `disable_custom_init_weights`: 是否禁用 family 的“加载时自定义 init weights 逻辑”

## 2) Builder 装配相关（build-time / run-time 边界）
- `FamilyArtifacts`
  - family 插件 build-time 的产物集合：
    - `training_args`
    - `bundle`
    - `adapter`
    - `dataloader`
    - `tracker`
    - `validator`（可选；family-specific）
    - `start_step`（用于 resume / warm-start）
- `DistillRuntime`
  - `DistillTrainer.run()` 所需的最小集合：
    - `training_args`
    - `method`（`DistillMethod`）
    - `dataloader`
    - `tracker`
    - `start_step`

