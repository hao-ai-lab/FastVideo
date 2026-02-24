# `fastvideo/distillation/specs.py`

**目的**
- 把 config 里的“选择项”做成轻量 dataclass，便于：
  - YAML 解析（`yaml_config.py`）
  - builder/registry dispatch（`builder.py` / `registry.py`）

**关键类型**
- `RecipeSpec`
  - `family`: family 名称（例如 `"wan"`）
  - `method`: method 名称（例如 `"dmd2"`）
- `RoleSpec`
  - `family`: 该 role 的 family（默认可继承 `recipe.family`）
  - `path`: 模型权重路径（HF repo 或本地目录）
  - `trainable`: 是否训练该 role（只影响 `requires_grad`/模式；具体 optimizer 由 method 决定）
  - `disable_custom_init_weights`: 是否禁用 family 的“加载时自定义 init weights 逻辑”
    - 这是一个 build-time/loader 语义开关（用于 teacher/critic 等 auxiliary roles）
    - 角色名不应在 family 内被 hard-code；由 YAML 显式声明更清晰

**注意**
- role 名称本身（`student/teacher/critic/...`）是字符串。
  framework 不强行规定“canonical roles”，由 method 决定语义与依赖。
