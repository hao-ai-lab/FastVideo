# `fastvideo/distillation/models/__init__.py`

**目的**
- models 是 build-time 插件层（model plugins）：
  - 从 YAML config 读取 role spec
  - 加载模型模块（transformer/vae/...）
  - 构建 `RoleManager`
  - 构建 adapter / dataloader / tracker / validator

**为什么需要 model plugins？**
- 把 “装配/加载/数据/分布式细节” 与 “算法/rollout/loss/update policy” 分离：
  - model plugin 专注 build-time 高内聚
  - method 专注算法高内聚
  - entrypoint/dispatch 不需要 N×M 组合逻辑
