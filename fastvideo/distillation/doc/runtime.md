# `fastvideo/distillation/runtime.py`

**目的**
- 定义 distillation builder 的结构化输出类型，明确 build-time 与 run-time 的边界。

**关键类型**
- `FamilyArtifacts`
  - build-time family 插件的产物集合：
    - `training_args`
    - `bundle`
    - `adapter`
    - `dataloader`
    - `tracker`
    - `start_step`（用于 resume / warm-start）
- `DistillRuntime`
  - `DistillTrainer.run()` 所需的最小集合：
    - `training_args`
    - `method`（`DistillMethod`）
    - `dataloader`
    - `tracker`
    - `start_step`

**设计意图**
- family 负责把 “零件” 装配成 `FamilyArtifacts`
- method 负责把算法绑定到（bundle + adapter）上
- trainer 只接收 `method` 并开始训练

