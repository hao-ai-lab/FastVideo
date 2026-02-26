# `fastvideo/distillation/utils/`

**目的**
- 放置 distillation 子系统的中性工具代码（不属于某个 model plugin / method）。

当前包含：
- `config.py`：YAML loader + schema/types（`DistillRunConfig` / `DistillRuntime`）。
- `dataloader.py`：通用 dataloader 构建（按 dataset kind/schema 复用 FastVideo 现有实现）。
- `moduleloader.py`：通用组件加载（`PipelineComponentLoader` 的薄封装）。
- `module_state.py`：module 的 trainable 状态设置（`requires_grad` + train/eval）。
- `tracking.py`：tracker 初始化（wandb / tensorboard 等；由 trainer 持有）。
- `checkpoint.py`：role-based checkpoint/save-resume（Phase 2 runtime）。
