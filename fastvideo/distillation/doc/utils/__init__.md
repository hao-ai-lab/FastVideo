# `fastvideo/distillation/utils/`

**目的**
- 放置 distillation 子系统的中性工具代码（不属于某个 model plugin / method）。

当前包含：
- `config.py`：YAML loader + schema/types（`DistillRunConfig` / `DistillRuntime`）。
- `data.py`：通用 dataloader 构建（按 dataset kind/schema 复用 FastVideo 现有实现）。
- `tracking.py`：tracker 初始化（wandb / tensorboard 等）。
- `checkpoint.py`：role-based checkpoint/save-resume（Phase 2 runtime）。
