# `fastvideo/distillation/utils/tracking.py`

**目的**
- 把 tracker 初始化从 model plugin 中抽离出来，避免 “模型集成层” 持有 infra 细节。

**当前包含**
- `build_tracker(training_args, config=...)`
  - 读取 `training_args.trackers / training_args.tracker_project_name / output_dir / wandb_run_name`
  - 只在 global rank0 选择真实 tracker；其余 rank 返回 no-op tracker
  - tracker log dir 默认在 `output_dir/tracker/`

**设计意图**
- tracker 属于 infra：entrypoint/trainer 负责持有；method 只负责产出要 log 的 metrics/媒体（video/image/file 等，tracker API 里常叫 artifacts）。
