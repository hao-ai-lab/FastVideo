# `fastvideo/distillation/utils/data.py`

**目的**
- 把 “dataloader 构建” 从 model plugin（原 families/，现 `models/`）中抽离出来，
  让插件更聚焦在加载模块与组装 adapter/bundle。

**当前包含**
- `build_parquet_t2v_train_dataloader(training_args, parquet_schema=...)`
  - 复用 FastVideo 现有的 `build_parquet_map_style_dataloader(...)`
  - 仅做最小封装：从 `training_args` 读取必要参数（data_path/batch/workers/seed/cfg_rate/text_len 等）

**边界**
- 这里不包含 model/pipeline 语义（例如 Wan 的 forward/backward 细节）。
- 若未来要支持更多 dataset kind（webdataset / precomputed / i2v / ode-init ...），
  推荐在本目录新增更通用的 builder（或引入 `DataSpec` 再做统一 dispatch）。
