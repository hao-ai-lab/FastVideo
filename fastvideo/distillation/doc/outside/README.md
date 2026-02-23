# `fastvideo/distillation/outside/README.md`

**目的**
- 解释 `fastvideo/distillation/outside/` 的存在理由：
  - Phase 2/2.9 期间，我们需要引入新的 YAML config，但又不想直接改动主 repo 的
    `fastvideo/configs/` 树（避免冲突/侵入式修改）。

**约定**
- Phase 2 entrypoint（`fastvideo/training/distillation.py`）只接受**真实路径**：
  - `--config fastvideo/distillation/outside/fastvideo/configs/distillation/<run>.yaml`
- `outside/` 只放 data/config（不要放可 import 的 Python 代码）。

**推荐目录结构**
- `fastvideo/distillation/outside/fastvideo/configs/distillation/*.yaml`

