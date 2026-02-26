# `fastvideo/distillation/utils/moduleloader.py`

**目的**
- 把 “从 FastVideo 模型路径加载某个子模块（transformer/vae/…）” 的通用逻辑收敛成一个 util，
  便于多个 model plugin 复用，避免每个 plugin 都复制一份 loader 细节。

**当前包含**
- `load_module_from_path(model_path, module_type, training_args, disable_custom_init_weights=False)`
  - 解析/下载 `model_path`
  - 读取 FastVideo 的 per-module config entry
  - 调用 `PipelineComponentLoader.load_module(...)`
  - 可选跳过自定义 init weights（legacy flag：`_loading_teacher_critic_model`）

**边界**
- ✅ 这里只做 “单模块加载”，不做 role 语义、也不做 optimizer/scheduler。
- ✅ “哪些模块需要加载/共享/复用” 仍由 model plugin 决定。

