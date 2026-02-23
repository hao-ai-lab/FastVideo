# `fastvideo/distillation/validators/wan.py`

**定位**
- `WanValidator`：Wan distillation 的 validation/sampling 实现（Phase 2 standalone）。
- 负责：
  - 读取 validation dataset（json）
  - 调用 Wan pipeline 生成视频样本
  - 保存 mp4 到 `output_dir`
  - 通过 tracker（例如 wandb）记录 artifacts

**关键点**
- validator 运行在分布式环境下：
  - 以 SP group 为单位做采样，最终由 global rank0 聚合写文件与 log
- 通过 `loaded_modules={"transformer": student_transformer}` 复用训练中的 student 模块权重。

**依赖**
- 当前使用 `WanDMDPipeline` 做采样推理（FlowMatch scheduler + DmdDenoisingStage）。
  这属于 validation 选择（并不影响 adapter 的“无算法命名耦合”约束）。

**可演进方向（Phase 3+）**
- 将 validation steps/guidance 等采样配置从 `TrainingArgs` 迁移到更明确的配置块（例如 `validation:`）。
- 进一步抽象 validator API，使其更容易被不同 family/method 复用。

