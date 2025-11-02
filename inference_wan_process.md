### 运行 scripts/inference/v1_inference_wan.sh 后端到端流程（Wan 2.1 T2V 1.3B）

以下描述当执行脚本 `scripts/inference/v1_inference_wan.sh` 时，FastVideo 在代码层面按什么顺序、通过哪些文件/函数完成“参数解析 → 模型与管线初始化 → 推理 → 保存输出”。

- 脚本关键参数（摘自 `scripts/inference/v1_inference_wan.sh`）
  - `--model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
  - `--num-gpus 1 --tp-size 1 --sp-size 1`
  - CPU/GPU offload 开关：`--dit-cpu-offload False --vae-cpu-offload False --text-encoder-cpu-offload True --pin-cpu-memory False`
  - 采样参数：`--height 480 --width 832 --num-frames 77 --num-inference-steps 50 --fps 16 --guidance-scale 6.0 --flow-shift 8.0 --seed 1024`
  - 文本输入：`--prompt-txt assets/prompt.txt`（与 `--prompt` 互斥）
  - 输出：`--output-path outputs_video/`

---

### 1) CLI 入口与子命令注册

- 控制台入口：`pyproject.toml` 指定 `fastvideo = fastvideo.entrypoints.cli.main:main`
- 主函数：`fastvideo/entrypoints/cli/main.py` 的 `main()`
  - 构建 `FlexibleArgumentParser`
  - 加载子命令：`generate`（来自 `fastvideo/entrypoints/cli/generate.py`）
  - 解析参数后，调用子命令的 `cmd(args)`

关键代码引用：
- 子命令类：`fastvideo/entrypoints/cli/generate.py` 的 `GenerateSubcommand`
  - `_get_init_arg_names()` 返回初始化 `VideoGenerator` 用的参数键：`["num_gpus", "tp_size", "sp_size", "model_path"]`
  - `_get_generation_arg_names()` 返回采样参数（来自 `SamplingParam` 的字段）
  - `cmd()`：
    1) 校验 `--model-path` 与 `--prompt`/`--prompt-txt` 互斥要求
    2) 将 CLI 参数拆分为 `init_args`（给初始化）与 `generation_args`（给生成）
    3) 调用 `VideoGenerator.from_pretrained(model_path, **init_args)`
    4) 调用 `generator.generate_video(prompt=..., **generation_args)`

---

### 2) VideoGenerator 构造与全局推理参数组装

- 入口：`fastvideo/entrypoints/video_generator.py`
  - `VideoGenerator.from_pretrained(model_path, **kwargs)`：
    - 将 `model_path` 与 CLI init 参数合并，构造 `FastVideoArgs.from_kwargs(**kwargs)`（见 `fastvideo/fastvideo_args.py`）
      - 其中包含 `PipelineConfig.from_kwargs(...)` 的创建与更新（见下一节）
    - 转到 `VideoGenerator.from_fastvideo_args(fastvideo_args)`
  - `from_fastvideo_args`：
    - 选择执行器：`Executor.get_class(fastvideo_args)`，默认 `distributed_executor_backend == "mp"` → `MultiprocExecutor`
    - 返回 `VideoGenerator(fastvideo_args, executor_class=..., ...)`

FastVideoArgs 要点（`fastvideo/fastvideo_args.py`）：
- 校验与填充：`check_fastvideo_args()` 会设置 `tp_size/sp_size` 默认值、检查分布式参数合法性等
- 包含嵌套的 `pipeline_config`（下一节）与 offload/编译/STA/VSA 等运行开关

PipelineConfig 选择（`fastvideo/configs/pipelines/base.py` + `.../registry.py`）：
- `PipelineConfig.from_kwargs(...)` 内部调用 `get_pipeline_config_cls_from_name(model_path)`（`fastvideo/configs/pipelines/registry.py`）
  - 基于模型 ID 做精确/部分匹配：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` → `WanT2V480PConfig`（`fastvideo/configs/pipelines/wan.py`）
- 本例 CLI 还传了 `--flow-shift 8.0`，覆盖默认 `WanT2V480PConfig.flow_shift`（默认 3.0）

Sampling 参数来源（用于生成阶段）：
- `SamplingParam.add_cli_args(...)`（`fastvideo/configs/sample/base.py`）为 `height/width/num_frames/steps/guidance_scale/...` 暴露 CLI
- 生成时再组合成 `ForwardBatch`（后述）

---

### 3) 执行器与多进程 Worker 启动

- 选择：`Executor.get_class(...)`（`fastvideo/worker/executor.py`）→ `MultiprocExecutor`
- 初始化：`MultiprocExecutor._init_executor()`（`fastvideo/worker/multiproc_executor.py`）
  - `world_size = fastvideo_args.num_gpus`（脚本是 1）
  - 为每个 rank 启动 `WorkerMultiprocProc` 子进程；通过管道进行 RPC
  - 在子进程内：`WorkerWrapperBase.init_worker(...)` 创建 `Worker`，随后 `worker.init_device()` 完成本地设备与分布式初始化，并构建推理管线

Worker 初始化与构建 Pipeline（`fastvideo/worker/gpu_worker.py`）：
- `init_device()`：设置设备、初始化 `torch.distributed`、然后 `self.pipeline = build_pipeline(self.fastvideo_args)`

---

### 4) Pipeline 构建与模块加载

- 构建：`fastvideo/pipelines/__init__.py::build_pipeline(fastvideo_args, pipeline_type=PipelineType.BASIC)`
  1) `maybe_download_model(model_path)`：必要时从 Hugging Face Hub 下载模型（`fastvideo/utils.py`）
  2) `verify_model_config_and_directory(model_path)`：读取 `model_index.json`
  3) 根据 `_class_name` 与 `workload_type` 解析到具体 Pipeline 类（`fastvideo/pipelines/pipeline_registry.py`）
  4) 实例化：例如本例为 `fastvideo/pipelines/basic/wan/wan_pipeline.py::WanPipeline(model_path, fastvideo_args)`

WanPipeline 结构（`wan_pipeline.py`）：
- `_required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]`
- `initialize_pipeline(...)`：将 scheduler 替换为 `FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)`（本例为 8.0）
- `create_pipeline_stages(...)`：按顺序注册推理阶段（详见第 6 节）

ComposedPipelineBase 基类（`fastvideo/pipelines/composed_pipeline_base.py`）：
- `__init__` 里调用 `self.load_modules(...)` 加载各组件；随后在 `forward()` 首次调用前 `post_init()` 确保已完成 `initialize_pipeline` 与 `create_pipeline_stages`
- 组件加载 `load_modules(...)`：
  - 读取 `model_index.json`，逐个模块使用 `PipelineComponentLoader.load_module(...)`（`fastvideo/models/loader/component_loader.py`）
    - `TransformerLoader`：读 `transformer/*.safetensors`，按 `dit_precision`（缺省 bf16）构建，并可按 `dit_cpu_offload/use_fsdp_inference/pin_cpu_memory` 做 FSDP 分片/CPU offload
    - `VAELoader`：读 `vae/*.safetensors`，按 `vae_precision`（Wan 缺省 fp32）构建
    - `TextEncoderLoader`/`TokenizerLoader`：加载 T5 文本编码器与分词器
    - `SchedulerLoader`：先按 diffusers 配置构建，再在 `WanPipeline.initialize_pipeline` 中被自定义 scheduler 覆盖

---

### 5) 调用 generate_video 与批量 prompt 处理

- 入口：`VideoGenerator.generate_video(...)`（`fastvideo/entrypoints/video_generator.py`）
  - 若传入 `--prompt-txt`，则依次读取文件每一行作为一个 prompt 批次处理
  - 每个 prompt：
    1) 组装/更新 `SamplingParam`（若未显式传入则 `SamplingParam.from_pretrained(model_path)` 后再被 CLI 覆盖）
    2) 计算输出文件路径（对非法字符、重名做处理）
    3) 调用 `_generate_single_video(prompt, sampling_param, **kwargs)`

注意：`SamplingParam.seed` 会存入 `ForwardBatch.seed`。当前代码中随机数生成器 `generator` 未在此处由 `seed` 显式构建并传入（`LatentPreparationStage` 会用到 `batch.generator`）。

---

### 6) 单个视频生成：ForwardBatch 构造 → 执行器远程 forward → 分阶段推理

单视频路径：`VideoGenerator._generate_single_video(...)`（`fastvideo/entrypoints/video_generator.py`）
- 校验与对齐：
  - 若使用 VAE 时间压缩（Wan），`orig_latent_num_frames = (num_frames-1)//4 + 1`
  - 若不能被 `num_gpus` 整除，会按“向上/向下取整”策略调整 `num_frames`（本例 `num_gpus=1` 不触发）
  - `height/width` 对齐到 16 的倍数
- 记录 `n_tokens` 等调试信息；构建 `ForwardBatch`：
  - 关键字段：`prompt/negative_prompt/num_frames/height/width/guidance_scale/num_inference_steps/eta/n_tokens/...`
- 执行：`executor.execute_forward(batch, fastvideo_args)` → 通过多进程 RPC 交给 rank0 Worker 的 `pipeline.forward(...)`

Pipeline.forward 的阶段序（`ComposedPipelineBase.forward` 调用已注册的 stages）：
1) InputValidationStage（`fastvideo/pipelines/stages/input_validation.py`）：输入合法性检查（文件存在、尺寸范围等）
2) TextEncodingStage（`stages/text_encoding.py`）：
   - 使用分词器与 T5 文本编码器得到 `prompt_embeds`（与可选 `negative_prompt_embeds`，当 `guidance_scale>1` 时启用 CFG）
3) ConditioningStage（`stages/conditioning.py`）：
   - 预留用于 CFG 拼接等（当前实现基本直通）
4) TimestepPreparationStage（`stages/timestep_preparation.py`）：
   - 基于 `num_inference_steps` 或自定义 `timesteps/sigmas`，调用 scheduler 的 `set_timesteps(...)`，写入 `batch.timesteps`
5) LatentPreparationStage（`stages/latent_preparation.py`）：
   - 依据 VAE 压缩比计算 latent 的形状：`(B, C, T_lat, H/8, W/8)`（Wan 默认时空压缩 4×8）
   - 若 `batch.latents` 为空，则用 `randn_tensor(shape, generator=batch.generator, ...)` 采样初始噪声，并乘以 `scheduler.init_noise_sigma`（若有）
   - 记录 `raw_latent_shape`
6) DenoisingStage（`stages/denoising.py`）：
   - 选择注意力后端：`get_attn_backend(...)`（受 `FASTVIDEO_ATTENTION_BACKEND/FASTVIDEO_ATTENTION_CONFIG` 等影响；脚本中变量为空 → 走默认策略）
   - 处理（可选）序列并行：当 `sp_size>1` 时按时间维切分/聚合（本例 `sp_size=1` 不触发）
   - 逐 timestep：
     - 准备 `latent_model_input = scheduler.scale_model_input(latents, t)` 与条件/无条件（CFG）分支所需的嵌入
     - Transformer 前向：预测噪声 `noise_pred`（以及 `noise_pred_uncond` 用于 CFG）
     - 应用 CFG：`noise = uncond + cfg_scale * (cond - uncond)`，可选 `guidance_rescale`
     - 调度一步：`latents = scheduler.step(noise, t, latents, ...)`
     - 可选保存轨迹（`return_trajectory_*`）
7) DecodingStage（`stages/decoding.py`）：
   - 将最终 `latents` 送入 VAE 解码成像素域视频张量（[0,1] 范围），支持 tiling；Wan 缺省 `vae_tiling=False`
   - 可选 `vae_cpu_offload`（本例 False）

阶段执行完毕后，`ForwardBatch.output` 为 `[B, C, T, H, W]`。Worker 返回结果到主进程。

---

### 7) 结果收集与保存

- 返回主进程后，`VideoGenerator._generate_single_video`：
  - 通过 `einops.rearrange` 将 `[B, C, T, H, W]` 转为逐帧网格图，汇集为帧序列
  - `imageio.mimsave(output_path, frames, fps=...)` 保存为 mp4（`outputs_video/` 下以 prompt 片段命名，自动去除非法字符并避免重名）
  - 若 `return_frames=True` 则直接返回帧数据；否则返回包含 `samples/frames/prompt/size/generation_time/...` 的字典

---

### 8) 本例参数对行为的具体影响小结

- 模型与配置：
  - `model_path = Wan-AI/Wan2.1-T2V-1.3B-Diffusers` → 选择 `WanT2V480PConfig`
  - `--flow-shift 8.0` 覆盖默认 3.0，用于 `FlowUniPCMultistepScheduler(shift=8.0)` 的时间移位
- 采样与尺寸：
  - `height=480, width=832, num_frames=77, fps=16, steps=50`
  - Wan 的 latent 尺寸按 VAE 压缩比：`H_lat=480/8=60, W_lat=832/8=104, T_lat=(77-1)//4+1=20`
- 指导与负面提示：
  - `guidance_scale=6.0` → 启动 CFG；`--negative-prompt` 将在 `TextEncodingStage` 编码并在 `DenoisingStage` 用于无条件分支
- 并行与 offload：
  - `num_gpus=1, tp=1, sp=1` → 单进程单设备，无序列并行切分
  - `text-encoder-cpu-offload=True` → 文本编码器优先放在 CPU（节省显存）；`dit/vae` offload 关闭，保持在 GPU 上
- 随机种子：
  - `--seed 1024` 存入 `ForwardBatch.seed`；当前代码未在此处由该 seed 构造 `torch.Generator` 传入 `LatentPreparationStage`，因此具体随机性是否复现取决于全局 RNG 状态（可按需在后续版本中补充生成器构造）

---

### 9) 关键调用链（简表）

1. shell: `fastvideo generate ...`
2. `fastvideo.entrypoints.cli.main:main`
3. `GenerateSubcommand.cmd`（解析/切参）
4. `VideoGenerator.from_pretrained` → `FastVideoArgs.from_kwargs` → `PipelineConfig.from_kwargs`
5. `VideoGenerator.from_fastvideo_args` → `Executor.get_class` → `MultiprocExecutor`
6. 子进程 `Worker.init_device` → `build_pipeline` → `WanPipeline`
7. `WanPipeline.post_init`：`initialize_pipeline`（scheduler）→ `create_pipeline_stages`
8. 主进程 `VideoGenerator.generate_video`（批量 prompt）→ `_generate_single_video`
9. `executor.execute_forward` → 子进程 `pipeline.forward` 依序运行各 `Stage`
10. 子进程返回 `ForwardBatch.output` → 主进程整理帧并 `imageio.mimsave`

---

### 10) 相关文件索引

- CLI 与入口
  - `pyproject.toml`（console script）
  - `fastvideo/entrypoints/cli/main.py`，`fastvideo/entrypoints/cli/generate.py`
- 全局参数与配置
  - `fastvideo/fastvideo_args.py`
  - `fastvideo/configs/pipelines/base.py`，`.../pipelines/registry.py`
  - `fastvideo/configs/pipelines/wan.py`
  - `fastvideo/configs/sample/base.py`，`.../sample/wan.py`
- 执行器与 worker
  - `fastvideo/worker/executor.py`
  - `fastvideo/worker/multiproc_executor.py`，`fastvideo/worker/worker_base.py`，`fastvideo/worker/gpu_worker.py`
- Pipeline 框架与构建
  - `fastvideo/pipelines/__init__.py`
  - `fastvideo/pipelines/composed_pipeline_base.py`
  - `fastvideo/pipelines/pipeline_registry.py`
  - `fastvideo/pipelines/basic/wan/wan_pipeline.py`
  - `fastvideo/pipelines/stages/*.py`（text_encoding / conditioning / timestep_preparation / latent_preparation / denoising / decoding）
- 模型组件加载
  - `fastvideo/models/loader/component_loader.py`
- 工具与下载
  - `fastvideo/utils.py`（HF snapshot 下载、配置校验、对齐尺寸等）


