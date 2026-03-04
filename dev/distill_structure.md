# FastVideo Distill 训练结构梳理（现状）

本文是对当前仓库 distill 训练代码的“结构/逻辑”阅读笔记，目标是把：

- 入口在哪里
- 训练时有哪些模型（student/teacher/critic）
- 每一步在算什么 loss、更新谁
- Self-forcing 相对 DMD 的差异
- 目前实现的耦合点/限制

写清楚，方便后续重构和对齐实现细节。

## 代码入口与文件分布

- 训练主逻辑
  - `fastvideo/training/distillation_pipeline.py`：`DistillationPipeline`
    （DMD/DMD2 风格 distillation）
  - `fastvideo/training/self_forcing_distillation_pipeline.py`：
    `SelfForcingDistillationPipeline`（Self-forcing distillation）
- Wan 封装/可运行入口（torchrun 直接跑这些文件）
  - `fastvideo/training/wan_distillation_pipeline.py`：
    `WanDistillationPipeline`
  - `fastvideo/training/wan_self_forcing_distillation_pipeline.py`：
    `WanSelfForcingDistillationPipeline`
  - `fastvideo/training/wan_i2v_distillation_pipeline.py`：
    `WanI2VDistillationPipeline`
- 推理/验证用 pipeline（用于 `_log_validation`）
  - `fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py`：`WanDMDPipeline`
  - `fastvideo/pipelines/basic/wan/wan_causal_dmd_pipeline.py`：
    `WanCausalDMDPipeline`
  - I2V：`fastvideo/pipelines/basic/wan/wan_i2v_dmd_pipeline.py`
- 断点续训/保存
  - `fastvideo/training/training_utils.py`：
    `save_distillation_checkpoint` / `load_distillation_checkpoint`
- 测试
  - `fastvideo/tests/training/distill/test_distill_dmd.py`
  - `fastvideo/tests/training/self-forcing/test_self_forcing.py`

## 先说结论：当前 distill 覆盖范围

1. **模型侧目前明显偏 Wan**
   - `TrainingPipeline._normalize_dit_input` 固定走
     `normalize_dit_input('wan', ...)`
   - teacher/critic 加载时强制设置
     `training_args.override_transformer_cls_name = "WanTransformer3DModel"`
   - 因此“想 distill 其它架构”目前不是简单换配置就能跑通的。

2. **算法侧两条主线**
   - **DMD/DMD2**：`DistillationPipeline`
     （student + teacher(real score) + critic(fake score)）
   - **Self-forcing**：`SelfForcingDistillationPipeline`
     （在 DMD2 框架里，把 student 前向改成 causal/self-forcing 的
     blockwise generation + KV cache）

3. **默认/隐含假设**
   - DMD 路径里多处 `unflatten(0, (1, T))`，等价于默认 `batch_size == 1`
     （脚本里确实几乎都设 `--train_batch_size 1`）。
   - DMD loss 需要 teacher 的 cond/uncond 两次前向做 CFG；uncond embedding
     目前依赖 validation 阶段用 negative prompt 编码得到
     （见 “unconditional conditioning 的来源” 一节）。

## DistillationPipeline（DMD/DMD2）训练逻辑

### 1) 参与训练/冻结的模块（roles）

`DistillationPipeline.load_modules()` + `initialize_training_pipeline()` 搭出
“三模型”结构：

- **student / generator**：`self.transformer`
  - 来自 `--pretrained_model_name_or_path`
  - 这是要训练的主模型
  - VAE：`self.vae` 同样来自 student pipeline，但在 distill 中 **冻结**
- **teacher / real score**：`self.real_score_transformer`
  - 来自 `--real_score_model_path`
  - `requires_grad_(False)` + `eval()`
- **critic / fake score**：`self.fake_score_transformer`
  - 来自 `--fake_score_model_path`
  - **要训练**，有自己的一套 optimizer/scheduler
- （可选）MoE/双 transformer 支持（teacher/critic 优先）
  - teacher/critic 会尝试加载 `transformer_2`
  - 用 `boundary_ratio`（→ `boundary_timestep`）决定高噪/低噪 expert 选择：
    - `_get_real_score_transformer(t)`
    - `_get_fake_score_transformer(t)`（并设置 `train_fake_score_transformer_2`）

> 备注：student 侧也可能有 `transformer_2`（例如 Wan2.2），但 DMD 路径的
> optimizer step 目前主要写死在 `self.transformer` 上；self-forcing 路径则会
> 在 `optimizer` / `optimizer_2` 间二选一。重构时建议统一“哪个 timestep
> 更新哪个 expert”的决策来源与实现方式。

### 2) 关键超参/调度

- `generator_update_interval`
  - `step % generator_update_interval == 0` 才更新 student（generator）
  - critic 每 step 都更新
- `pipeline_config.dmd_denoising_steps` → `self.denoising_step_list`
  - 这是 “student 训练/模拟推理” 用到的一组离散 timestep
  - 若 `--warp_denoising_step`，会根据 scheduler 的 time shift 做一次映射
- timestep sampling 范围（用于 DMD loss / critic loss 的随机 timestep）
  - `min_timestep = min_timestep_ratio * num_train_timesteps`
  - `max_timestep = max_timestep_ratio * num_train_timesteps`
  - 最终都会 clamp 到 `[min_timestep, max_timestep]`
- teacher CFG
  - `real_score_guidance_scale`

### 3) batch / tensor 形状约定（T2V）

从 parquet dataloader 读到的典型字段：

- `vae_latent`: `[B, C, T_lat, H_lat, W_lat]`
- `text_embedding`, `text_attention_mask`
- `info_list`（日志用）

进入 distill 训练后，关键变换：

- `_prepare_dit_inputs` 里把 `latents` permute 成 `[B, T_lat, C, H, W]`
  并把 `self.video_latent_shape` 记录为这个形状
- `_build_distill_input_kwargs` 会再 permute 回 `[B, C, T_lat, H, W]`
  作为 transformer 的 `hidden_states`

### 4) 一个 step 内发生什么（train_one_step）

`train_one_step` 的大体结构：

1. 收集 `gradient_accumulation_steps` 个 batch
   - 每个 batch 会构建 attention metadata，并复制出一份 `attn_metadata_vsa`
   - `attn_metadata_vsa`：保留当前 VSA sparsity（稀疏注意力）
   - `attn_metadata`：强行把 `VSA_sparsity = 0`（等价 dense）
   - 代码用法上：student forward 常用 `attn_metadata_vsa`，
     teacher/critic forward 用 `attn_metadata`（dense）

2. **可选更新 student（generator）**
   - 条件：`current_step % generator_update_interval == 0`
   - 对每个 accumulated batch：
     - 先算 `generator_pred_video`
       - `--simulate_generator_forward`：`_generator_multi_step_simulation_forward`
         （从纯噪声开始按 `denoising_step_list` 模拟推理，最后一步保留梯度）
       - 否则：`_generator_forward`
         （对数据里的 `vae_latent` 加噪声做一次 denoise；这里把 batch 维度
         `unflatten` 写死成 `(1, T)`，隐含 `B=1`）
     - 再算 `dmd_loss = _dmd_forward(generator_pred_video, batch)` 并反传
   - clip grad → `self.optimizer.step()` → EMA update（若开启）

3. **更新 critic（fake score）**（每个 step 都做）
   - 对每个 accumulated batch：
     - 先 `generator_pred_video`（no_grad）
     - 再随机采样 `fake_score_timestep`，把 `generator_pred_video` 加噪声得到
       `noisy_generator_pred_video`
     - critic 预测 `fake_score_pred_noise`
     - 目标是 `target = noise - generator_pred_video`
     - `flow_matching_loss = mean((pred - target)^2)`
   - clip grad → critic optimizer step → critic scheduler step
   - 注意：代码里 `self.lr_scheduler.step()`（student 的 lr scheduler）也会在
     critic 更新后每 step 都跑一次，即使这个 step 没有更新 student。
     如果 `generator_update_interval > 1`，这会让 student 的 lr schedule
     “按 step 走”而不是“按 optimizer step 走”。

### 5) DMD loss 细节（_dmd_forward）

简化写成：

- 输入：`x = generator_pred_video`（student 产出的 clean latent）
- 随机采样 timestep `t`
- 加噪声：`x_t = add_noise(x, eps, t)`
- critic（fake score）得到 `x_hat_fake`
- teacher（real score）分别做 cond/uncond 得到 `x_hat_real_cond`,
  `x_hat_real_uncond`，再做 CFG：

  `x_hat_real = x_hat_real_cond + (x_hat_real_cond - x_hat_real_uncond) * s`

- 构造一个“梯度方向”（代码变量名为 `grad`）：

  `g = (x_hat_fake - x_hat_real) / mean(abs(x - x_hat_real))`

- DMD2 风格的 stop-grad target：

  `L = 0.5 * mse(x, (x - g).detach())`

并把中间 latent 存到 `training_batch.dmd_latent_vis_dict` 里用于可视化。

### 6) unconditional conditioning 的来源

DMD loss 里 teacher 需要 uncond forward：

- `training_batch.unconditional_dict` 来自 `_prepare_dit_inputs`，但它只在
  `self.negative_prompt_embeds` 已经存在时才会被创建
- `self.negative_prompt_embeds` 当前是在 `_log_validation` 里，通过
  `validation_pipeline.prompt_encoding_stage` 对 `SamplingParam.negative_prompt`
  编码得到（会同时设置 attention mask）
- 因此目前的“默认用法”是：训练脚本基本都会开 `--log_validation`，保证在
  train 开始前 `_log_validation` 跑一次，从而初始化 negative prompt embedding

> 重构时建议显式化这条依赖：uncond embedding 不应依赖
> “是否开启 validation logging”。

## SelfForcingDistillationPipeline（Self-forcing）训练逻辑

Self-forcing pipeline 继承自 `DistillationPipeline`，但有三个核心变化：

### 1) scheduler 换成 SelfForcingFlowMatchScheduler

`initialize_training_pipeline` 里把 `self.noise_scheduler` 替换为
`SelfForcingFlowMatchScheduler(..., training=True, extra_one_step=True)`，
并引入 self-forcing 的一批超参：

- `dfake_gen_update_ratio`：generator 每 N step 更新一次（其余 step 只更 critic）
- `num_frame_per_block`, `independent_first_frame`, `same_step_across_blocks`,
  `last_step_only`, `context_noise` 等

### 2) generator forward 变成 causal/blockwise + KV cache

`_generator_multi_step_simulation_forward` 在 self-forcing 里被彻底重写：

- 按 block 生成视频（每 block `num_frame_per_block` 帧）
- 每个 block 内做 denoising loop（遍历 `denoising_step_list`），并用随机
  `exit_flag` 决定在哪个 timestep 停止并保留梯度
- KV cache / cross-attn cache
  - `_initialize_simulation_caches` 预分配 cache 张量
  - 每生成完一段会用 `context_noise`（通常 0）再跑一次 timestep=0 的 forward
    来更新 cache（no_grad）
- gradient masking
  - 当生成帧数 > 最小帧数时，会把一部分早期帧 detach，确保只在后面某些帧上回传梯度
  - 额外还有 “last 21 frames” 的处理逻辑：把更早的帧 decode→encode 变成
    image latent，再拼回去

总体上，这是把“推理时的 autoregressive/cached 过程”搬进训练。

### 3) train_one_step 变成“交替训练 generator / critic”

- `train_generator = (step % dfake_gen_update_ratio == 0)`
- generator 更新时
  - `generator_loss` 默认仍然走 `_dmd_forward`
    （所以仍依赖 teacher + critic）
  - optimizer step 会在 `transformer` 和 `transformer_2` 之间二选一
    （基于 `self.train_transformer_2`）
- critic 更新时
  - `critic_loss` 仍然是 `faker_score_forward`（flow matching）

## Checkpoint / Resume（distill 专用）

`save_distillation_checkpoint` / `load_distillation_checkpoint` 的要点：

- 保存 generator + critic 的 distributed checkpoint（可用于 resume）
- 同时保存 “consolidated generator weights”（用于推理/部署）
- 支持（可选）MoE 的 `transformer_2` / `critic_2` / `real_score_2`
- 保存 random state（保证 resume 后 timestep/noise 采样一致）
- EMA 以单独文件的方式保存（避免不同 rank 形状不一致）

## 目前实现中最重要的耦合点/限制（重构 checklist）

- **Wan-only 逻辑散落多处**
  - normalize 走 `'wan'`
  - teacher/critic 强制 `WanTransformer3DModel`
- **batch_size 隐式假设为 1**
  - `_generator_forward`, `_dmd_forward`, `faker_score_forward` 里对
    `add_noise(...).unflatten(0, (1, T))` 的写法
- **uncond embedding 的初始化路径不显式**
  - 依赖 `_log_validation` 运行过才能构造 `unconditional_dict`
- **student lr scheduler 的 step 粒度**
  - 当前按 “训练 step” 走，而不是按 “generator optimizer step” 走
- **MoE/dual-transformer 的选择与更新逻辑不统一**
  - teacher/critic 有 boundary 选择逻辑
  - student 的 optimizer 选择逻辑在 DMD 与 self-forcing 两条路径不一致
