# Phase：Add WanGame（把 Wangame 接入新 distillation 框架）

> 目标：在 **不回到 legacy training pipeline** 的前提下，让 `fastvideo/training/distillation.py`
> 可以通过 YAML（schema v2）跑起 **WanGame** 的 finetune / distill（优先 finetune）。
>
> 本文件只做“代码层面规划”，不修改代码。

---

## 0) 我们现在在 FastVideo 里哪里能看到 WanGame？

WanGame 不是“想象中的模型”，FastVideo 里已经有一整套（legacy）实现：

- **模型（DiT）**：`fastvideo/models/dits/wangame/*`
  - 关键类：`WanGameActionTransformer3DModel`
- **pipeline config**：`fastvideo/configs/models/dits/wangamevideo.py` +
  `fastvideo/configs/pipelines/wan.py:WanGameI2V480PConfig`
- **推理 pipeline（ODE）**：`fastvideo/pipelines/basic/wan/wangame_i2v_pipeline.py`
  - `WanGameActionImageToVideoPipeline`
- **推理 pipeline（SDE/DMD）**：`fastvideo/pipelines/basic/wan/wangame_causal_dmd_pipeline.py`
  - `WanGameCausalDMDPipeline`
- **训练 pipeline（legacy）**：`fastvideo/training/wangame_training_pipeline.py`
  - 重点参考：`_prepare_dit_inputs()` + `_build_input_kwargs()`
- **数据 schema**：`fastvideo/dataset/dataloader/schema.py:pyarrow_schema_wangame`

结论：要把 WanGame 接到新框架，核心工作是把 legacy pipeline 里
“raw batch -> model forward primitives”的逻辑迁入 **adapter**。

---

## 1) 现在我们的新框架需要什么（接口映射）

新框架（Phase 3.x）的一条 run path 是：

- `fastvideo/training/distillation.py`
  -> parse YAML (`fastvideo/distillation/utils/config.py`)
  -> `fastvideo/distillation/dispatch.py:build_runtime_from_config()`
  -> `fastvideo/distillation/trainer.py:DistillTrainer.run(method, dataloader, ...)`

其中 **model plugin** 必须返回：

- `RoleManager`（roles -> RoleHandle -> modules/optimizers/schedulers）
- `Adapter`（operation-centric primitives）
- `DataLoader`
- `Validator`（可选，method 决定怎么调用）

因此“加 Wangame”就是补齐一个新的 model plugin + adapter + validator。

---

## 2) 设计目标（Add WanGame 的 Definition of Done）

### ✅ 最小可用（建议先交付这一档）

- 支持 `recipe.family: wangame`
- 支持 `recipe.method: finetune`（优先）
- `log_validation: true` 时能正确走 `WanGameActionImageToVideoPipeline`（ODE）
- 训练 input 与 legacy `fastvideo/training/wangame_training_pipeline.py` 一致：
  - I2V concat：`noisy_video_latents + mask + image_latents`
  - action conditioning：`viewmats / Ks / action` 来自 `process_custom_actions(...)`

### ✅ 进阶（第二档）

- 支持 `recipe.method: dmd2`（distill）
- validation 根据 `ValidationRequest.sampler_kind` 切换：
  - `ode` -> `WanGameActionImageToVideoPipeline`
  - `sde` -> `WanGameCausalDMDPipeline`

> 注意：WanGame 的训练通常不依赖 text CFG；DMD2 中的 uncond/cond
> 在 wangame 上可能等价（见风险点）。

---

## 3) 文件改动规划（建议的最小集合）

### 3.1 新增：model plugin

- [ ] `fastvideo/distillation/models/wangame.py`
  - `@register_model("wangame")`
  - 主要职责：
    - 设置 `training_args.override_transformer_cls_name = "WanGameActionTransformer3DModel"`
      （必要时增加可配置项，支持 causal transformer）
    - 加载 shared：`vae`（用于 `normalize_dit_input("wan", ...)`）+ `noise_scheduler`
    - 为每个 role 加载 `transformer`（以及可选 `transformer_2`）
    - `apply_trainable(...)` + activation ckpt（沿用 wan plugin 逻辑）
    - 构建 `RoleManager`
    - 构建 `WanGameAdapter`
    - dataloader：使用 `pyarrow_schema_wangame`
    - `log_validation` 时构建 `WanGameValidator`

### 3.2 新增：adapter

- [ ] `fastvideo/distillation/adapters/wangame.py`
  - 复用 `WanAdapter` 的通用 mechanics（timestep/noise/attn_metadata/backward 的模式）
  - 重点实现（对齐 legacy `wangame_training_pipeline.py`）：
    - `prepare_batch(raw_batch, current_vsa_sparsity, latents_source=...)`
      - 从 parquet batch 取：
        - `vae_latent`（video x0）
        - `clip_feature`（image_embeds）
        - `first_frame_latent`（image_latents）
        - `keyboard_cond` / `mouse_cond`
        - `pil_image`（给 validation 或 debug）
      - 计算 timesteps/noise/sigmas，得到 noisy_video_latents
      - 构造 I2V 输入：
        - 生成 `mask_lat_size`（与 legacy 一致）
        - `noisy_model_input = cat([noisy_video_latents, mask, image_latents], dim=1)`
      - `process_custom_actions(...)` -> `viewmats / Ks / action`
      - 组装 transformer forward 所需的 `input_kwargs`
    - `predict_noise(handle, noisy_latents, t, batch, conditional, attn_kind)`
    - `predict_x0(handle, noisy_latents, t, batch, conditional, attn_kind)`
      - **关键约束**：`noisy_latents` 参数应代表“noisy video latents”；
        adapter 内部用 batch 里的 `image_latents/mask/action` 拼出完整 hidden_states。
    - `add_noise(clean_latents, noise, t)`：只对 video latent 做 add_noise
    - `backward(loss, ctx, ...)`：延续现有 forward_context 机制

> 备注：`conditional` 在 wangame 上可能不区分（先忽略即可），但接口必须收敛一致，
> 以兼容 `DMD2Method` 的调用方式。

### 3.3 新增：validator

- [ ] `fastvideo/distillation/validators/wangame.py`
  - API 对齐 `WanValidator`：`log_validation(step, request=ValidationRequest)`
  - pipeline 选择：
    - `request.sampler_kind == "ode"`：`WanGameActionImageToVideoPipeline`
    - `request.sampler_kind == "sde"`：`WanGameCausalDMDPipeline`
  - batch 构造参考 legacy：
    - `fastvideo/training/wangame_training_pipeline.py:_prepare_validation_batch`
  - 只要 method 提供 `sample_handle`（通常 student），validator 就能跑。

### 3.4 改动：dispatch builtin registrations

- [ ] `fastvideo/distillation/dispatch.py:ensure_builtin_registrations()`
  - 显式 import 新增的 `fastvideo.distillation.models.wangame`

### 3.5 可选：dataloader util

当前 `fastvideo/distillation/utils/dataloader.py` 只有 T2V helper。
wangame 需要 I2V+action schema，因此建议：

- [ ] `fastvideo/distillation/utils/dataloader.py`
  - 新增 `build_parquet_wangame_train_dataloader(training_args, parquet_schema=pyarrow_schema_wangame)`
  - 内部仍调用 `fastvideo.dataset.build_parquet_map_style_dataloader(...)`

---

## 4) 配置（YAML）规划（schema v2）

建议新增一个最小 finetune 配置（示意）：

```yaml
recipe:
  family: wangame
  method: finetune

roles:
  student:
    family: wangame
    path: weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers
    trainable: true

training:
  data_path: /abs/path/to/wangame/preprocessed/combined_parquet_dataset
  # shape / dist / lr / validation 同 wan 的 schema 写法即可

pipeline_config:
  flow_shift: 5
  sampler_kind: ode

method_config:
  attn_kind: dense
```

如果要支持 DMD2（第二档），需要：
- roles 增加 teacher/critic（如果暂时没有更大 teacher，可先 teacher==student 跑通 e2e）
- `pipeline_config.sampler_kind: sde`（validation 走 `WanGameCausalDMDPipeline`）
- `method_config` 增加 DMD2 必需字段（`rollout_mode`, `dmd_denoising_steps`, ...）

---

## 5) 关于目录结构：要不要 `models/wan/wan.py + models/wan/wangame.py`？

**不需要强制做**。

当前 dispatch 只关心注册 key（`@register_model("...")`），文件放哪都行。
因此建议先保持扁平：

- `fastvideo/distillation/models/wan.py`
- `fastvideo/distillation/models/wangame.py`

等到 wan 系变体更多时（wan_t2v / wan_i2v / wangame / lingbot / ...），
再做结构化重排成 `models/wan/*`，并把共同的 loader 逻辑抽成内部 helper。

---

## 6) 风险点 / 决策点（需要提前讲清楚）

1) **DMD2 的 CFG（cond/uncond）语义在 wangame 上可能不存在**
   - wangame training pipeline 里 `encoder_hidden_states=None`，主要 conditioning 是 image+action。
   - 我们可以先让 `conditional` flag 在 adapter 里等价处理：
     - `conditional=True/False` 都走同一条分支
     - DMD2 的 `real_cond_x0 - real_uncond_x0` 近似为 0，guidance scale 失效但训练可跑通
   - 若未来确实需要“uncond”，需要定义：
     - uncond 是否代表“去掉 action”？“去掉 image cond”？还是别的？

2) **train_action_only / action_warmup_steps（细粒度 trainable）**
   - legacy `wangame_training_pipeline.py` 支持只训练 action 路径参数（pattern-based）
   - 目前新框架 roles 只有 `trainable: bool`（整模块开关），不足以表达这一点
   - 建议先把这一点作为可选增强：
     - 最小实现：先不做（或要求用户在 wangame 上 train 全模型）
     - 完整实现：在 `models/wangame.py` 内做参数 pattern 的 requires_grad 筛选

3) **validation pipeline 的 ODE/SDE 差异**
   - 需要 validator 根据 `sampler_kind` 切 pipeline，避免出现“看起来像 ODE”但期望 SDE 的错觉。

---

## 7) 验收（我们如何确认接入是对的）

- finetune：
  - e2e 跑通（2~10 steps）+ 能输出 validation 视频
  - 与 legacy finetune 的 step0/stepN 视觉趋势大体一致（不追求完全 bitwise）
- distill（若做第二档）：
  - e2e 跑通（few-step）
  - validation 选择 `sde` 时，视觉应接近 legacy DMD pipeline 的 sampling 形态

