# Phase：Add WanGame（把 Wangame 接入新 distillation 框架）

> 目标：在 **不回到 legacy training pipeline** 的前提下，让 `fastvideo/training/distillation.py`
> 可以通过 YAML（schema v2）跑起 **WanGame** 的 finetune / distill（优先 finetune）。
>
> 本文件最初用于“代码层面规划”，现在也用来记录已落地的实现与遗留 TODO。

---

## 当前进展（已落地）

> 下面是“已经写进代码库并通过静态检查（`compileall` + ruff）”的部分。
> GPU 端到端训练/验证需要你在有 GPU 的机器上跑（我们这边环境可能没有 driver）。

### ✅ 已实现（最小可用：finetune）

- `recipe.family: wangame` + `recipe.method: finetune`
- 新增 model plugin / adapter / validator：
  - `fastvideo/distillation/models/wangame.py`
  - `fastvideo/distillation/adapters/wangame.py`
  - `fastvideo/distillation/validators/wangame.py`
- builtin dispatch 注册：
  - `fastvideo/distillation/dispatch.py:ensure_builtin_registrations()`
- dataloader helper（复用 parquet loader，支持 `path:N` + 多路径）：
  - `fastvideo/distillation/utils/dataloader.py:build_parquet_wangame_train_dataloader()`
- examples（可直接跑）：
  - `examples/distillation/wangame/finetune_wangame2.1_i2v_1.3B.yaml`
    - 已把 legacy `finetune_wangame.slurm` 的 `DATA_DIR` 语义搬进来
    - 用 YAML folded string（`>-`）让超长 `data_path` 更可读
  - `examples/distillation/wangame/finetune-temp.sh`

### ✅ 已实现（为 future DMD2 做好 primitives）

- `WanGameAdapter` 已提供 DMD2 所需的 operation primitives：
  - `num_train_timesteps / shift_and_clamp_timestep / add_noise`
  - `predict_noise / predict_x0 / backward`
  - attention metadata（dense/vsa）+ forward_context
- `WanGameValidator` 支持根据 `ValidationRequest.sampler_kind` 选 pipeline：
  - `ode` -> `WanGameActionImageToVideoPipeline`
  - `sde` -> `WanGameCausalDMDPipeline`

### ✅ 关键对齐说明（legacy vs 新框架）

- **训练 noising scheduler**：新框架 wangame 训练使用
  `FlowMatchEulerDiscreteScheduler`，与 legacy training loop（`TrainingPipeline`）
  一致（训练阶段并不使用 UniPC 做 noising）。
- **validation sampler**：validator 走 pipeline（ODE/SDE）时，仍由 pipeline
  自己持有对应 scheduler（例如 ODE pipeline 使用 `FlowUniPCMultistepScheduler`）。

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

- [x] `fastvideo/distillation/models/wangame.py`
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

- [x] `fastvideo/distillation/adapters/wangame.py`
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

- [x] `fastvideo/distillation/validators/wangame.py`
  - API 对齐 `WanValidator`：`log_validation(step, request=ValidationRequest)`
  - pipeline 选择：
    - `request.sampler_kind == "ode"`：`WanGameActionImageToVideoPipeline`
    - `request.sampler_kind == "sde"`：`WanGameCausalDMDPipeline`
  - batch 构造参考 legacy：
    - `fastvideo/training/wangame_training_pipeline.py:_prepare_validation_batch`
  - 只要 method 提供 `sample_handle`（通常 student），validator 就能跑。

### 3.4 改动：dispatch builtin registrations

- [x] `fastvideo/distillation/dispatch.py:ensure_builtin_registrations()`
  - 显式 import 新增的 `fastvideo.distillation.models.wangame`

### 3.5 可选：dataloader util

当前 `fastvideo/distillation/utils/dataloader.py` 只有 T2V helper。
wangame 需要 I2V+action schema，因此建议：

- [x] `fastvideo/distillation/utils/dataloader.py`
  - 新增 `build_parquet_wangame_train_dataloader(training_args, parquet_schema=pyarrow_schema_wangame)`
  - 内部仍调用 `fastvideo.dataset.build_parquet_map_style_dataloader(...)`

TODO（更通用的方向，暂不做）：
- [ ] 扩展到更多 dataset kind（webdataset / precomputed / ode-init ...），
  并用更统一的 config/dispatch 管理（例如 `DataSpec`）。

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

### 4.1 roles 的“自由扩展字段”策略（保留核心字段强校验）

建议 roles 仍强制解析这三个核心字段（保证错误信息清晰）：
- `family / path / trainable`

但允许 roles 下出现其它任意 key，并把它们原样保留（例如 `RoleSpec.extra`），
由 `models/wangame.py` 自行解释（例如 action-only train patterns / cls_name / init 行为）。

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
   先说明：在我们新框架里，`DMD2Method` 的 teacher CFG 语义来自 “文本 CFG”：
   - `conditional=True`：用 prompt（正向）作为 conditioning；
   - `conditional=False`：用 negative / uncond prompt 作为 conditioning；
   - 最终 teacher 的 “real score” 用类似：
     `real = uncond + guidance_scale * (cond - uncond)`。

   这在 Wan(T2V) 上成立，因为：
   - legacy / 我们的 `WanAdapter` 都显式构造了 negative prompt embeds（`ensure_negative_conditioning()`），
     并把它塞进 `training_batch.unconditional_dict`，从而 `conditional` flag 有真实语义差异。

   但在 **WanGame** 上，legacy 训练/推理几乎完全不依赖 text：
   - legacy train：`fastvideo/training/wangame_training_pipeline.py:_build_input_kwargs()`
     明确设置：
     - `"encoder_hidden_states": training_batch.encoder_hidden_states  # None (no text conditioning)`
     - `"encoder_hidden_states_image": image_embeds`（I2V conditioning）
     - `"viewmats" / "Ks" / "action"`（action conditioning）
   - legacy inference pipeline：`fastvideo/pipelines/basic/wan/wangame_i2v_pipeline.py`
     没有 `TextEncodingStage`；因此 `ForwardBatch.prompt_embeds` 为空。
     `LatentPreparationStage` 会把空 prompt embeds 变成一个 dummy（形状 `[B, 0, hidden_size]`），并且：
     - `batch.do_classifier_free_guidance = False`
     （见 `fastvideo/pipelines/stages/latent_preparation.py`）
   - legacy scripts/示例也在暗示“没 CFG”：
     - `examples/inference/basic/basic_wangame.py` 里 `guidance_scale=1.0`
     - `examples/training/finetune/WanGame2.1_1.3b_i2v/finetune_wangame.slurm`
       里 `validation_guidance_scale "1.0"`

   因此：如果我们直接把 DMD2 的 `conditional/unconditional` 套到 wangame 上，
   在 **不重新定义 uncond 的前提下**，它们很可能等价（cond == uncond），导致：
   - `real_cond_x0 - real_uncond_x0 ≈ 0`
   - guidance_scale 形式上存在，但语义上失效
   - 训练可能仍能跑通，但已经不是“文本 CFG 意义下的 DMD2”

   **建议的落地策略（按风险从低到高）：**
   - A（最小可用，推荐先做）：在 `wangame` 的 adapter 里把 `conditional` 当作 no-op
     （cond/uncond 同路），并在文档里明确 “wangame 暂不支持文本 CFG”。
   - B（定义 wangame 的 uncond 语义）：由 `method_config` 显式声明 uncond 的定义，例如：
     - `uncond_mode: none|zero_action|zero_image|zero_both`
     - `zero_action`：把 action/viewmats/Ks 置零或置 None（需要确认 transformer 对 None 的容忍度）
     - `zero_image`：把 `encoder_hidden_states_image` 置零（保持 shape）
     这样 `conditional` 才有可解释的差异。
   归属建议：
   - 放在 `method_config`：因为它只在 “需要 CFG 的算法” 中有意义；finetune 等方法不应被迫理解 uncond 语义。
   - 但执行必然落在 adapter（如何 zero_action/zero_image 是模型相关的），因此 adapter 需要提供一个 operation
     来承接该语义（例如“构造 uncond conditioning variant”）。

   **新增 TODO（需要实现）**
   - [ ] 为 wangame + DMD2 引入 `method_config.uncond_mode`
     - DMD2Method：读取该字段，并在 teacher CFG 时把 `conditional=False`
       映射到对应的 “uncond variant”
     - WanGameAdapter：提供可解释的 uncond 变体构造（避免硬编码 DMD2 名词），例如：
       - `conditional=False` 时按 `uncond_mode` 将 action/image conditioning 置零
       - 或提供一个更显式的 operation（如 `build_conditioning_variant(...)`）

2) **train_action_only / action_warmup_steps（细粒度 trainable）**
   legacy `wangame_training_pipeline.py` 支持更细粒度的训练策略：
   - `train_action_only`：冻结 base DiT，只训练 action 相关参数（pattern-based）
   - `action_warmup_steps`：前 N 步把 action 参数 `requires_grad=False`，之后再打开

   重要补充：这两个 knob 在 FastVideo 的 `TrainingArgs` 里已经存在（`fastvideo/fastvideo_args.py`），
   也就是说 **YAML 的 `training:` 段本身就能表达**：
   - `training.train_action_only: true`
   - `training.action_train_target: both|action_mlp|prope`
   - `training.action_warmup_steps: 1000`

   我们新框架目前的问题不是 “config 表达不了”，而是：
   - roles 只有 `trainable: bool`（整模块开关），无法表达“同一个 transformer 内只训练某些 param”
   - warmup 属于 step-time policy，应该由 method（或 method_config）驱动，而不是 loader 一次性决定

   **关于你提的方案：roles 允许自由 dict（而非完全结构化）**
   - 优点：wangame 这类 family 很容易出现 role-specific knobs（比如 per-role train patterns / cls_name / init 行为），
     不需要每加一个字段就改全局 schema。
   - 缺点：弱化静态校验，typo 会变成 runtime 才爆；文档/IDE 提示也会更差。

   **折中建议（更可控）：**
   - roles 仍解析核心字段（`family/path/trainable`），但把未知字段原样保留到 `RoleSpec.extra`（或 `role_cfg_raw`），
     由 `models/wangame.py` 自己读取解释。
   - 这样既能满足“roles 自由 dict 扩展”，也不牺牲核心字段的错误提示质量。

3) **validation pipeline 的 ODE/SDE 差异**
   现状：FastVideo 已经把 wangame 分成了两个 inference pipeline：
   - ODE：`WanGameActionImageToVideoPipeline`（`fastvideo/pipelines/basic/wan/wangame_i2v_pipeline.py`）
   - SDE/DMD：`WanGameCausalDMDPipeline`（`fastvideo/pipelines/basic/wan/wangame_causal_dmd_pipeline.py`）

   对新框架而言，最小可用的做法是：
   - `validators/wangame.py` 根据 `ValidationRequest.sampler_kind` 选择 pipeline class

   你提的“一个 pipeline 支持多 sampler（ode/sde）”是好方向（我们对 Wan 已经这么做过）：
   - 优点：减少 pipeline 分叉/重复逻辑，validation 不容易“走错范式”
   - 代价：需要对 wangame pipeline 做更侵入式的重构（引入 `pipelines/samplers/`，并把 denoising loop 抽出来）

   **建议顺序：**
   - 接入阶段先做 validator 选 pipeline（改动小、风险低）
   - 稳定后再把 wangame pipeline 也升级为 `sampler_kind` 可切换（更优雅，但属于额外工程）

---

## 7) 验收（我们如何确认接入是对的）

- finetune：
  - e2e 跑通（2~10 steps）+ 能输出 validation 视频
  - 与 legacy finetune 的 step0/stepN 视觉趋势大体一致（不追求完全 bitwise）
- distill（若做第二档）：
  - e2e 跑通（few-step）
  - validation 选择 `sde` 时，视觉应接近 legacy DMD pipeline 的 sampling 形态

---

## 8) 目前遗留 / 下一步（WanGame 接入方向）

- [ ] DMD2 on wangame 的 `uncond` 语义（`method_config.uncond_mode`）
- [ ] action-only / warmup（把 legacy 的 `train_action_only / action_warmup_steps`
  接到新框架：归属在 method/role policy 而非 model plugin）
- [ ] 若需要减少 ODE/SDE pipeline 分叉：将 wangame inference pipeline 也升级为
  `sampler_kind` 可切换（侵入式更强，建议放更后面的 phase）
