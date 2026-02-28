# Phase: Add Causal WanGame + Diffusion-Forcing SFT (DF-SFT)

> 目标：在现有 distillation framework 上，新增一种 **causal Wangame** 的
> supervised finetuning 方法（DFSFT），用于把 Wangame 从「双向 / bidirectional」
> 的训练范式，迁移到「因果 / causal」范式。
>
> 参考实现：FastGen 的 `CausalSFTModel`（diffusion forcing for SFT）。

---

## 0. 背景与动机

我们已经把 Wangame 的 `finetune` 与 `dmd2` 跑通。
下一步要做 **bidirectional -> causal**。

我建议先落地一个 **Diffusion-Forcing SFT**（DFSFT）baseline：

- 仅训练 `student`（SFT/DSM loss，和 FastGen 对齐）；
- 使用 **block-wise inhomogeneous timesteps**（`t_inhom`，按 chunk 采样），
  让 causal student 在训练时就面对“历史上下文不一定是干净的”的分布；
- 不引入 teacher/critic 依赖，降低第一版风险。

> 这不是“few-step distill”。它是“训练一个 causal 的基础模型”。
> 如果后面要把 causal Wangame distill 成 4/8 steps，再做 CausVid/DMD2
> diffusion-forcing distillation 更合适。

---

## 1. 本阶段产物（Deliverables）

- [ ] **Model 侧**：Wangame 支持 `causal` 变体（通过 role 的 `extra` 参数触发）
- [ ] **Method 侧**：新增 `dfsft`（diffusion-forcing SFT）方法
- [ ] **Examples**：新增一份 DFSFT 的 YAML + temp.sh（端到端可跑）
- [ ] **Validation（关键变更）**：支持“**真正的 causal rollout**”验证，并支持
  **context noise**：
  - causal rollout 指 **streaming / chunk-wise** 生成（KV-cache + block processing），
    而不是“用 causal transformer 但仍一次性 full-video forward”；
  - **目标是 ODE-style 的 causal streaming**（用 `scheduler.step(...)` 做数值推进），
    而不是 DMD/SDE-style 的 `pred_x0 -> add_noise(next_t, eps)` rollout；
  - 若要同时保留 SDE-style（用于对照/legacy 对齐），应当由 config 显式选择，
    而不是在 validator 里隐式“偷换语义”。

---

## 2. 配置语义（Config）

### 2.1 Causal variant（Role extra）

不新增新的 family（避免 `wangame_causal` 这种对外语义膨胀）。
仍然是：

```yaml
roles:
  student:
    family: wangame
    path: ...
    trainable: true
    # extra fields (RoleSpec.extra)
    variant: causal
    # （可选）更细粒度的 causal invariant：用于表达‘是哪一种因果约束/训练范式’
    # 例如：strict / block / sliding_window / bidirectional_train_causal_eval ...
    causal_invariant: block
```

- `variant: causal` 由 `models/wangame` 插件解释。
- 未来如果需要更细粒度，可扩展为：
  - `variant: causal|bidirectional`
  - `causal: true|false`
  - `num_frames_per_block` / `sliding_window_num_frames`（可选）

### 2.2 DFSFT method config（与 FastGen 对齐）

推荐：把 DFSFT 的关键 knob 放到 `method_config`：

```yaml
recipe:
  family: wangame
  method: dfsft

method_config:
  # diffusion-forcing (SFT) 核心：按 chunk 采样 inhomogeneous t
  chunk_size: 3

  # t sampling（可以复用我们已有的 min/max ratio 语义；最终落到 [0,1000]）
  min_timestep_ratio: 0.02
  max_timestep_ratio: 0.98

  # 可选：更接近“history noisy cache”的效果（第一版可以先不做）
  context_noise: 0
```

说明：
- `chunk_size`：决定 `t_inhom` 的 block 划分（FastGen 也用 chunk_size）。
  - 对 Wang/Wangame，建议默认 3（与 `num_frames_per_block` 一致）。
- `context_noise`：**context noise timestep**（整型）。用于 causal rollout 时
  “更新 KV cache 的上下文噪声水平”（见第 4 节 Validation 设计）。

### 2.3 Validation config（真正的 causal rollout + context noise）

我们需要区分两种 validation 语义：

1) **full-video rollout**（非 streaming）：`WanGameActionImageToVideoPipeline.forward(...)`
2) **streaming causal rollout**（推荐用于 causal student 的验证）：`WanGameCausalDMDPipeline.streaming_*`

建议把 validation 的配置拆成两类：

- **run/trainer 级（什么时候验证、用什么 validation dataset）**：放在 `training:` 下；
- **method 级（怎么采样、走什么 rollout 语义）**：放在 `training.validation:` 下（method 读取）。

这样 validator 不需要 “从 training_args 猜 method 语义”，而 method 也不会越权去管理 dataloader。

建议把选择权交给 method（或 method_config），让 validator 只负责执行：

```yaml
training:
  validation:
    # full | streaming_causal
    rollout_mode: streaming_causal

    # 对 full rollout：ode/sde 选择 sampling loop
    # 对 streaming_causal：同样允许 ode/sde，但这是一个“明确的语义选择”
    sampler_kind: ode

    # 采样步数（对 ODE pipeline 即 num_inference_steps）
    sampling_steps: [40]

    # SDE/DMD: 需要显式 step list（few-step schedule）
    # sampling_timesteps: [1000,750,500,250]
    # warp_denoising_step: true

    # causal cache 的 context noise（timestep）
    context_noise: 0
```

> 备注：
> - 现阶段的 `WanGameCausalDMDPipeline / MatrixGameCausalDenoisingStage` 是 DMD/SDE-style；
>   要实现 **ODE-style streaming**，需要新增一个 causal ODE denoising stage（见第 4 节）。
> - 我们仍可以保留 SDE-style streaming（用于对照/legacy 对齐），但必须由 config 显式选择，
>   避免 “training/validation 语义混用导致 reviewer 困惑”。

### 2.4 default_pipeline_config：ODE solver 选择（可选）

我们目前的 `pipeline_config.sampler_kind=ode` 默认会选择 `FlowUniPCMultistepScheduler`
（见 `fastvideo/pipelines/samplers/wan.py`）。为了做对照实验/调试，建议增加一个可选字段：

```yaml
default_pipeline_config:
  sampler_kind: ode
  ode_solver: unipc   # unipc | euler
```

约束（推荐强校验）：
- `sampler_kind=sde` 时不允许 `ode_solver=unipc`（因为 SDE/DMD-style rollout 需要
  `add_noise(next_t, eps)`；UniPC 的 `add_noise` 对 “任意 timestep 值” 不鲁棒）。
- `ode_solver=euler` 时应强制 deterministic（`stochastic_sampling=false`），否则就变成
  “SDE-like Euler”。

---

## 3. 训练逻辑（DFSFT 的算法定义）

目标：对齐 FastGen `CausalSFTModel`（`fastgen/methods/fine_tuning/sft.py`）。

核心步骤（单 step）：

1) 取真实数据 `x0`（video latents）。
2) 采样 `eps_inhom ~ N(0, I)`。
3) 采样 `t_inhom`：形状 **[B, T_lat]**，按 chunk/block-wise 采样，chunk 内
   timestep 相同。
4) 前向扩散：`x_t = add_noise(x0, eps_inhom, t_inhom)`
5) 学生预测：`pred = student(x_t, t_inhom, cond)`（预测 noise/x0/v，取决于
   adapter/model 的 pred_type）
6) DSM loss：对齐噪声调度器语义（最简单是 MSE(pred_eps, eps_inhom)）。

关键点：
- DFSFT 不需要 teacher。
- “diffusion forcing”体现在 `t_inhom`（按 chunk 的独立噪声水平），而不是
  直接对 KV tensor 加噪。

---

## 4. 代码改动清单（按文件）

### 4.1 models（Wangame causal variant）

- [ ] `fastvideo/distillation/models/wangame.py`
  - 读取 `role_spec.extra.get("variant")`（或 `causal: true`）
  - 当 `variant == "causal"`：加载 transformer 时覆盖 cls 为
    `CausalWanGameTransformer3DModel`（FastVideo 已存在该类：
    `fastvideo/models/dits/wangame/causal_model.py`）
  - 目标：**同一份 ckpt 既可作为 bidirectional student，也可作为 causal
    student 初始化**（如果 state_dict 不兼容，需要记录为风险点并加 fallback）。

> 备注：如果实现细节需要拆文件，可以内部新增
> `fastvideo/distillation/models/wangame/causal.py`，但对外 family
> 仍然是 `wangame`。

### 4.2 methods（新增 dfsft）

- [ ] `fastvideo/distillation/methods/fine_tuning/dfsft.py`（新增）
  - `@register_method("dfsft")`
  - 仅依赖 `roles.student`
  - `single_train_step()`：实现第 3 节 DFSFT
  - 复用现有 finetune 的 optimizer/lr scheduler wiring

- [ ] `fastvideo/distillation/methods/__init__.py`
  - 暴露/导入新方法（取决于我们当前 registry/dispatch 的约定）

- [ ] （可能需要）`fastvideo/distillation/adapters/wangame.py`
  - 确认 `predict_noise/add_noise` 支持 `timestep` 为 **[B, T_lat]**
  - 如果当前只支持 [B]，需要扩展并加形状检查。

### 4.3 examples（端到端验证）

- [ ] `examples/distillation/wangame/finetune_wangame2.1_i2v_1.3B_dfsft_causal.yaml`
  - `roles.student.variant: causal`
  - `recipe.method: dfsft`
  - `training.validation.dataset_file / training.validation.every_steps`
  - `training.validation.sampling_steps: [40]`

- [ ] `examples/distillation/wangame/dfsft-temp.sh`（新增）
  - 跟现在 `run.sh` 一样只负责 export CONFIG + torchrun

### 4.4 validation（真正的 causal rollout）

> 这是本阶段新增的关键设计点：**不要**默认复用“full-video validator”来验证 causal 模型。

- [ ] `fastvideo/distillation/validators/wangame.py`（或新增 `wangame_causal.py`）
  - 支持 `rollout_mode: streaming_causal`：
    - pipeline：**需要 ODE-style 的 causal streaming pipeline**
      - 方案 A（推荐）：扩展/新增 `WanGameCausalPipeline`（同一条 pipeline），内部按
        `sampler_kind={ode|sde}` 选择不同 denoising stage
      - 方案 B（过渡）：保留 `WanGameCausalDMDPipeline`（仅 SDE），但这不满足本阶段目标
  - ODE-style streaming 的调用方式（与现有 streaming API 对齐）：
    1) `pipeline.streaming_reset(batch, fastvideo_args)`
    2) 循环 `pipeline.streaming_step(...)` 直到生成完成
    3) 聚合每个 chunk 的 frames，拼成完整 video 再落盘/上报 tracker
  - 支持 `context_noise`（对齐 legacy self-forcing 语义）：
    - cache update 前对 context latent 做一次显式 `scheduler.add_noise(x0, eps, t_context)`
      再 forward 更新 KV cache（避免 “只改 timestep embedding，效果像没开”）

- [ ] `fastvideo/pipelines/stages/`（新增 causal ODE denoising stage）
  - `MatrixGameCausalOdeDenoisingStage`（或同等命名）
    - block/chunk 框架与 `MatrixGameCausalDenoisingStage` 一致（KV-cache + action）
    - block 内 loop 使用 `scheduler.step(...)`（ODE loop）
    - **每个 block 必须 reset solver state**（见风险点：UniPC 多步历史不能跨 block 泄漏）

- [ ] `fastvideo/distillation/validators/base.py`
  - 若需要：扩展 `ValidationRequest` 以携带
    - `rollout_mode`
    - `context_noise`
    - `sampling_timesteps`（已有）
  - 目标：把“验证用什么 pipeline/rollout”的决策交给 method。

---

## 5. 验收标准（Definition of Done）

- [ ] DFSFT 端到端可跑（不需要 teacher/critic）
- [ ] step0 validation 能出视频，不 crash
- [ ] 训练若干步后，validation 质量有可见提升
- [ ] 同一份 wangame checkpoint：bidirectional finetune 和 causal dfsft
  都能启动（若 causal 需要不同 ckpt，要明确写在配置/README）
- [ ] 支持用 streaming causal rollout 做验证（并能开启/关闭 context noise）

---

## 6. 风险点 / 需要提前确认的问题

1) **权重兼容**：`CausalWanGameTransformer3DModel` 是否能直接 load
   bidirectional wangame 的 transformer 权重。
   - 如果不能：需要一个 conversion 逻辑（或要求 user 提供 causal init ckpt）。

2) **t_inhom 的 shape 语义**：
   - Wangame transformer 是否真正支持 [B, T_lat] timesteps；
   - scheduler.add_noise 是否支持 per-frame timesteps（不支持就需要 reshape
     或 per-frame add_noise）。

3) **chunk_size 与模型结构对齐**：
   - DFSFT 的 chunk_size 是否必须等于模型的 `num_frames_per_block`；
   - 如果用户配错，建议直接 error。

4) **“40-step causal” 的含义**：
   - DFSFT 训练的是基础模型；推理时可以设 `num_inference_steps=40`。
   - 但“few-step（4/8）”仍需要 distillation（DMD2/CM/CausVid）。

5) **“真正 causal rollout” vs “full-video rollout”**（重要决策点）：
   - 如果我们只用 `WanGameActionImageToVideoPipeline.forward(...)` 做验证，
     这并不能覆盖 deployment 的 streaming/KV-cache 语义；
   - 但 streaming rollout 目前依赖 `WanGameCausalDMDPipeline`（DMD/SDE-style step list），
     因此需要明确：
     - DFSFT baseline 先用 streaming + step list 验证（更贴近 causal 部署）；
     - 或者额外实现一个 “streaming ODE” sampler（更大工程，建议后置）。

6) **step list / context noise 的配置入口**（重要决策点）：
   - 现有 streaming pipeline 读取 `pipeline_config.dmd_denoising_steps / warp_denoising_step / context_noise`；
   - 我们的新框架更希望把 few-step schedule/rollout knobs 放到 `method_config` 或 `training.validation`；
   - 因此需要一个明确策略：
     - 方案 A（最小改动）：validator 在构建 validation pipeline 时，把
       `training.validation.*` 写入 `args_copy.pipeline_config.*`（validation-only）；
     - 方案 B（更干净）：把 streaming pipeline 改为优先读 `ForwardBatch.sampling_timesteps`
       + `ValidationRequest.context_noise`，彻底摆脱 pipeline_config 依赖（工程量更大）。

7) **context noise 的“语义一致性”**（潜在坑）：
   - legacy self-forcing 里 context noise 是：
     `x0 -> add_noise(x0, eps, context_timestep)` 再更新 cache；
   - 现有 `MatrixGameCausalDenoisingStage._update_context_cache` 只把 timestep 传给 transformer，
     是否也应显式 add_noise（以匹配语义）需要确认，否则“开了 context noise 但效果像没开”。

8) **ODE solver 的 state reset（新增风险点，必须明确）**：
   - `FlowUniPCMultistepScheduler` 是 multistep solver，内部有历史状态；
   - 在 streaming causal rollout 中，**每个 block/chunk** 都应当像独立的 ODE 求解过程：
     - 要么每个 block 调用一次 `scheduler.set_timesteps(...)`（会清空 `_step_index` 和历史）
     - 要么为每个 block 构建新的 scheduler 实例
   - 否则会出现 solver 历史跨 block 泄漏，导致质量漂移且很难定位。

---

## 7. FastGen 对照（便于后续实现 CausVid）

- DFSFT (SFT + diffusion forcing)：
  - `fastgen/methods/fine_tuning/sft.py::CausalSFTModel`
  - `fastgen/networks/noise_schedule.py::sample_t_inhom_sft`

- Diffusion-forcing distillation（未来）：
  - `fastgen/methods/distribution_matching/causvid.py::CausVidModel`
