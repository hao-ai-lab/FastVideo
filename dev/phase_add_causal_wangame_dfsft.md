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
- [ ] **Validation**：沿用现有 validator，能够用 `validation_sampling_steps=40`
  做验证（ODE 或 SDE 均可，默认用 ODE）

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
  context_noise: 0.0
```

说明：
- `chunk_size`：决定 `t_inhom` 的 block 划分（FastGen 也用 chunk_size）。
  - 对 Wang/Wangame，建议默认 3（与 `num_frames_per_block` 一致）。
- `context_noise`：未来如果我们实现“prefill cache 前对历史 x0 加噪”，
  这个值将用于控制历史噪声强度。

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
  - `training.validation_sampling_steps: "40"`

- [ ] `examples/distillation/wangame/dfsft-temp.sh`（新增）
  - 跟现在 `run.sh` 一样只负责 export CONFIG + torchrun

---

## 5. 验收标准（Definition of Done）

- [ ] DFSFT 端到端可跑（不需要 teacher/critic）
- [ ] step0 validation 能出视频，不 crash
- [ ] 训练若干步后，validation 质量有可见提升
- [ ] 同一份 wangame checkpoint：bidirectional finetune 和 causal dfsft
  都能启动（若 causal 需要不同 ckpt，要明确写在配置/README）

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

---

## 7. FastGen 对照（便于后续实现 CausVid）

- DFSFT (SFT + diffusion forcing)：
  - `fastgen/methods/fine_tuning/sft.py::CausalSFTModel`
  - `fastgen/networks/noise_schedule.py::sample_t_inhom_sft`

- Diffusion-forcing distillation（未来）：
  - `fastgen/methods/distribution_matching/causvid.py::CausVidModel`

