# Phase: Causal（对齐 FastGen 的 causal / cache 设计）

> 目标：让 **causal/streaming** 能力像 FastGen 一样成为“可选增强”，只落在 causal 变体上；
> 同时把 **预处理器（preprocessors：VAE / text encoder / image encoder / …）** 的管理收敛到
> **student**，避免 teacher/critic 重复加载与显存浪费，并减少跨 role 的耦合。

## 1. 背景与动机

在 FastVideo 的 distillation 框架里，我们会遇到典型组合：

- teacher：bidirectional（质量高、但不适合 streaming rollout）
- student：causal（需要 KV-cache / streaming rollout）

如果把 KV-cache 相关的接口（`kv_cache` / `crossattn_cache` / `current_start` / …）直接塞进
通用的 `predict_noise/predict_x0`，会导致：

- **所有模型**都被迫兼容 causal 的 cache 语义（污染接口）；
- method 代码不得不理解具体 cache 结构（污染算法层）；
- 加一个新模型/新任务时，很容易“接口爆炸”。

FastGen 的经验是：

- **只有 causal network** 需要并实现 cache（`store_kv=True` + internal caches）；
- **预处理器只在 student 侧管理/按需初始化**（VAE/text encoder 主要用于数据编码与可视化 decode），
  teacher forward 通常直接吃 latent，不需要再加载一份 VAE。

本 phase 将这两条经验收敛到我们的 distillation 框架中。

## 2. 关键原则

### 2.1 只有 student 管 preprocessors

定义：

- preprocessors = `vae` / `text_encoder` / `image_encoder` / （未来可能的）action encoder 等。

策略：

- **只在 student 对应的 model plugin 中初始化/持有** preprocessors（必要时 lazy init）。
- teacher/critic role **不再重复加载** preprocessors。
- 训练/蒸馏计算以 latent 为主，validation/wandb 可视化 decode 时再用 student 的 VAE。

收益：

- 显存更友好（尤其是多 role / 大模型 teacher）。
- 语义更清晰：teacher 的职责是提供 score/target，而不是承担数据编解码。

### 2.2 causal 能力只落在 *_causal 变体

将“streaming rollout / KV-cache / blockwise causal mask / context noise”等能力视为
**causal 变体的 runtime primitive**，而不是所有模型的通用能力。

对应的 method（例如 self-forcing/DFSFT/causal validator）如果需要 cache，
应当只依赖 causal 变体提供的可选接口（或不透明 `cache_state`），而非强制所有模型支持。

#### CausalModelBase：不污染 ModelBase 的接口

为了避免把 `kv_cache/crossattn_cache/...` 这类实现细节塞进通用的
`ModelBase.predict_noise/predict_x0`，我们引入一个 causal 专属基类：

- `ModelBase`：保持通用 primitives（`predict_noise/predict_x0/add_noise/...`），**不出现 cache**。
- `CausalModelBase(ModelBase)`：仅定义 causal/streaming 所需的最小契约，形态参考 FastGen。

FastGen 的经验是：cache 是 **causal network 内部可变状态**，method 不传递 cache 张量本体。
method 只通过 “清空缓存 + 运行 forward（可选 store_kv）” 来驱动 streaming rollout。

因此在我们的 `CausalModelBase` 中更优雅的 API 组合是：

- `clear_caches(handle, cache_tag=...)`：开始新 rollout / 新 validation request 前清空。
- `predict_noise_streaming(..., cache_tag=..., store_kv=..., cur_start_frame=...)`：
  在 causal 路径中通过 `store_kv` 控制是否写入历史 cache。

而不是设计成：

- `predict_noise_kvcache(kv_cache=...)`：这会迫使 method 管理 cache 生命周期与结构（污染算法层），
  且最终仍需要额外的 reset/init API。

## 3. 目录层级拆分（models）

将 `fastvideo/distillation/models/*` 拆成更清晰的层级结构，按 “family + variant” 组织：

```text
fastvideo/distillation/models/
  base.py                 # ModelBase：方法无关的 operation-centric primitives
  wan/
    __init__.py
    wan.py                # Wan（T2V）bidirectional primitives
  wangame/
    __init__.py
    wangame.py            # WanGame bidirectional primitives（不含 cache）
    wangame_causal.py     # WanGame causal primitives（包含 streaming/cache）
```

注：

- `wangame_causal.py` 的职责是提供 “causal 专属的 predict/rollout primitive”，例如：
  - `clear_caches()`（或 `clear_*`）等 cache 生命周期管理；
  - `store_kv` / cache reset / cache update 的封装（cache **由 causal 变体内部持有**）；
  - streaming rollout 所需的 block/chunk 约束。
- `wangame.py` 保持纯 bidirectional（不引入 cache 语义）。

## 4. 运行时组织（role → 独立网络实例）

对齐 FastGen：**每个 role 持有自己的 transformer 实例**（student/teacher/critic 可以是不同类/不同权重），
但 preprocessors 只由 student 管理。

落地方式（保持我们现有 RoleManager/RoleHandle 架构）：

- `RoleHandle.modules["transformer"]`：每个 role 独立的 denoiser/transformer
- `ModelBase`（或 family-specific model plugin）：
  - 统一持有 student preprocessors（VAE/text encoder…）
  - 统一实现 batch→forward primitives 的规范化
  - 根据 role 的 variant 选择对应的 bidi/causal 实现路径（必要时由 `roles.<role>.extra.variant` 决定）

## 5. 风险点与验证

### 5.1 latent 语义一致性

该设计隐含前提是：teacher/student/critic 共享同一 latent 语义（通常意味着同一 VAE 语义）。
如果未来要支持跨 family（例如 teacher=SDXL, student=Wan），需要额外的对齐层（暂不在本 phase 处理）。

### 5.2 cache 生命周期与 no-grad 语义

causal cache 的更新通常应在 `torch.no_grad()` 下进行，以避免历史 cache 引入梯度/爆显存。
需要在实现阶段明确：

- cache 初始化/重置的时机（每个 rollout / 每个 validation request）
- cache 的 dtype/device 与 FSDP/activation checkpoint 的交互

### 5.3 验证策略

最小验证集合：

- bidirectional Wan/WanGame：原有 DMD2/finetune 训练与 validator 不回归；
- causal WanGame：streaming rollout 能跑通（block mask 正确、cache 正确更新）；
- 多 role：teacher=bidi + student=causal 能正确构建并 forward。

## 6. TODO（实施时的文件清单）

- [ ] `fastvideo/distillation/models/` 目录结构调整（新增子目录、移动文件、更新 imports）
- [ ] preprocessors 收敛到 student：移除 teacher/critic 侧的 preprocessor 初始化与依赖
- [ ] `wangame_causal.py`：封装 cache/streaming primitives（仅 causal 需要）
- [ ] 更新 `dispatch.py` / `register_model` 的 import 路径（保持注册行为不变）
- [ ] 更新必要的 doc（只更新本 phase 相关文档）
