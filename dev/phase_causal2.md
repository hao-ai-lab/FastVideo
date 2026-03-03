# Phase: Causal2（更贴近 FastGen 的“student 管 shared components”语义）

> 目标：进一步把 “shared / expensive components（VAE / text encoder / image encoder / …）”
> 的生命周期与加载语义收敛到 **student**，并让一个 run 内部自然支持
> “student/teacher/critic 多模型（transformers）并存”，而不引入额外的中间概念。

## 1. 背景：FastGen 的语义是什么？

FastGen 在配置层面并不使用 `variant: causal` 这种字段，而是通过“选择不同的 net 配置类”
来表达“我要跑 causal 还是 bidirectional”：

- `config.model.net = CausalWan_1_3B_Config` → student 是 causal
- `config.model.teacher = Wan_1_3B_Config` → teacher 仍然可以是 bidirectional

它的核心思路可以概括成两点：

1) **student/net 是主模型（main network）**：很多“数据侧/推理侧的 shared 组件”
   会天然围绕它组织（例如 preprocessors、一些全局 shape 约束等）。
2) **teacher 只是额外的网络实例**：提供 target/score，不应该因为 teacher 存在就把
   preprocessors 再加载一份、再维护一份生命周期。

## 2. 我们想对齐的“语义”是什么？

### 2.1 只让 student 负责 shared components（preprocessors）

定义 shared components（示例）：

- VAE（latent encode/decode + latent norm）
- text encoder / tokenizer（如果训练/验证需要 text conditioning）
- image encoder（I2V 可能需要）
- 其它可能跨 role 复用、且体积/显存开销大的组件

原则：

- **一个 run 只加载一份 shared components**，默认由 `roles.student` 持有/提供。
- teacher/critic role **只加载它们各自的 transformer**（以及必要的轻量配置），不再重复加载 VAE 等。

收益：

- 显存与初始化成本显著降低（尤其 teacher 大模型时）。
- 更清晰的职责划分：teacher 是“提供 target 的网络”，不是“数据编解码提供者”。

### 2.2 一个 run 直接持有多个模型（多 role transformers）

对齐 FastGen：“一个训练 job 同时持有 student + teacher（+ critic/…）”。

在我们的框架里，最自然的落地是：

- `RoleManager` 仍然是“多 role modules”的容器（student/teacher/critic/...）。
- model plugin 负责：
  - 构建每个 role 的 transformer（以及 trainable 开关/activation checkpointing 等）；
  - 加载 shared components（只加载一次）；
  - 把 dataloader batch 规范化为 methods 可用的 forward primitives；
  - 提供 operation-centric primitives（`add_noise / predict_noise / predict_x0 / backward / ...`）。

也就是说：**“多模型并存”是运行时事实，不需要额外引入 ModelComponents 之类中间层**。

## 3. 配置（YAML）层面的约定

### 3.1 shared components 的来源：固定使用 student（从简）

从简规则（不做 fallback / 不做复杂合并）：

- shared components 的加载路径 = `roles.student.path`
  - 这也是 `training.model_path` 的默认来源（用于 pipeline registry / 组件加载等）。
- 如果没有 `roles.student`，则：
  - 对于需要 shared components 的 recipe：直接报错（强约束，避免 silent 行为）。

未来如果需要支持 “teacher/student family 不同”，再引入更复杂的机制（例如多套 shared components）。

### 3.2 causal / bidirectional 的选择（FastGen 风格）

我们可以同时支持两种表达方式（但优先推荐 FastGen 风格）：

- FastGen 风格（推荐）：通过 `recipe.family` 选择模型变体
  - `recipe.family: wangame` / `recipe.family: wangame_causal`
  - `wangame_causal` 默认所有 role 走 causal transformer（除非 role 显式声明 bidi）
- 兼容表达（可选）：`roles.<role>.variant: causal|bidirectional`
  - 用于 “student causal + teacher bidirectional” 等混合场景

注：即使使用 `recipe.family: wangame_causal`，仍建议保留 per-role override 的能力，
以覆盖 FastGen 常见组合（student causal + teacher bidi）。

## 4. 代码落地点（文件与职责）

以 wangame 为例：

```text
fastvideo/distillation/models/wangame/
  common.py            # role transformer 构建的共享逻辑（不涉及 preprocessors）
  wangame.py            # bidi 版本：加载 shared components + bidi primitives
  wangame_causal.py     # causal 版本：在 wangame.py 基础上增加 cache/streaming primitives
  __init__.py           # register_model("wangame") (and maybe "wangame_causal")
```

关键点：

- `common.py` 只负责 “按 role 构建 transformer handle”，不加载 VAE/text encoder 等。
- `wangame.py / wangame_causal.py` 只在 **一个地方**加载 shared components（来自 student path），并复用给所有 role。
- methods 永远通过 `model.predict_* (handle=RoleHandle, ...)` 这类 operation-centric API 调用网络；
  methods 不直接“知道/管理”VAE 等加载细节。

## 5. 风险与边界（明确不做）

- **跨 family shared components**：例如 teacher=SDXL, student=Wan。
  - 这会带来 latent 语义不一致、conditioning schema 不一致等问题。
  - 本 phase 不解决；遇到则应当在构建期直接 error，避免 silent mismatch。
- **让 config 变成无结构 dict 并把语义搬进 utils**：
  - “路径可配置”不等于“语义可抽象”；加载/调度的语义仍然高度 family 相关。
  - 仍坚持 model plugin 层吸收 family 差异，methods 保持算法纯净。

## 6. TODO（实施清单）

- [ ] 新增 `recipe.family: wangame_causal`（更 FastGen 风格），默认所有 role 为 causal
- [ ] 明确并 enforce：shared components 仅从 `roles.student.path` 加载
- [ ] 文档化：每个 model/method 文件开头声明本文件会读取的 config keys（降低阅读成本）

