# 重构讨论：把复杂度“移到 config” vs 保持“清晰分层”

这份文档记录 FastVideo distillation 重构（Phase 2 → 3+）过程中，我们对
架构抽象与配置复杂度之间取舍的讨论，以及对 Phase 4+ 的可能方向。

核心矛盾：

- **方案 A（配置驱动加载 / 更少概念）**
  - 在 YAML 里显式写清模型加载细节（VAE、text encoder 等的路径/开关）。
  - `Method` 直接调用共享的 `utils/` 来完成加载与组装。
  - 希望减少“概念数量”（Family / Adapter / Bundle / Validator / …）。

- **方案 B（清晰分层 / 显式边界）**
  - `Method` 只关注算法与训练接口（algorithm-only）。
  - 模型/管线差异由 `Adapter`（operation-centric）吸收；构建期组装由
    `Family/Builder` 处理。
  - 配置保持结构化并可校验（schema + validation）。

本文档不是最终结论，主要记录：
1) 共识是什么；
2) 分歧是什么；
3) 实际 trade-off；
4) 收敛路径（如何减少概念但不破坏边界）。

---

## 术语表（当前 FastVideo 语义）

- **Role**：配置里用于索引参与者的字符串 key，例如 `student`、`teacher`、
  `critic`……（不设“高低贵贱”，只是 key）。
- **RoleHandle**：每个 role 对应的句柄，包含 `modules/optimizers/schedulers`
  以及类似 `trainable` 的标志。
- **ModelBundle**：持有所有 `RoleHandle` 的容器（未来可能改名）。
- **Adapter**：训练原语的 operation-centric 接口（如 `prepare_batch`、
  `predict_noise/x0`、`add_noise`、`backward` 等）。它不应包含算法策略。
- **Family**：构建期工厂：根据 config 组装 bundle + adapter + validator。
- **Validator**：训练期 validation 的采样/记录层，由 method 通过
  `ValidationRequest` 提供关键参数（steps/sampler/guidance/…）。

---

## 共同目标（共识）

1) **消除隐式耦合**
   - 避免“validation 的 side-effect 初始化训练状态”这种隐藏依赖。
   - 避免算法名（例如 “dmd”）泄漏进 adapter / validator。

2) **可扩展**
   - 新增 method 不应导致入口文件/组合爆炸。
   - 新增 role 不应导致“每个 role 一个函数”的 API 爆炸。

3) **职责清晰**
   - 算法决策归 `Method`。
   - 模型/管线的工程差异归 `Adapter` / 构建期组装。

4) **配置表达力强，但要安全**
   - 配置需要能表达当前 distillation，也要能表达未来训练语义（例如
     finetune = 只有 student 的一种“特殊 distill”）。

---

## 核心分歧

### A) “把抽象复杂度移到 config”

直觉是：
- 如果 config 能声明 `vae_path`、`text_encoder_path` 等，那么代码可以更
  简单：少一些抽象层、少一些类。
- Loading/组装细节可以抽到 `utils/`，由 method 直接调用。

### B) “配置驱动加载仍需要边界”

反驳点：
- 即使 YAML 写清楚了 *路径*，加载/运行的 *语义* 更准确地说是 **pipeline contract 相关**
  （也就近似“模型家族相关”）：
  - 有哪些子模块（`transformer` / `vae` / `text_encoder` / …）；
  - latent 的归一化/布局；
  - attention metadata（VSA/VMoBA）如何构建；
  - conditioning 的结构与缓存（text dict 的形状、neg/uncond 的初始化与
    broadcast）；
  - dtype/offload/并行切分（tp/sp/hsdp）的要求。

如果 `Method` 调用 `utils.load_vae(...)`，而 `utils` 内部其实知道
“Wan 的 VAE 怎么 load、SDXL 的 VAE 怎么 load”，那么 **method 仍然被污染**
（哪怕是间接污染）。我们只是把耦合从“文件 A”搬到了“文件 utils”，并没有
真正消除耦合。

---

## 讨论：能否做一个“完全通用”的 config → load → 调度？

先澄清“通用”到底指什么：

- **通用的模块加载机制**：可以统一（FastVideo 大多数 pipeline 的确会走
  `PipelineComponentLoader.load_module()` 这条路径）。
- **通用的 pipeline contract（需要哪些模块 + 如何初始化 + 如何采样/训练）**：
  **很难做到完全通用**，通常只能做到“框架通用 + 插件/实现分家族”。

为什么“contract”无法完全靠 config 自动推导？因为差异不仅在 *加载路径*，还在：

1) **需要加载哪些模块本身就不同**
   - Wan：`text_encoder/tokenizer/vae/transformer/scheduler`
   - SD3.5：`text_encoder(1/2/3) + tokenizer(1/2/3)`（三套编码器）
   - Hunyuan：`text_encoder_2/tokenizer_2`（两套）
   - LTX2：额外的 `audio_vae`、`vocoder`
   - Cosmos：额外的 `safety_checker`

2) **即使模块名相同，初始化/替换策略也不同**
   - 有些 pipeline 会在 `initialize_pipeline()` 里“重建/替换” scheduler，
     而不是按权重目录加载（例如 Cosmos 使用 `FlowMatchEulerDiscreteScheduler`，
     TurboDiffusion 使用 `RCMScheduler`）。
   - Wan 也会基于 `sampler_kind` 构建 scheduler，并改变 stage graph（ODE/SDE）。

3) **部分 pipeline 需要特殊的加载/回退逻辑**
   - LTX2 的 tokenizer 目录可能不存在，需要从 `text_encoder/gemma` 回退。
   - StepVideo 的 text encoder/clip 不是走 `model_index.json` 加载，而是在
     `initialize_pipeline()` 里手工构建，并加载自定义 `.so`。

4) **stage graph 差异是真实存在的**
   - refine pipeline（LongCat）会插入 refine init / timestep override 等 stage；
   - audio pipeline（LTX2）会有 audio decode stage；
   - 不同 pipeline 对 conditioning/timestep/latent 准备的处理逻辑不一致。

因此结论不是“完全无法抽象出通用框架”，而是：

- 我们可以抽象出 **通用框架**（统一入口、统一 YAML schema、统一 Trainer、统一
  Method 接口、统一 Adapter primitives、统一 Validator request/日志框架）。
- 但要做到“**只靠一份 dict config + 一个通用 utils** 就能自动适配所有模型家族”，
  很容易被上述 contract 差异击穿；最终要么：
  - 在 utils 里堆满 if/else（隐式 family），要么
  - 让 method 变成“懂所有 pipeline 语义的上帝对象”（method 污染）。

这也是为什么我们目前更倾向于保留“家族层”的存在（哪怕未来换个名字）：
它是“contract 插件层”，负责把 config 映射到 **正确的 pipeline/adapter/runtime 组装**，
把 method 留在算法层。

---

## “配置驱动加载”擅长什么、不擅长什么

配置擅长：
- **选择声明**：用哪个权重、哪些可选模块要加载、从哪里加载。
- **策略 knob**：例如 `sampler_kind`、`flow_shift`、validation steps、
  update interval 等。
- **可复现**：把 run 的 YAML 原样上传到 W&B（后续复盘非常方便）。

配置不擅长：
- **跨 role 的一致性约束**（例如共享 VAE、latent 空间兼容、scheduler 类型一致）。
- **防止“看似能跑但悄悄不一致”**（dtype/offload 不一致、negative embedding
  布局不一致、VSA metadata 不一致等）。

这些一致性通常需要：
- 单一组装点（builder/family），和/或
- 一个共享的 context（由 role-bound view 使用）。

---

## 为什么现在还需要 Family/Builder（务实角度）

即使只让 student 初始化 preprocessors，我们仍然需要一个地方来：
- 为所有 roles 加载 `RoleHandle.modules`（student/teacher/critic/…）；
- 决定哪些资源共享、哪些 per-role；
- 初始化共享缓存（neg/uncond conditioning 等）；
- 按 run config 构建 validator（采样 pipeline）并保持一致；
- 只在 rank0 创建 tracker / run 元信息。

如果移除 Family，工程上往往会以另外一种形式“复活”同样的东西：
- `SharedContext`，
- `RuntimeFactory`，
- 或“method 构造函数里做 assembly”。

因此问题变成：我们到底想要 **显式的组装层**，还是 **隐式的组装层**？
通常显式层更容易 review、更容易写测试、更容易维护 invariant。

---

## 建议的收敛方案（减少概念，但不把语义推给 method）

我们可以通过以下方式减少“概念感”，同时保持边界：

1) **重命名 / 重塑概念**
   - 例如 `ModelBundle` → `RoleManager`（语义更直观）。
   - `Family` → `RuntimeFactory`（如果“Family”这个词更难理解）。

2) **让 method “看起来像持有多个 adapter”，但底层共享 context**
   - 保留一份共享 adapter + context。
   - 增加轻量 per-role view：
     - `adapter.bind(handle)` → `RoleAdapterView`
     - method 持有 `student_api` / `teacher_api` / …（view），而不是 raw adapter。
   - 既保留共享资源的一致性，又贴近心智模型：
     “method 与 role-specific API 交互”。

3) **配置保持结构化，但允许受控扩展**
   - 保持顶层 schema（例如 `recipe/models/training/pipeline_config/method_config`）。
   - 允许在 `method_config`（或未来的 `family_config`）里放自由 dict，
     但解析与校验集中在一个入口，避免 dict 在全栈到处传。

4) **infra ownership 下沉到 trainer/entrypoint**
   - tracker 应由 trainer/runtime 持有（或 entrypoint 创建），而不是由 family
     的实现细节决定。

---

## 下一轮需要明确的问题（可作为下一次讨论议题）

1) “load preprocessors” 的 policy 应该放在哪里？
   - Family（assembly） vs Adapter（context） vs Method（algorithm）。

2) 如果未来支持跨 family distill（Wan teacher → SDXL student），如何保证
   “共享 VAE/latent 空间一致”的 invariant？

3) 当前 adapter API 哪些部分仍然过于 role-centric？
   - 目标：尽量 operation-centric primitives。

4) 是否需要一个显式的 `SharedContext` 类型？
   - 还是继续把共享状态作为 adapter 的内部实现细节。
