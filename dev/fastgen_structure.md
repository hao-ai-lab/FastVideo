# FastGen 的 distillation 设计梳理（可借鉴点）

这份文档是我在本机 `~/alex/FastGen` 仓库里阅读 distillation 相关代码后的结构总结，
重点关注 “架构设计优点/可复用模式”，用于反哺 FastVideo 的 distill 重构。

> 结论先行：FastGen 把 distillation 的关键复杂度分摊到了
> `Trainer(基础设施)` / `Method(算法+多网络训练)` / `Network(架构+采样)` /
> `Dataset(数据与conditioning供给)` 四层，并用 config + callback + checkpoint
> 把它们解耦起来。

## 1. FastGen 的核心分层（强推荐）

FastGen 的高层调用链可以简化成：

```
train.py
  -> load config (python config + Hydra overrides)
  -> instantiate(config.model_class)  # "method" = FastGenModel 子类
  -> Trainer(config).run(model)
       -> (DDP/FSDP wrap) + callbacks + checkpointer
       -> dataloader + preprocess_data
       -> for iter:
            loss_map, outputs = model_ddp.single_train_step(data, iter)
            backward(total_loss)
            model.optimizers_schedulers_step(iter)
            model.optimizers_zero_grad(iter)
```

这套结构的关键点在于：

- **Trainer 永远只认识一个接口：`single_train_step()` + `total_loss`**
- “distillation 算法/多网络/多优化器/交替更新”等复杂逻辑都封装进 **method 类**
- **network 类只负责 forward / noise schedule /（可选）sample**，并提供少量 hook
- dataset 负责把 `real/condition/neg_condition` 等输入准备齐全（甚至可以预存为常量）

这种分层特别适合你想做的 “`models={teacher, student, critic,...}` + 算法解耦”。

## 2. 仓库组织方式（distill 相关）

FastGen 的 repo 结构（与 distill 相关部分）：

- `train.py`：入口，加载 config，实例化 `model_class`，调用 `Trainer.run`
- `fastgen/trainer.py`：通用训练循环（DDP/FSDP、grad accum、validate、save/load、callbacks）
- `fastgen/methods/`：**训练方法/算法**（distill 的主体逻辑在这里）
  - `methods/model.py`：`FastGenModel` 基类（多网络训练接口、precision、EMA、采样 loop 等）
  - `methods/distribution_matching/dmd2.py`：DMD2（student/teacher/fake_score + 可选 discriminator）
  - `methods/distribution_matching/causvid.py`：CausVid（因果 video distill，复用 DMD2 框架）
  - `methods/distribution_matching/self_forcing.py`：Self-Forcing（继承 CausVid/DMD2，仅改 student rollout）
  - 其它：CM/sCM/TCM/MeanFlow、KD、SFT 等（同一接口体系）
- `fastgen/networks/`：架构实现（EDM/SD/Wan/CogVideoX/Cosmos…）
  - `networks/network.py`：`FastGenNetwork` / `CausalFastGenNetwork` 抽象接口
- `fastgen/datasets/`：数据加载（webdataset + 预计算 latent/embedding）
  - `datasets/wds_dataloaders.py`：支持 `files_map/presets_map` 注入常量 conditioning（比如 neg prompt）
- `fastgen/configs/`：配置系统（方法 config、实验 config、网络 config、数据 config）
- `fastgen/utils/`：基础设施（instantiate、distributed ddp/fsdp、checkpointer、logging、autoresume）
- `fastgen/callbacks/`：回调系统（EMA、grad clip、wandb、profiler、param count…）

## 3. 配置系统：python config + LazyCall + Hydra override（优秀）

FastGen 采用：

- `attrs` 定义结构化 config（如 `BaseConfig/BaseModelConfig/BaseTrainerConfig`）
- `OmegaConf/DictConfig` 存储 LazyCall 结构并支持 object
- Hydra 的 override 语法做命令行参数覆盖：`python train.py --config=... - key=value`
- 训练启动时把 resolve 后的 config 序列化为 `config.yaml`，保证可复现

关键模式：

1) **三层 config 分离**

- `configs/config.py`：BaseConfig（训练通用字段）
- `configs/methods/config_*.py`：方法级默认参数（比如 DMD2 要 fake_score_optimizer）
- `configs/experiments/**/config_*.py`：具体实验（比如 WanT2V 的 input_shape、t_list、lr）

2) **LazyCall/instantiate（“延迟实例化”）**

- config 内用 `LazyCall` 记录 `_target_` + kwargs
- 真正创建对象统一走 `instantiate(cfg)`

价值：

- algorithm/network/dataloader/callback 全部可插拔
- method 和 trainer 都不需要硬编码具体类名

## 4. Trainer：完全算法无关的训练循环（非常干净）

`fastgen/trainer.py` 的设计要点：

- **只要求模型实现 `single_train_step(data, iteration)`**
- backward 统一对 `loss_map["total_loss"]` 做
- grad accumulation 通过：
  - DDP：`ddp_sync_grad(model_ddp, sync_grads)`
  - FSDP：`fsdp_sync_grad(model, sync_grads)`
- optimizer step / scheduler step 通过调用 model 的接口完成：
  - `model.optimizers_schedulers_step(iteration)`
  - `model.optimizers_zero_grad(iteration)`
- validate 可以复用 `single_train_step`（no_grad + autocast），并且支持
  `global_vars_val` 在一次 validation 中跑多个设置（例如限制 `MAX_VAL_STEPS`）

这个框架天然支持你想要的 “算法与模型解耦”：Trainer 永远不关心 roles。

## 5. Method（算法层）：用“对象”承载多网络与更新策略（关键借鉴点）

### 5.1 FastGenModel（统一训练接口 + 多网络容器）

`fastgen/methods/model.py::FastGenModel` 承担了大量“应放在 distill 框架层”的事：

- precision / AMP / FSDP precision 管理（`precision`, `precision_amp`, `precision_fsdp`）
- teacher 构建与冻结（`build_teacher()`）
- student 初始化（可从 teacher 或单独 ckpt）+ EMA 初始化（`_setup_ema()`）
- inference 的统一入口（`generator_fn` / `_student_sample_loop` / `sample`）
- checkpoint 需要的统一映射：
  - `model_dict` / `optimizer_dict` / `scheduler_dict`
  - `fsdp_dict`（决定哪些 module 要被 FSDP sharding；可选择把 teacher 放进去）

> 这几乎就是你想要的 `ModelBundle/ModelHandle`，只是 FastGen 的实现把它融合在
> `FastGenModel` 类里（OOP 风格）。

### 5.2 DMD2Model：多优化器/交替更新，Trainer 完全不需要知道

`fastgen/methods/distribution_matching/dmd2.py::DMD2Model` 的设计非常值得抄：

- 模型组成：
  - `net`：student（训练的 generator）
  - `teacher`：冻结
  - `fake_score`：critic（训练）
  - `discriminator`：可选（GAN loss 时训练）
- **交替更新策略**由 config 控制：`student_update_freq`
- 关键：通过覆写 `get_optimizers()` / `get_lr_schedulers()` 实现
  “每个 iteration step 哪些 optimizer/scheduler 走一步”

伪代码大概是：

```python
if iter % student_update_freq == 0:
  optimizers = [net_optimizer]
else:
  optimizers = [fake_score_optimizer, (optional) discriminator_optimizer]
```

这种方式的价值：

- 训练 loop 不会膨胀（Trainer 永远固定）
- method 可以自由扩展更多 role/更多 optimizer，而不用改 Trainer
- scheduler 的 step 粒度天然与 optimizer step 对齐（避免 update ratio 引入 lr schedule 偏差）

### 5.3 SelfForcingModel：继承链复用算法框架，只替换 student rollout

Self-forcing 在 FastGen 里是：

```
SelfForcingModel -> CausVidModel -> DMD2Model -> FastGenModel
```

它的主要改动非常克制：

- 不改 Trainer，不改 DMD2 的 loss 框架
- 只覆写 `gen_data_from_net()`：从普通 student forward 换成
  `rollout_with_gradient()`（blockwise causal rollout，只有 exit step 保留梯度）
- rollout 的随机 exit steps 用广播同步（rank0 采样，其他 rank broadcast）
- KV cache 用 **network 内部缓存**（`CausalFastGenNetwork.clear_caches()`），
  method 侧只调用 `store_kv=True/False` 的 forward

这体现出非常强的“算法复用”能力：Self-forcing 只是 DMD2 的一个 student 采样策略。

## 6. Network 抽象：统一 forward contract + causal 扩展（非常实用）

`fastgen/networks/network.py` 的接口设计对 distill 非常友好：

- `FastGenNetwork.forward(x_t, t, condition=..., fwd_pred_type=..., feature_indices=...)`
  - 允许 method 侧用统一的方式拿到：
    - x0/eps/flow 等不同 pred_type
    - 中间 features（给 discriminator）
    - logvar（给某些 consistency/uncertainty 变体）
- noise schedule 在 network 内统一（`self.noise_scheduler`），method 层不用各写一遍
- `CausalFastGenNetwork` 增加：
  - `chunk_size/total_num_frames`
  - `clear_caches()` 抽象，明确缓存生命周期

另一个小但很关键的点：

- `FastGenModel._student_sample_loop` 是通用 multistep loop，
  但如果 network 实现了 `preserve_conditioning(x, condition)`，
  loop 会自动调用它来保留 I2V / V2W 的 conditioning 帧/掩码（避免 loop 被各种模型特例污染）

## 7. 数据/conditioning 供给：把 uncond/neg prompt 变成“数据常量”（非常推荐）

FastGen 的 WebDataset loader（`datasets/wds_dataloaders.py`）提供了两个很棒的能力：

- `files_map`：把某些 key 从外部文件加载为常量（每个 batch 都带上）
  - 典型用法：`neg_condition`（negative prompt embedding）从 `.npy` 读取一次
- `presets_map`：把某些 key 直接用预设常量填充（比如 WAN 的负面 prompt 字符串）

这带来的直接收益：

- CFG/uncond 的输入不依赖 “validation 是否跑过” 或 “训练时是否临时 encode”
- 能把 expensive 的 negative prompt embedding 预先算好并缓存
- 对 offline / 大规模训练更友好（避免每 step 重新 encode）

`Trainer.preprocess_data()` 进一步提供可选的在线 encode：

- 若 batch 里是 raw video/image/text，则自动用 network 内的
  `vae/text_encoder/image_encoder` 编码成 latent/embeddings
- 同时保留 `*_raw` 字段，便于日志/可视化

这等价于把“pipeline stage”做成一个轻量的 preprocessing hook，集中在 Trainer。

## 8. 分布式与 checkpoint：围绕 `model_dict` 泛化（非常可维护）

### 8.1 DDP training_step 的包装（很巧）

`fastgen/utils/distributed/ddp.py::DDPWrapper` 做了一个小技巧：

- 训练逻辑在 `single_train_step`，但 DDP 的 hook 是绑定在 `forward()` 上的
- wrapper 临时把 `module.forward` 指到 `single_train_step`，然后调用 `self(...)`
  以触发 DDP 的 forward/backward hook

这让 “训练 step 不是 forward” 的设计依然能吃到 DDP 的正确行为。

### 8.2 FSDP2：支持 meta-init 的内存友好加载

`FastGenModel._get_meta_init_context()` + `utils/distributed/fsdp.py::model_to_fsdp` 支持：

- 非 rank0 在 `torch.device("meta")` 上构建大模型（几乎零内存）
- rank0 负责加载权重
- FSDP wrap 后通过 `sync_module_states` 广播权重到所有 rank

对于 10B+ 模型，这个设计非常关键：大幅降低启动时间与 I/O contention。

### 8.3 Checkpointer：同一套保存/加载适配 DDP 与 FSDP

`utils/checkpointer.py`：

- 非 FSDP：rank0 写单个 `.pth`（包含 model/optim/scheduler/grad_scaler/callbacks/iteration）
- FSDP：用 `torch.distributed.checkpoint` 分别保存每个 `model_dict` key 的
  sharded state（例如 `<path>.net_model`、`<path>.fake_score_model`），并且把
  scheduler/grad_scaler/callbacks/iteration 仍写到 `<path>.pth`

这种按 `model_dict` key 泛化的 checkpoint 方式，对多网络 distill 非常自然。

## 9. 对 FastVideo 的直接启发（建议落地清单）

结合你要做的 “`models={teacher, student, critic, ...}` + 算法解耦”，FastGen
给出的可落地模式：

1) **Trainer 只依赖统一接口**
- 固化为：`algorithm/model.single_train_step()` 返回 `loss_map["total_loss"]`
- 其余都放到算法层：update ratio、多优化器 step、cache 管理等

2) **把多网络/多 optimizer 的“调度”做成可覆盖函数**
- `get_optimizers(iter)` / `get_lr_schedulers(iter)` 的模式非常清晰
- 这比在 Trainer 里写 if/else 或在 pipeline 里复制 train_loop 更可维护

3) **role → checkpoint 的映射要结构化**
- 借鉴 `model_dict/optimizer_dict/scheduler_dict` 的思路：
  `models` 映射天然就是 checkpoint 的 namespace

4) **uncond/negative conditioning 不要依赖 validation**
- 学 FastGen：把 `neg_condition` 做成 dataset 常量（files_map/presets_map）
  或者训练开始时一次性 encode 并缓存/广播

5) **network 抽象要显式支持 distill 需求**
- forward contract 支持 `fwd_pred_type`、features、可选采样
- causal network 明确 cache 生命周期（`clear_caches()`）

6) **大模型加载要考虑 meta-init + FSDP2**
- 如果 FastVideo 也要跑 14B/更大 teacher，多机多卡启动成本会很明显
- meta-init + rank0 broadcast 是成熟方案
