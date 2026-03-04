# FastGen Framework Summary

Reference codebase: `~/alex/pkgs/FastGen`

## Naming Convention (Important!)

FastGen's naming maps to concepts differently than FastVideo:

| FastGen term | FastVideo term | What it actually is |
|---|---|---|
| `FastGenNetwork` | `ModelBase` | Neural network + noise schedule |
| `FastGenModel` (e.g. `DMD2Model`) | `DistillMethod` (e.g. `DMD2Method`) | Training algorithm / method |
| `Trainer` | `DistillTrainer` | Training loop orchestrator |

In FastGen: **"model" = method**, **"network" = model**.

---

## Architecture Overview

```
train.py
  -> load python config (attrs + OmegaConf + Hydra overrides)
  -> model = instantiate(config.model_class)   # a FastGenModel subclass
  -> Trainer(config).run(model)
       -> DDP/FSDP wrap
       -> model.init_optimizers()
       -> checkpointer.load(...)
       -> for iter in range(max_iter):
            for accum_iter:
              data = preprocess_data(batch)     # VAE/text encode
              loss_map, outputs = model.single_train_step(data, iter)
              backward(loss_map["total_loss"])
            model.optimizers_schedulers_step(iter)
            model.optimizers_zero_grad(iter)
            maybe_validate(model)
            maybe_save_checkpoint(model)
```

**Key invariant**: Trainer only ever calls `single_train_step()` and reads
`loss_map["total_loss"]`. All algorithm complexity lives inside the model
(method) class.

---

## Layer 1: FastGenNetwork (the neural network)

**File**: `fastgen/networks/network.py`

Abstract base that wraps a transformer/UNet with its noise schedule:

```python
class FastGenNetwork(ABC, torch.nn.Module):
    def __init__(self, net_pred_type="x0", schedule_type="edm", **kwargs):
        self.net_pred_type = net_pred_type       # "x0", "eps", "v", "flow"
        self.noise_scheduler = get_noise_schedule(schedule_type, **kwargs)

    @abstractmethod
    def forward(self, x_t, t, condition=None, *,
                fwd_pred_type=None,              # override pred type
                return_features_early=False,     # for discriminator
                feature_indices=None,
                return_logvar=False,
                **fwd_kwargs) -> Tensor: ...

class CausalFastGenNetwork(FastGenNetwork):
    """Adds chunk_size, total_num_frames, clear_caches()."""
    @abstractmethod
    def clear_caches(self): ...
```

The network owns the noise schedule, which provides:
- `forward_process(x0, eps, t)` — add noise: `alpha(t)*x0 + sigma(t)*eps`
- `sample_t(n, time_dist_type, ...)` — sample training timesteps
- `sample_t_inhom(n, seq_len, chunk_size, ...)` — per-chunk timestep sampling (causal)
- `x0_to_eps()`, `eps_to_x0()`, `flow_to_x0()`, `convert_model_output()` — conversions
- `latents(noise, t_init)` — scale noise by sigma for initial latent

### Noise Schedule Hierarchy

```
BaseNoiseSchedule (abstract)
├── EDMNoiseSchedule      # sigma(t)=t, alpha(t)=1, t∈[0.002, 80]
├── AlphasNoiseSchedule   # from alphas_cumprod
│   ├── SDNoiseSchedule
│   ├── SDXLNoiseSchedule
│   └── CogVideoXNoiseSchedule
├── RFNoiseSchedule       # rectified flow: alpha(t)=1-t, sigma(t)=t
└── TrigNoiseSchedule     # alpha(t)=cos(t), sigma(t)=sin(t)
```

Time sampling distributions: `uniform`, `lognormal`, `logitnormal`, `polynomial`,
`shifted`, `log_t`.

---

## Layer 2: FastGenModel (the training method)

**File**: `fastgen/methods/model.py` (~717 lines)

Template-method base class that all training algorithms inherit from:

```python
class FastGenModel(torch.nn.Module):
    # --- Construction ---
    def build_model(self):
        self.net = instantiate(config.net)           # student network
        self.build_teacher()                          # optional frozen teacher
        self._setup_ema()                             # optional EMA copies

    def build_teacher(self):
        self.teacher = instantiate(config.teacher or config.net)
        self.teacher.eval().requires_grad_(False)

    # --- Polymorphic dictionaries (key for checkpoint/FSDP) ---
    @property
    def model_dict(self) -> dict:     # {"net": ..., "teacher": ..., "ema": ...}
    @property
    def fsdp_dict(self) -> dict:      # subset to be FSDP-sharded
    @property
    def ema_dict(self) -> dict:       # all EMA networks
    @property
    def optimizer_dict(self) -> dict:
    @property
    def scheduler_dict(self) -> dict:

    # --- Training interface (overridden by subclasses) ---
    @abstractmethod
    def single_train_step(self, data, iteration) -> (loss_map, outputs): ...

    @abstractmethod
    def _get_outputs(self, gen_data, ...) -> dict: ...

    # --- Optimizer management ---
    def init_optimizers(self): ...
    def get_optimizers(self, iteration) -> list[Optimizer]: ...
    def get_lr_schedulers(self, iteration) -> list[Scheduler]: ...
    def optimizers_schedulers_step(self, iteration): ...
    def optimizers_zero_grad(self, iteration): ...

    # --- Inference ---
    @staticmethod
    def generator_fn(net, noise, condition, t_list, ...): ...
    def _student_sample_loop(self, net, noise, condition, ...): ...

    # --- Precision ---
    def set_precision(self, precision, precision_amp, ...): ...
```

### Method Inheritance Tree

```
FastGenModel
├── SFTModel                         # supervised fine-tuning
│   └── CausalSFTModel
├── KDModel                          # knowledge distillation (pre-paired data)
│   └── CausalKDModel
├── CMModel                          # consistency model
│   ├── TCMModel                     # two-stage consistency
│   ├── SCMModel                     # simplified consistency (TrigFlow)
│   └── MeanFlowModel               # mean flow matching
├── DMD2Model                        # distribution matching distillation v2
│   ├── FdistillModel                # f-divergence weighted DMD
│   ├── LADDModel                    # latent adversarial diffusion distillation
│   └── CausVidModel                 # causal video DMD
│       └── SelfForcingModel         # self-forcing (causal rollout)
```

---

## Layer 3: Trainer (training loop)

**File**: `fastgen/trainer.py` (~544 lines)

Completely algorithm-agnostic. Handles:
- DDP/FSDP wrapping
- Gradient accumulation with sync control
- Data preprocessing (VAE encode, text encode)
- Validation (reuses `single_train_step` with no_grad)
- Checkpoint save/load
- Callback system (EMA update, grad clip, logging, profiling)

The trainer never knows about roles, multiple networks, or alternating updates.
All of that is encapsulated in the model (method) class.

---

## Method Details

### SFTModel (Supervised Fine-Tuning)

**File**: `fastgen/methods/fine_tuning/sft.py`

Simple diffusion training with optional CFG dropout:

```
t = sample_t(batch_size)
eps = randn_like(real_data)
x_t = forward_process(real_data, eps, t)
condition = mix_with_neg_condition(condition, cond_dropout_prob)
pred = net(x_t, t, condition)
loss = denoising_score_matching_loss(pred, x0=real_data, eps=eps, t=t)
```

Condition dropout: randomly replaces condition with `neg_condition` per sample,
with configurable `cond_dropout_prob` and `keys_no_dropout` for selective dropout.

### KDModel (Knowledge Distillation)

**File**: `fastgen/methods/knowledge_distillation/KD.py`

Learns from pre-constructed teacher ODE trajectories:

- **Single-step**: student maps pure noise → x0, loss = MSE(pred, target)
- **Multi-step**: student maps intermediate path point → x0
  - `path` tensor: `[B, num_inf_steps, C, F, H, W]` (pre-computed teacher trajectory)
  - Samples random t from `t_list`, gathers corresponding path point
  - Supports 2-step and 4-step distillation

**CausalKDModel**: Uses `sample_t_inhom()` for per-chunk timestep sampling.

### DMD2Model (Distribution Matching Distillation v2)

**File**: `fastgen/methods/distribution_matching/dmd2.py` (~532 lines)

Three networks: student (`net`), teacher (frozen), fake_score (critic).
Optional: discriminator (GAN loss).

**Alternating updates** controlled by `student_update_freq`:

```
if iter % student_update_freq == 0:
    # Student update
    gen_data = net(input, t_student)               # generate x0
    x_t = forward_process(gen_data, eps, t)        # re-noise
    teacher_x0 = teacher(x_t, t, cfg=True)         # teacher prediction (CFG)
    fake_score_x0 = fake_score(x_t, t).detach()    # critic prediction

    vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0)
    gan_loss = gan_loss_generator(discriminator(features))  # optional
    total_loss = vsd_loss + gan_weight * gan_loss

    optimizers = [net_optimizer]
else:
    # Critic + discriminator update
    fake_score_loss = denoising_score_matching_loss(fake_score(x_t, t), gen_data, eps, t)
    gan_loss = gan_loss_discriminator(real_logits, fake_logits)  # optional

    optimizers = [fake_score_optimizer, discriminator_optimizer]
```

**VSD Loss** (the core DMD2 gradient):

```python
def variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0,
                                         additional_scale=None):
    w = 1 / (|gen_data - teacher_x0|.mean() + eps)   # adaptive weight
    if additional_scale: w *= additional_scale         # f-div weighting
    vsd_grad = (fake_score_x0 - teacher_x0) * w
    pseudo_target = gen_data - vsd_grad
    return 0.5 * MSE(gen_data, pseudo_target)
```

### FdistillModel (F-Divergence Distillation)

**File**: `fastgen/methods/distribution_matching/f_distill.py`

Extends DMD2 with importance weighting via learned discriminator logits:

```python
f_div_weighting = {
    "rkl": lambda r: 1,          # reverse KL
    "kl":  lambda r: r,          # forward KL
    "js":  lambda r: 1-1/(1+r),  # Jensen-Shannon
    "sf":  lambda r: 1/(1+r),    # squared Hellinger
    "sh":  lambda r: r**0.5,     # Hellinger
    ...
}
```

Density ratio estimated from discriminator logits → used as `additional_scale`
in VSD loss. Optional EMA histogram normalization across timestep bins.

### CausVidModel / SelfForcingModel (Causal Video Distillation)

**Files**: `fastgen/methods/distribution_matching/causvid.py`, `self_forcing.py`

**CausVidModel** extends DMD2 for causal (autoregressive) video generation:
- Per-chunk timestep sampling via `sample_t_inhom()`
- Autoregressive rollout with KV cache management
- Context noise injection for cache warmup

**SelfForcingModel** extends CausVid, only overrides `gen_data_from_net()`:
- Blockwise causal rollout with random exit timesteps
- Only exit steps retain gradients (memory efficient)
- Exit steps broadcast-synced across ranks

Inheritance chain: `SelfForcingModel → CausVidModel → DMD2Model → FastGenModel`

### CMModel (Consistency Model)

**File**: `fastgen/methods/consistency_model/CM.py`

Learns consistency trajectory:

```
r = t_to_r_sigmoid(t, ratio)            # map t → smaller r
y_t = forward_process(x0, eps, t)
y_r = ode_solver(teacher, y_t, t, r)    # CD mode (or forward_process for CT)
D_yt = net(y_t, t)
D_yr = net(y_r, r).detach()
loss = ||D_yt - D_yr|| / (t - r)        # weighted consistency loss
```

Weighting options: `default` (1/(t-r)), `c_out`, `sigma_sq`.

### TCMModel (Two-Stage Consistency)

**File**: `fastgen/methods/consistency_model/TCM.py`

Wraps network in `TCMPrecond` (boundary conditions for consistency).
Two training stages with different loss formulations.

### SCMModel (Simplified Consistency / TrigFlow)

**File**: `fastgen/methods/consistency_model/sCM.py`

Uses `TrigFlowPrecond` with trigonometric noise schedule.
Supports pseudo-Huber loss and logvar-based adaptive weighting.

### MeanFlowModel

**File**: `fastgen/methods/consistency_model/mean_flow.py` (~503 lines)

Velocity-parameterized consistency training with mean flow matching.

---

## Loss Functions (Centralized)

**File**: `fastgen/methods/common_loss.py`

```python
def denoising_score_matching_loss(pred_type, net_pred, x0, eps, t):
    """Unified DSM loss for all prediction types."""
    if pred_type == "x0":   target = x0
    elif pred_type == "eps": target = eps
    elif pred_type == "v":   target = alpha(t)*eps - sigma(t)*x0
    elif pred_type == "flow": target = eps - x0
    return MSE(net_pred, target)

def variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0,
                                         additional_scale=None):
    """VSD loss with adaptive weighting (DMD2 core)."""
    w = 1 / (|gen_data - teacher_x0|.mean() + eps)
    if additional_scale: w *= additional_scale
    pseudo_target = gen_data - (fake_score_x0 - teacher_x0) * w
    return 0.5 * MSE(gen_data, pseudo_target)

def gan_loss_generator(fake_logits):
    return softplus(-fake_logits).mean()

def gan_loss_discriminator(real_logits, fake_logits):
    return softplus(fake_logits).mean() + softplus(-real_logits).mean()
```

---

## Configuration System

**Structure**: attrs dataclasses + OmegaConf + LazyCall + Hydra overrides

### Three-level config hierarchy

1. **Base config** (`fastgen/configs/config.py`):
   ```python
   @attrs.define
   class BaseModelConfig:
       net: dict                           # network config (LazyCall)
       teacher: Optional[dict] = None
       guidance_scale: Optional[float]
       net_optimizer: dict
       net_scheduler: dict
       sample_t_cfg: SampleTConfig         # timestep distribution params
       input_shape: list[int]
       pretrained_model_path: str
       use_ema: Any = False
       student_sample_steps: int = 1
       fsdp_meta_init: bool = False

   @attrs.define
   class SampleTConfig:
       time_dist_type: str = "uniform"
       train_p_mean: float = -1.1
       train_p_std: float = 2.0
       min_t: float = 0.002
       max_t: float = 80.0
       t_list: Optional[list[float]] = None
   ```

2. **Method configs** (`fastgen/configs/methods/config_*.py`):
   - `config_dmd2.py`: adds `fake_score_optimizer`, `discriminator`, `student_update_freq`
   - `config_f_distill.py`: adds `FdistillConfig` (f_div type, ratio bounds)
   - `config_cm.py`: adds `CMLossConfig` (weighting, use_cd, ratio)

3. **Experiment configs** (`fastgen/configs/experiments/<model>/config_<method>.py`):
   - Concrete settings: model paths, input_shape, lr, t_list, etc.

### LazyCall / instantiate

Config entries use `LazyCall` to record `_target_` + kwargs. Objects are created
via `instantiate(cfg)` at runtime, enabling full pluggability.

---

## Distributed Training

### DDP Trick

`DDPWrapper` temporarily redirects `module.forward` → `single_train_step` so
that DDP's forward/backward hooks fire correctly even though the training logic
isn't a standard forward pass.

### FSDP2 with Meta-Init

For large models (10B+):
1. Non-rank0 processes build model on `torch.device("meta")` (zero memory)
2. Rank0 loads weights
3. FSDP wrap with `sync_module_states` broadcasts weights

### Checkpoint

- **Non-FSDP**: rank0 saves single `.pth` (model + optim + scheduler + iteration)
- **FSDP**: `torch.distributed.checkpoint` per `model_dict` key (sharded),
  plus scalar state in `.pth`

Checkpoint keys derived directly from `model_dict` / `optimizer_dict` / `scheduler_dict`
properties — naturally supports multi-network methods.

---

## Data Pipeline

**File**: `fastgen/datasets/wds_dataloaders.py`

WebDataset-based with two key features:

- `files_map`: load constants from files (e.g., pre-computed neg_condition embedding)
- `presets_map`: inject literal constants into every batch

`Trainer.preprocess_data()` handles optional online encoding:
- VAE: `data["real"] = net.vae.encode(data["real"])`
- Text: `data["condition"] = net.text_encoder.encode(data["condition"])`
- I2V/V2W: first-frame conditioning, image embeddings

---

## Key Design Principles

1. **Trainer is algorithm-agnostic**: only knows `single_train_step()` + `total_loss`
2. **Method = multi-network container + algorithm**: owns nets, optimizers, update logic
3. **Alternating updates via `get_optimizers(iter)`**: no trainer changes needed
4. **Network owns noise schedule**: method layer doesn't reimplement diffusion math
5. **Centralized loss functions**: all methods share `common_loss.py`
6. **Inheritance for algorithm variants**: SelfForcing only overrides `gen_data_from_net()`
7. **Polymorphic dicts for checkpoint**: `model_dict`/`optimizer_dict` scale to any number of roles
8. **Config-driven pluggability**: LazyCall + instantiate for all components
