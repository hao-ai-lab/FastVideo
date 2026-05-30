# 🌊 AnyFlow Any-Step Video Distillation

**AnyFlow** ([paper](https://arxiv.org/abs/2605.13724), [project page](https://nvlabs.github.io/AnyFlow/), [official code](https://github.com/NVlabs/AnyFlow), [model weights](https://huggingface.co/collections/nvidia/anyflow)) is an any-step video diffusion framework built on flow maps. A single distilled checkpoint can be evaluated at NFE ∈ {1, 2, 4, 8, 16, 32} without retraining, and quality scales **monotonically** with steps — unlike consistency-based distillation, which often degrades as NFE grows.

The student network ``u_θ(x_t, t, r)`` predicts the *average velocity* from time ``t`` back to time ``r``, so one Euler step is

```
x_r = x_t - ((t - r) / N) · u_θ(x_t, t, r)
```

for any ``t > r``.

## 📊 Model Overview

NVIDIA publishes four checkpoints under [`nvidia/anyflow`](https://huggingface.co/collections/nvidia/anyflow):

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers` — bidirectional T2V, Wan2.1 1.3B base
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers` — bidirectional T2V, Wan2.1 14B base
- `nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers` — frame-autoregressive variant, 1.3B
- `nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers` — frame-autoregressive variant, 14B

FastVideo currently supports the bidirectional T2V variants for training; the FAR variants can be loaded for inference through the diffusers integration.

## ⚙️ Inference

For inference, load the published checkpoint directly through diffusers; FastVideo's training-side ``WanModel`` config maps the HF AnyFlow ``delta_embedder`` weights onto its internal layout via ``param_names_mapping`` so the same checkpoint can be used as the ``init_from`` for the on-policy YAML below.

## 🧠 Algorithm

Training runs in two stages. Both use the dual-timestep Wan backbone — enabled by ``pipeline.dit_config.r_embedder: true`` in the YAML, which allocates a sibling ``condition_embedder.delta_embedder`` and fuses its embedding with the standard timestep embedding via either an additive or a gated mixer.

### Stage 1 — Pretrain (flow-map central-difference)

Method: ``AnyFlowPretrainMethod`` (``fastvideo/train/methods/distribution_matching/anyflow_pretrain.py``)

For each batch, sample ``(t, r) ∈ [0, 1]`` as ``(max, min)`` of two uniform draws, then:

- a ``diffusion_ratio`` fraction (default 0.5) gets ``r = t`` — recovers plain flow matching;
- a ``consistency_ratio`` fraction (default 0.25) gets ``r = 0`` — forces consistency to clean data;
- the remainder is free.

The student forward at ``(t, r)`` is trained against the central-difference target

```
target = (eps - x_0) - (t - r) · dF/dt
```

where ``dF/dt`` is estimated from the student's own forward at ``(t ± δ, r)`` with the sample also moved along the flow trajectory by ``v_pred · (δ / N)``. Per-timestep weighting uses ``beta08`` (``w(t) = t · sqrt(1 - t)``, renormalized). A stop-gradient scale-balance keeps the non-diffusion branches' loss magnitude aligned with the diffusion branch.

### Stage 2 — On-policy DMD

Method: ``AnyFlowMethod`` (``fastvideo/train/methods/distribution_matching/anyflow.py``)

Inherits ``DMD2Method``. The student is rolled out for ``student_sample_steps`` Euler-flow steps from pure noise; one randomly-chosen step is gradient-enabled (broadcast from rank 0 so every worker agrees), the rest run under ``torch.no_grad``. With ``use_mean_velocity: true`` (default) the rollout uses ``r = t_next`` at each step, matching AnyFlow's ``WanAnyFlowPipeline.training_rollout``.

The inherited ``_dmd_loss`` (VSD with fake-score critic) consumes the rollout output and the teacher's CFG prediction. The optional pinned ``t_list_override`` lets configs reproduce the paper's hand-tuned 4-step schedule ``[999, 937, 833, 624, 0]``.

## 🚀 Training Scripts

### Stage 1 — pretrain

```bash
bash examples/train/run.sh \
    examples/train/configs/distribution_matching/wan/anyflow_pretrain_t2v.yaml
```

**Key configuration** (in ``examples/train/configs/distribution_matching/wan/anyflow_pretrain_t2v.yaml``):

- Global batch size: 32 (8 GPUs × 4 per-GPU)
- Learning rate: 5e-5
- Flow shift: 5.0
- ``diffusion_ratio`` / ``consistency_ratio``: 0.5 / 0.25
- ``epsilon`` (finite-difference step): 5 (absolute train-timestep units)
- ``weight_type``: ``beta08``
- ``fuse_guidance_scale``: 3.0
- Training steps: 6000

### Stage 2 — on-policy

```bash
bash examples/train/run.sh \
    examples/train/configs/distribution_matching/wan/anyflow_onpolicy_t2v.yaml \
    --models.student.init_from outputs/wan2.1_anyflow_pretrain/checkpoint-final
```

(Or point ``models.student.init_from`` directly at ``nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`` to bootstrap from the paper weights and skip Stage 1.)

**Key configuration**:

- Global batch size: 8 (8 GPUs × 1 per-GPU)
- Learning rate: 2e-6
- Flow shift: 5.0
- ``student_sample_steps``: 4
- ``t_list_override``: ``[999, 937, 833, 624, 0]``
- ``use_mean_velocity``: ``true`` (i.e. ``r = t_next`` during rollout)
- ``real_score_guidance_scale``: 3.0
- ``generator_update_interval``: 5 (DMD2 alternation)
- Training steps: 4000

## 🔌 Loading published AnyFlow checkpoints

The HF AnyFlow checkpoints expose ``condition_embedder.delta_embedder.*`` weights that FastVideo internally maps onto its ``condition_embedder.delta_embedder.mlp.*`` layout. This rename happens automatically through the regex in ``WanVideoArchConfig.param_names_mapping`` — no separate adapter is needed. The same regex is a no-op on plain Wan checkpoints (which don't contain any ``delta_embedder`` keys).

Set the YAML's ``pipeline.dit_config.r_embedder: true`` to allocate the ``delta_embedder`` module on the FastVideo side; when initializing from a plain Wan checkpoint the delta weights are deep-copied from ``time_embedder`` (matching AnyFlow's ``setup_flowmap_model()`` behavior).

## 🧭 Note on ``fuse_guidance_scale``

Stage 1 optionally fuses classifier-free guidance into the training target so the resulting checkpoint can be sampled at ``guidance_scale=1.0`` (no extra forward pass at inference time). The transformation is

```
noise_pred ← (noise_pred - (1 - g) · noise_pred_uncond) / g
```

with ``g = fuse_guidance_scale``. The negative prompt embedding comes from ``WanModel``'s ``ensure_negative_conditioning()`` — i.e. the dataset's configured ``sampling_param.negative_prompt``. Setting ``fuse_guidance_scale: 1.0`` skips the extra unconditional forward entirely.

The on-policy stage's ``real_score_guidance_scale`` (inherited from DMD2) follows the same parameterization conventions documented in [``dmd.md``](dmd.md#-note-on-real_score_guidance_scale).
