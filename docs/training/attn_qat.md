# Attn-QAT Training

Attn-QAT simulates low-bit attention during training while keeping the rest of
the training method unchanged. In the modular `fastvideo/train` framework it is
a per-role model option, not a separate training method: supervised fine-tuning
and DMD2 still own their losses and optimizer cadence.

This guide covers the QAD Wan2.1-T2V-1.3B MixKit workflow:

1. run a 4,000-step supervised Attn-QAT fine-tune;
2. export the stage-1 DCP checkpoint to Diffusers format; and
3. distill the student to three denoising steps with DMD2.

The ready-to-run configs and wrappers are in
`examples/train/scenario/qad_wan2_1_mixkit/`.

## Role-local attention backends

A DMD2 run owns three independent model roles. Configure the attention backend
on each role so fake quantization is applied only to the student:

```yaml
models:
  student:
    attention_backend: ATTN_QAT_TRAIN
  teacher:
    attention_backend: FLASH_ATTN
  critic:
    attention_backend: FLASH_ATTN
```

The override is active only while that role's transformer is constructed, then
the previous process-wide backend is restored. This lets student, teacher, and
critic use different implementations in one process. Invalid role-level names
fail during configuration instead of silently selecting another backend.

See [Training Infrastructure](train_infra.md) for the complete model-role
configuration reference.

## Prerequisites

- Install FastVideo and make the `fastvideo-kernel` Python package importable.
  `ATTN_QAT_TRAIN` intentionally fails instead of falling back to dense
  attention when its kernel cannot be loaded.
- Prepare the precomputed MixKit VAE latents and text embeddings.
- Run the commands below from the repository root. The supplied recipe expects
  four GPUs by default; set `NUM_GPUS` to override it.

Download the published preprocessed dataset:

```bash
bash examples/datasets/mixkit/download_dataset.sh
```

## Stage 1: supervised Attn-QAT fine-tuning

The stage-1 config uses `ATTN_QAT_TRAIN` on the student, sequence parallelism
across four GPUs, FP32 master weights, and 4,000 optimizer steps:

```bash
NUM_GPUS=4 \
  bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh
```

Pass a dataset directory as the first positional argument when it differs from
the default:

```bash
NUM_GPUS=4 \
  bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh \
  /path/to/combined_parquet_dataset
```

The wrapper calls `examples/train/run.sh`; the YAML file remains the source of
truth for optimizer, validation, checkpointing, and distributed settings.

## Export the stage-1 checkpoint

Modular training checkpoints use Distributed Checkpoint (DCP) format. Export
the student before using it to initialize stage 2:

```bash
bash examples/train/scenario/qad_wan2_1_mixkit/export_stage1.sh \
  checkpoints/wan_t2v_qat_finetune/checkpoint-4000 \
  checkpoints/wan_t2v_qat_finetune/diffusers
```

Both arguments are optional; the command above shows their defaults.

## Stage 2: three-step DMD2 distillation

Stage 2 loads the exported student weights, keeps Attn-QAT on the student, and
uses Flash Attention for the teacher and critic:

```bash
NUM_GPUS=4 \
  bash examples/train/scenario/qad_wan2_1_mixkit/run_stage2.sh \
  data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset \
  checkpoints/wan_t2v_qat_finetune/diffusers/transformer/model.safetensors
```

The migrated recipe preserves these behaviors:

| Behavior | Modular configuration |
|---|---|
| Student fake-quantized attention | `models.student.attention_backend: ATTN_QAT_TRAIN` |
| Teacher and critic full-precision attention | Role-local `FLASH_ATTN` |
| Generator update every five critic steps | `method.generator_update_interval: 5` |
| Three-step rollout | `method.dmd_denoising_steps: [1000, 757, 522]` |
| Score timestep range | `method.min_timestep_ratio: 0.02`, `max_timestep_ratio: 0.98` |
| Legacy guidance `cond + 2(cond - uncond)` | Standard CFG scale `3.0` |
| Stage handoff | DCP checkpoint to Diffusers export to student override weights |

The timestep ratios apply to randomly sampled teacher and critic score
timesteps; `dmd_denoising_steps` separately controls the student rollout. See
[DMD Distillation](../distillation/dmd.md) for general DMD concepts.

## Architecture-specific Triton routing

The training kernel is runtime-JIT-compiled Triton code and selects its route on
every call. It supports different query and key/value sequence lengths for
cross-attention; key and value must have the same sequence length.

| Hardware/configuration | Route |
|---|---|
| SM100, validated non-causal BF16 QAT configuration with head dimension 128 | Large-tile forward and split backward. Long equal-length sequences (at least 16,384 tokens) use 64x128 backward tiles with tuned launch parameters in every forward mode; in `fast` mode without exact-M they also use complete forward tiles without bounds masks and accumulator lifetime tuning. Other optimized cases use the masked forward loop and 64x64 backward. Optimized backward requires a 16-aligned KV length |
| SM120, including RTX 5090 | Previous forward tiling with joined quantized/STE P@V operations and a shallower backward pipeline for long sequences |
| SM121, including DGX Spark / GB10 | Previous forward tiling with split quantized/STE P@V operations and a shallower backward pipeline for long sequences |
| Unsupported configurations | Previous Triton implementation |

On GB10, the joined P@V path does not preserve the split path's softmax
statistics, outputs, or gradients. Joining is restricted to SM120, even when
`FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV=1` is set on GB10.

Warp specialization is disabled automatically on SM100, SM120, and SM121 because the
Triton 3.7 NVWS compiler pass aborts for this kernel on Blackwell. No user
setting is required.

The available tuning and comparison controls are:

| Environment variable | Default | Effect |
|---|---|---|
| `FASTVIDEO_ATTN_QAT_FWD_MODE` | `fast` | Selects `fast`, `balanced`, or `reference` forward tiling on the SM100 optimized route |
| `FASTVIDEO_ATTN_QAT_FWD_EXACT_M` | `0` | Set to `1` to recompute reference-order softmax statistics and keep `dV` bitwise-compatible on the SM100 optimized route |
| `FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED` | `1` | Set to `0` to force the previous SM100 forward and backward for comparison |
| `FASTVIDEO_ATTN_QAT_SM100_WIDE_BWD` | `1` | Set to `0` to retain 64x64 backward tiles for long equal-length SM100 attention. Applies in every forward mode, including `reference` and `FWD_EXACT_M=1` |
| `FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV` | `1` | On SM120 only, set to `0` to compare against the split P@V path; GB10 always uses split P@V |
| `FASTVIDEO_ATTN_QAT_SM121_FWD_DV` | `1` | On GB10 only, set to `0` to fall back to the legacy re-quantized-P dV (see *GB10 forward-consistent dV*) |
| `FASTVIDEO_ATTN_QAT_SM121_DV_STATS` | `save` | On GB10 only: `save` keeps the forward's online maxima and denominators for backward; `recompute` replays the forward in backward instead |

The first invocation JIT-compiles the selected configuration; later calls reuse
the Triton cache. To measure the production shape, run
`python benchmarks/benchmark_attn_qat_train.py` from `fastvideo-kernel/`.

On two B200 GPUs with Wan2.1 1.3B, 31,200 tokens, full activation checkpointing,
and AdamW, the default long-sequence SM100 route measured about 10% lower full-step
latency than the previous SM100 optimized path in guarded SP2 acceptance runs.

## GB10 forward-consistent dV

On GB10 (SM121) the backward replaces the legacy dV computation for the
validated training configuration: non-causal, contiguous BF16 QAT attention
with head dimension 128, fake-quantized QKV in backward, P fake quantization
enabled, both additional P scaling modes disabled, and K smoothing disabled.
The public attention signature is unchanged; other configurations and other
GPUs keep the legacy backward.

The legacy backward re-quantizes the *normalized* probabilities. The E4M3
block scale of a 16-key group rounds to zero once the group's largest
probability drops below about `2^-10` (with uniform attention this happens
from sequence length 171), and every key in that group then receives no V
gradient. On real Wan2.1 activations at 8192 and 31200 tokens, 95-100% of the
groups underflow and whole heads receive `dV = 0`. The forward-consistent
path instead reconstructs the forward's quantized online probabilities from
the saved per-tile running maxima and the final denominator, and applies
those weights to the upstream gradient in FP32: V's fake quantizer uses an
identity STE, so dV is built from the weights the forward actually used. For
uniform self-attention the expected dV is therefore 1.03125, because
`E4M3(1/6) * E2M1(1/E4M3(1/6)) = 0.171875 * 6`; this differs from the dense
attention gradient of 1. Forward output, dQ, dK, the saved STE output and M
are unchanged bit for bit.

Cost: `save` retains `B * H * ceil(N_kv / 32) * N_q * 4` bytes of maxima plus
`B * H * N_q * 4` bytes of denominators per layer; with non-reentrant
activation checkpointing only one layer's statistics are live at a time.
`recompute` does not retain them across the forward/backward boundary but
still allocates them and scratch outputs during backward and adds a full
forward replay. Both modes run the legacy backward for dQ/dK plus an
additional dV kernel; measure the cost on the production shape with
`python benchmarks/benchmark_attn_qat_train.py` from `fastvideo-kernel/`,
comparing `FASTVIDEO_ATTN_QAT_SM121_FWD_DV=1` against `=0`.

The contract tests in `fastvideo-kernel/tests/test_attn_qat_sm121_dv.py`
cover the uniform dV value, an independent Torch forward and frozen-weight V
gradient reference, bitwise parity of the unchanged tensors, determinism and
non-reentrant activation checkpointing. They do not establish training
convergence or behaviour on SM100 or SM120. The shared quantizer excludes
explicitly masked values from scale selection and makes empty groups decode
to a finite zero; callers must supply a correct validity mask.

For import and backend-selection failures, see [Debugging](../utilities/debugging.md).
