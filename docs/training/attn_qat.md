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
bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh
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
| SM100, validated non-causal BF16 QAT configuration with head dimension 128 | Large-tile forward and split 64x64 backward; optimized backward requires a 16-aligned KV length |
| SM120, including RTX 5090 | Previous forward tiling with joined quantized/STE P@V operations and a shallower backward pipeline for long sequences |
| Unsupported configurations | Previous Triton implementation |

Warp specialization is disabled automatically on SM100 and SM120 because the
Triton 3.7 NVWS compiler pass aborts for this kernel on Blackwell. No user
setting is required.

The available tuning and comparison controls are:

| Environment variable | Default | Effect |
|---|---|---|
| `FASTVIDEO_ATTN_QAT_FWD_MODE` | `fast` | Selects `fast`, `balanced`, or `reference` forward tiling on the SM100 optimized route |
| `FASTVIDEO_ATTN_QAT_FWD_EXACT_M` | `0` | Set to `1` to recompute reference-order softmax statistics and keep `dV` bitwise-compatible on the SM100 optimized route |
| `FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED` | `1` | Set to `0` to force the previous SM100 forward and backward for comparison |
| `FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV` | `1` | Set to `0` to compare SM120 against the split P@V path |

The first invocation JIT-compiles the selected configuration; later calls reuse
the Triton cache. To measure the production shape, run
`python benchmarks/benchmark_attn_qat_train.py` from `fastvideo-kernel/`.

For import and backend-selection failures, see [Debugging](../utilities/debugging.md).
