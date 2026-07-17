# Attn-QAT on the modular trainer

Attn-QAT is integrated as a per-role model execution option in
`fastvideo/train`. It is not a separate training method: `FineTuneMethod` and
`DMD2Method` keep ownership of their losses and optimizer cadence, while each
role chooses the attention implementation used when its transformer is built.

## Why the backend is role-local

A DMD2 run owns three independent model roles:

```yaml
models:
  student:
    attention_backend: ATTN_QAT_TRAIN
  teacher:
    attention_backend: FLASH_ATTN
  critic:
    attention_backend: FLASH_ATTN
```

This keeps fake-quantized attention on the student and full-precision attention
on the teacher and critic. Callers no longer need to set or coordinate a
process-wide `FASTVIDEO_ATTENTION_BACKEND` value; the explicit role selection
takes precedence while that transformer's layers are built. An invalid
role-level backend name fails during configuration instead of silently
selecting another implementation.

## QAD Wan2.1 MixKit recipe

The migrated two-stage configs live under
`examples/train/scenario/qad_wan2_1_mixkit/`:

- `stage1_attn_qat_finetune.yaml`: 4,000-step supervised Attn-QAT finetune.
- `stage2_attn_qat_dmd.yaml`: DMD2 distillation to timesteps
  `[1000, 757, 522]` with student-only Attn-QAT.

Use the existing MixKit download or preprocessing scripts to prepare the
Parquet dataset, then run the modular workflow from the repository root:

```bash
# Prepare precomputed VAE latents and text embeddings.
bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh

# Stage 1: Attn-QAT supervised finetune.
NUM_GPUS=4 bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh

# Export the stage-1 DCP checkpoint to a Diffusers model directory.
bash examples/train/scenario/qad_wan2_1_mixkit/export_stage1.sh \
    checkpoints/wan_t2v_qat_finetune/checkpoint-4000 \
    checkpoints/wan_t2v_qat_finetune/diffusers

# Stage 2: three-step Attn-QAT DMD2 distillation.
NUM_GPUS=4 bash examples/train/scenario/qad_wan2_1_mixkit/run_stage2.sh \
    data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset \
    checkpoints/wan_t2v_qat_finetune/diffusers/transformer/model.safetensors
```

The scenario wrapper scripts call `examples/train/run.sh` and only supply path and
distributed-dimension overrides. The YAML files remain the source of truth for
the training recipe.

## Legacy-to-modular behavior mapping

The migration preserves these training semantics:

| Behavior | Modular configuration |
|---|---|
| Student fake-quantized attention | `models.student.attention_backend: ATTN_QAT_TRAIN` |
| Teacher/critic full-precision attention | Role-local `FLASH_ATTN` |
| Generator update every five critic steps | `method.generator_update_interval: 5` |
| Three-step rollout | `method.dmd_denoising_steps: [1000, 757, 522]` |
| Score timestep range | `method.min_timestep_ratio: 0.02`, `max_timestep_ratio: 0.98` |
| Legacy teacher guidance `cond + 2(cond-uncond)` | Standard CFG scale `3.0` |
| Stage handoff | DCP checkpoint → `dcp_to_diffusers` → student override safetensor |

The attention training kernel is required when the QAT transformer is built.
`ATTN_QAT_TRAIN` refuses to silently fall back when that kernel is unavailable,
because doing so would turn the run into non-QAT training.

## Optimized kernel auto-discovery

The `ATTN_QAT_TRAIN` backend patches FastVideo's Triton attention automatically
with the optimized `qat_attn` forward and backward. No manual `PYTHONPATH` or
Python patch call is needed. FastVideo resolves the package in this order:

1. the source checkout named by `QAT_ATTN_REPO`;
2. an installed/importable `qat_attn` package in the active Python environment;
   and
3. a sibling checkout at `../nvfp4_qat_attn`.

With the standard sibling layout, stage 1 is therefore a single command:

```bash
cd /path/to/FastVideo
NUM_GPUS=4 bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh
```

For another layout, point at the repository root in the same command:

```bash
QAT_ATTN_REPO=/path/to/nvfp4_qat_attn NUM_GPUS=4 \
  bash examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh
```

The optimized production defaults are `QAT_ATTN_FWD_MODE=fast`, exact forward
softmax statistics (`QAT_ATTN_FWD_EXACT_M=1`), and the faster nondeterministic
backward (`QAT_ATTN_DETERMINISTIC=0`). Set `QAT_ATTN_FWD_EXACT_M=0` only when
maximum forward throughput is more important than bitwise `dV` parity with the
reference kernel. The first backward invocation JIT-compiles and caches the
standalone CUDA extension and kernel; later runs reuse those caches.
