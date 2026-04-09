# Attn QAT

Attn QAT in FastVideo currently has two related backends:

- `ATTN_QAT_TRAIN`: the training-oriented Triton attention path.
- `ATTN_QAT_INFER`: the inference-oriented CUDA kernel path.

Both are selected with `FASTVIDEO_ATTENTION_BACKEND`, but they come from
different codepaths and serve different purposes.

## Backend Overview

| Backend | Best for | Package requirement | Primary kernel location |
|---------|----------|---------------------|-------------------------|
| `ATTN_QAT_TRAIN` | finetuning, validation during training, reproducing the training attention path | `fastvideo_kernel` | `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py` |
| `ATTN_QAT_INFER` | standalone inference with the `attn_qat_infer` CUDA kernel | `attn_qat_infer` from the in-repo `fastvideo-kernel` checkout | `fastvideo-kernel/attn_qat_infer/` |

In FastVideo itself, backend selection is routed through:

- `fastvideo/envs.py`
- `fastvideo/platforms/cuda.py`
- `fastvideo/attention/backends/attn_qat_train.py`
- `fastvideo/attention/backends/attn_qat_infer.py`

The legacy training pipeline also contains explicit Attn QAT integration:

- `fastvideo/training/training_pipeline.py`

That pipeline forces generator loading through `ATTN_QAT_TRAIN` when
`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` or
`--generator_4bit_attn` is enabled.

## Where The Code Lives

Use these paths when you want to trace or modify the Attn QAT flow:

| Location | Purpose |
|----------|---------|
| `fastvideo/attention/backends/attn_qat_train.py` | FastVideo wrapper that imports and calls the Triton training kernel |
| `fastvideo/attention/backends/attn_qat_infer.py` | FastVideo wrapper that imports and calls the inference kernel |
| `fastvideo-kernel/CMakeLists.txt` | Kernel build definition that compiles the `attn_qat_infer` inference extensions |
| `fastvideo/platforms/cuda.py` | Chooses the concrete attention backend at runtime |
| `fastvideo/envs.py` | Documents supported `FASTVIDEO_ATTENTION_BACKEND` values |
| `fastvideo/training/training_pipeline.py` | Training-time forcing logic for the generator attention backend |
| `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py` | Triton implementation for `ATTN_QAT_TRAIN` |
| `fastvideo-kernel/attn_qat_infer/api.py` | Python API entrypoint for the inference kernel |
| `fastvideo-kernel/benchmarks/benchmark_*.py` | Kernel-side attention benchmark scripts for FlashAttn2, SageAttn3, FP4, and comparison plots |
| `fastvideo-kernel/attn_qat_infer/blackwell/api.cu` | CUDA implementation behind `ATTN_QAT_INFER` |
| `fastvideo-kernel/tests/test_attn_qat_train.py` | Kernel-level test coverage for the training path |
| `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v_qat_attn.sh` | Ready-to-run Wan 1.3B Attn QAT finetune launcher |
| `examples/training/finetune/wan_t2v_14B/finetune_t2v_qat_attn.sh` | Ready-to-run Wan 14B Attn QAT finetune launcher |

## Build And Install

Before using either backend, make sure the in-repo kernel package is built from
source.

```bash
git submodule update --init --recursive
cd fastvideo-kernel
./build.sh
```

For local development, the important result is:

- `ATTN_QAT_TRAIN` can import `fastvideo_kernel`
- `ATTN_QAT_INFER` can import `attn_qat_infer`

Backend-specific notes:

- `ATTN_QAT_TRAIN` is the Triton training path shipped through
  `fastvideo-kernel`.
- `ATTN_QAT_INFER` currently targets the Blackwell CUDA path in
  `fastvideo-kernel/attn_qat_infer/`, is built by `fastvideo-kernel/CMakeLists.txt`,
  and requires CUDA 12.8+.

For the full kernel build requirements and CUDA notes, see:

- [Attention Overview](../index.md)
- `fastvideo-kernel/README.md`

## Running Training

Today the checked-in Attn QAT training launchers use the legacy training
pipeline in `fastvideo/training/wan_training_pipeline.py`.

### Ready-made launchers

Use the provided SLURM scripts directly:

```bash
sbatch examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v_qat_attn.sh
sbatch examples/training/finetune/wan_t2v_14B/finetune_t2v_qat_attn.sh
```

Both scripts already set:

```bash
export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN
```

Before launching, update the script-local values that are environment-specific:

- `WANDB_API_KEY`
- `MODEL_PATH`
- `DATA_DIR`
- `VALIDATION_DATASET_FILE`
- output directory and SLURM resource requests

### What the launchers run

The training scripts eventually invoke:

```bash
torchrun fastvideo/training/wan_training_pipeline.py ...
```

If you are adapting the workflow to your own cluster or running outside SLURM,
the key Attn QAT requirement is still the same:

```bash
export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN
```

Then launch the normal Wan training pipeline with your preferred `torchrun`
arguments and training flags.

## Running Inference

For standalone inference, prefer `ATTN_QAT_INFER` when that kernel is available.
Use `ATTN_QAT_TRAIN` only if you specifically want to exercise the training-side
attention path during validation or debugging.

### Python API

This follows the same pattern as `examples/inference/basic/basic.py`, but with
the Attn QAT backend forced through the environment:

```python
import os

from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ATTN_QAT_INFER"

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
)

generator.generate_video(
    "A cinematic close-up of rain on a neon street at night.",
    output_path="video_samples",
    save_video=True,
)
```

To run inference with a finetuned checkpoint instead of the base model, replace
the model ID with your checkpoint path.

### CLI

You can also force the backend when using the FastVideo CLI:

```bash
FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER \
fastvideo generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --num-gpus 1 \
  --sp-size 1 \
  --tp-size 1 \
  --height 480 \
  --width 832 \
  --num-frames 77 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --prompt "A cinematic close-up of rain on a neon street at night." \
  --output-path outputs_video/
```

This is the same basic flow used by `scripts/inference/v1_inference_wan.sh`,
just with `FASTVIDEO_ATTENTION_BACKEND` set explicitly.

## Troubleshooting

- If `ATTN_QAT_TRAIN` fails to import, verify that `fastvideo-kernel` built
  successfully and exposes `fastvideo_kernel`.
- If `ATTN_QAT_INFER` fails to import, verify that the local build exposes the
  `attn_qat_infer` package.
- If you want a known-safe fallback for debugging, use
  `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.

See also:

- [Inference Optimizations](../../inference/optimizations.md)
- [Debugging](../../utilities/debugging.md)
