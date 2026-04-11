# Attention QAT

Attention QAT in FastVideo covers two related, but different, backends:

- `ATTN_QAT_INFER`: the inference-oriented CUDA kernel path
- `ATTN_QAT_TRAIN`: the training-oriented Triton attention path

Both are selected with `FASTVIDEO_ATTENTION_BACKEND`, but they are not
interchangeable. The main practical split is:

- use `ATTN_QAT_INFER` for standalone inference with the dedicated inference
  kernel
- use `ATTN_QAT_TRAIN` for finetuning, validation during training, or when you
  specifically want to reproduce the training-side attention path

## Quick Start

If your goal is "run Wan 2.1 14B with Attention QAT inference weights", this is
the shortest path:

1. Build the in-repo kernel package so FastVideo can import `attn_qat_infer`.
2. Download the Wan 2.1 14B QAT checkpoint.
3. Edit the provided inference example to point at the 14B base model and the
   downloaded QAT safetensors.
4. Run the example with `ATTN_QAT_INFER`.

### Step 1. Build the kernel package

Before using either Attention QAT backend, build the in-repo
`fastvideo-kernel` package from source:

```bash
git submodule update --init --recursive
cd fastvideo-kernel
./build.sh
```

After a successful build:

- `ATTN_QAT_TRAIN` should be able to import `fastvideo_kernel`
- `ATTN_QAT_INFER` should be able to import `attn_qat_infer`

`ATTN_QAT_INFER` currently targets the Blackwell CUDA path under
`fastvideo-kernel/attn_qat_infer/` and requires CUDA 12.8+.

### Step 2. Download the Wan 2.1 14B QAT checkpoint

FastVideo includes a helper script:

- `examples/inference/optimizations/download_14B_qat.sh`

By default it downloads:

- Hugging Face repo: `FastVideo/14B_qat_400`
- local directory: `checkpoints/14B_qat_400`

Prerequisites:

- `huggingface_hub` installed, for example:
  `uv pip install huggingface_hub`
- access to the model repo if it is private or gated:
  `huggingface-cli login`

Run the downloader:

```bash
bash examples/inference/optimizations/download_14B_qat.sh
```

To download into a custom directory:

```bash
bash examples/inference/optimizations/download_14B_qat.sh /path/to/14B_qat_400
```

The script prints a ready-to-copy `init_weights_from_safetensors=...` value at
the end.

### Step 3. Edit the provided inference example

The example to start from is:

- `examples/inference/optimizations/attn_qat_inference_example.py`

Open that file and update these two values:

1. Change the base model from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` to
   `Wan-AI/Wan2.1-T2V-14B-Diffusers`
2. Replace
   `init_weights_from_safetensors="safetensors_path"` with the directory that
   contains the downloaded `.safetensors` files

Example:

```python
import os

from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ATTN_QAT_INFER"

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=True,
    dit_cpu_offload=False,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=False,
    init_weights_from_safetensors="checkpoints/14B_qat_400",
)
```

Important:

- the checked-in example currently uses the `1.3B` base model until you edit it
- do not load the 14B QAT weights on top of the `1.3B` base model; the weights
  and model config will not match

### Step 4. Run the inference example

```bash
python examples/inference/optimizations/attn_qat_inference_example.py
```

Generated videos are written to `video_samples/` by default.

## Backend Overview

| Backend | Best for | Package requirement | Primary kernel location |
|---------|----------|---------------------|-------------------------|
| `ATTN_QAT_TRAIN` | finetuning, training-time validation, reproducing the training path | `fastvideo_kernel` | `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py` |
| `ATTN_QAT_INFER` | standalone inference with the dedicated CUDA kernel | `attn_qat_infer` from the in-repo `fastvideo-kernel` checkout | `fastvideo-kernel/attn_qat_infer/` |

FastVideo routes backend selection through:

- `fastvideo/envs.py`
- `fastvideo/platforms/cuda.py`
- `fastvideo/attention/backends/attn_qat_train.py`
- `fastvideo/attention/backends/attn_qat_infer.py`

The legacy training pipeline also contains explicit Attention QAT integration:

- `fastvideo/training/training_pipeline.py`

That pipeline forces generator loading through `ATTN_QAT_TRAIN` when
`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` or `--generator_4bit_attn` is
enabled.

## Inference Workflows

For standalone inference, prefer `ATTN_QAT_INFER` when the CUDA kernel is
available. Use `ATTN_QAT_TRAIN` for inference only if you intentionally want to
exercise the training-side attention path for debugging or parity checks.

### Minimal Python example

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

### Loading custom safetensors during inference

FastVideo supports loading custom transformer weights through
`init_weights_from_safetensors`.

This value can point to either:

- a directory containing one or more `.safetensors` files
- a single `.safetensors` file

For Wan 2.1 14B QAT inference, the common pattern is:

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=True,
    init_weights_from_safetensors="checkpoints/14B_qat_400",
)
```

### CLI example

You can also force the backend from the command line:

```bash
FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER \
fastvideo generate \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
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

If you want to use custom QAT transformer weights from the CLI, pass the same
custom weight override that the Python API uses:

```bash
FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER \
fastvideo generate \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --init-weights-from-safetensors checkpoints/14B_qat_400 \
  --num-gpus 1 \
  --output-path outputs_video/ \
  --prompt "A cinematic close-up of rain on a neon street at night."
```

## Training Workflows

Today the checked-in Attention QAT training launchers use the legacy training
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

Before launching, update the script-local values that depend on your
environment:

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
the main Attention QAT requirement is still:

```bash
export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN
```

Then launch the normal Wan training pipeline with your preferred `torchrun`
arguments and training flags.

## Where The Code Lives

Use these paths when you want to trace or modify the Attention QAT flow:

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
| `fastvideo-kernel/benchmarks/benchmark_*.py` | Kernel-side benchmark scripts for FlashAttn2, SageAttention3, FP4, and comparison plots |
| `fastvideo-kernel/attn_qat_infer/blackwell/api.cu` | CUDA implementation behind `ATTN_QAT_INFER` |
| `fastvideo-kernel/tests/test_attn_qat_train.py` | Kernel-level test coverage for the training path |
| `examples/inference/optimizations/attn_qat_inference_example.py` | Ready-to-edit inference example for custom Attention QAT weights |
| `examples/inference/optimizations/download_14B_qat.sh` | Helper script for downloading the Wan 2.1 14B QAT checkpoint |
| `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v_qat_attn.sh` | Ready-to-run Wan 1.3B Attention QAT finetune launcher |
| `examples/training/finetune/wan_t2v_14B/finetune_t2v_qat_attn.sh` | Ready-to-run Wan 14B Attention QAT finetune launcher |

## Troubleshooting

- If `ATTN_QAT_TRAIN` fails to import, verify that `fastvideo-kernel` built
  successfully and exposes `fastvideo_kernel`.
- If `ATTN_QAT_INFER` fails to import, verify that the local build exposes the
  `attn_qat_infer` package.
- If the Wan 2.1 14B example fails after you changed only the checkpoint path,
  make sure you also changed the base model to
  `Wan-AI/Wan2.1-T2V-14B-Diffusers`.
- If you hit issues with CPU memory pressure or obscure CUDA argument errors in
  the example script, try setting `pin_cpu_memory=False`.
- If you want a known-safe fallback for debugging, use
  `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.

## Related Pages

- [Attention Overview](../index.md)
- [Inference Optimizations](../../inference/optimizations.md)
- [Debugging](../../utilities/debugging.md)
