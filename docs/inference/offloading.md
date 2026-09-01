# Offloading

This page describes how to use offloading techniques for inference to reduce GPU memory usage while maintaining acceptable performance.

## Default Behavior

```python
dit_cpu_offload: bool = True
use_fsdp_inference: bool = False
dit_layerwise_offload: bool = True
text_encoder_cpu_offload: bool = True
image_encoder_cpu_offload: bool = True
vae_cpu_offload: bool = True
pin_cpu_memory: bool = True
lazy_module_load: bool | None = None
```

On unified-memory accelerators such as NVIDIA GB10 and Apple silicon, FastVideo
detects the selected device inside each worker and disables all five host-offload
modes before loading modules. Host and accelerator allocations share one physical
pool there, so offload adds transfers and duplicate residency instead of freeing
memory. CUDA FSDP sharding remains enabled when requested; MPS continues to
disable FSDP. `pin_cpu_memory` is not an offload mode and is left unchanged.

MiniMax H3 CUDA inference can use two levers that do not copy weights to a host
pool. `lazy_module_load` is the general path: each opted-in component loads on
first use and is freed after its last stage, so a later `generate()` reloads
from disk in-process and the DiT can drop before VAE decode. On GB10 it
auto-enables and owns deferral. `h3_sequential_load` is the H3-only fallback
when lazy is off: load Qwen3-VL, run conditioning, release that encoder, then
load the DiT and VAEs. When both would arm, sequential stands down so VAE
`torch.compile` can attach to the lazy proxy. Input preparation and unpatchify
read geometry from checkpoint `config.json` (VAE spatial ratio / latent
channels, DiT patch size) so those stages do not materialize weights just to
read two integers. The MLX FastH3 runtime always uses this phase order. When
host offload is off, DiT safetensors are read onto the accelerator instead of
CPU-then-copy. Both flags default to auto (`None`) and turn on for
unified-memory devices such as GB10; lazy then disables sequential. Pass
`--no-lazy-module-load` to keep every component resident (sequential may still
auto-arm). Two-node Spark
jobs still need this split: sequence parallel replicates the DiT on each GB10
(~66 GiB of weights plus activations). See
[Pair two NVIDIA DGX Sparks](../getting_started/installation/spark_pair.md).

## Behavior Explanation

!!! note
    For CLI usage, replace underscores (`_`) with hyphens (`-`).

### `use_fsdp_inference`

Enables [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for inference. The model weights are sharded across multiple GPUs to reduce memory usage per GPU, and weights are broadcast to all GPUs layer by layer during inference.

#### Performance Impact

FSDP inference introduces negligible performance overhead due to weight prefetching. Performance overhead may be visible when GPU interconnect is slow (e.g., multiple consumer-level GPUs connected by slow PCIe without GPU P2P support).

#### Usage Recommendation

We recommend enabling this option when multiple GPUs are available.

### `dit_cpu_offload`

Enables CPU offloading for FSDP inference. When enabled, the model weights are offloaded to CPU memory, and the weight of each layer is moved to GPU memory only when that layer is being computed.

#### Performance Impact

The PyTorch FSDP implementation does not overlap computation and data transfer perfectly for inference, so enabling this option will harm performance.

#### Usage Recommendation

This option only takes effect when FSDP is enabled. For single GPU usage, we recommend using `dit_layerwise_offload` instead.

### `dit_layerwise_offload`

This option is similar to `dit_cpu_offload`, but with two key differences:

1. It overlaps computation and PCIe data transfer
2. It only works for single GPU inference

#### Performance Impact

This option introduces negligible performance overhead.

#### Usage Recommendation

We recommend enabling this option for single GPU usage. This option is not compatible with FSDP.

### `h3_sequential_load`

MiniMax H3 only. When enabled, the pipeline loads Qwen3-VL, runs conditioning,
releases that encoder, then loads the DiT and VAEs. Default is auto: enabled on
unified-memory accelerators, disabled on discrete GPUs.

#### Performance Impact

On GB10 / Spark this is required so encoder, DiT, and VAE weights are not all
resident in one unified pool. On discrete GPUs (for example 4×GB200) sequential
load is unnecessary and it blocks a second `generate()` on the same worker
because the encoder has been released.

#### Usage Recommendation

Leave the default on Spark / DGX Spark when `lazy_module_load` is off. When
both would arm (the GB10 auto case), lazy owns deferral and sequential stands
down so VAE `torch.compile` can attach to the lazy proxy. Force
`--h3-sequential-load` only when you need the split on a discrete GPU without
lazy load. Use `--no-h3-sequential-load` when you need more than one prompt per
worker and have enough memory to keep the encoder.

### `text_encoder_cpu_offload`

When enabled, the text encoder model weights are offloaded to CPU memory, and text encoding is computed on CPU.

#### Performance Impact

This option significantly slows down text encoding computation, but text encoding is usually not the bottleneck.

#### Usage Recommendation

We recommend enabling this option only when OOM happens.

### `image_encoder_cpu_offload` and `vae_cpu_offload`

When enabled, the weights are stored in CPU memory and moved to GPU memory when the corresponding module is being computed. After computation, the weights are moved back to CPU memory.

#### Performance Impact

These options introduce performance overhead due to PCIe data transfer.

#### Usage Recommendation

We recommend enabling these options when OOM happens.

### `lazy_module_load`

Every option above moves weights between host and device. This one changes
whether they are in memory at all.

By default a pipeline loads every component before the first stage runs, so
peak memory is the sum of all of them even though no two are needed at the same
moment. With `lazy_module_load` enabled, each heavy component loads on first use
and is freed once the last stage that needs it has returned, so peak memory
becomes the largest overlapping set instead of the sum. MiniMax-H3 T2VA is
`max(text encoder, DiT, VAE)` rather than `text encoder + DiT + VAE`, because
the DiT is not held through VAE decode.

#### Performance Impact

A freed component is read from disk again on the next generation, so a
multi-prompt run pays one reload per component per request. For a large text
encoder that is tens of seconds. If pipeline-level `torch.compile` is enabled,
the compile setup is reapplied after each reload; PyTorch can reuse its graph
and kernel caches when the component structure and input shapes are unchanged.

#### Usage Recommendation

Enable this when a model does not fit at load time, which the CPU offload
options above cannot help with because they act after loading. It is
particularly relevant on unified-memory devices, where host and device draw on
the same pool and moving weights to the host frees nothing. FastVideo
auto-enables it there (`lazy_module_load=None`). Leave it off when the model
already fits, or pass `--no-lazy-module-load` to keep components resident for
later `generate()` calls.

This option applies to inference only. Training keeps every component resident
and logs a warning if the flag is set.

Deferral is opt-in per pipeline. Releasing a component and loading it again is
only safe when nothing outside the loader has changed it, and two common habits
break that without raising: mutating a component after load, as LongCat does
when it enables block-sparse attention, and reading a component's attributes
while stages are built, as the shared denoising stage does to pick an attention
backend. A pipeline therefore lists the components it has checked in
`_lazy_module_names`, which is empty in the base class. MiniMax-H3 opts in. On
a pipeline that has not, the flag is a no-op: hooks are not installed and no
warning is logged. Sequential MiniMax-H3 (`h3_sequential_load`) reloads the
text encoder for a later `generate()` on the same worker; you do not need to
start a new generator.

## General Recommendations

### Single GPU Inference

We recommend enabling `dit_layerwise_offload`. If OOM happens, also enable `image_encoder_cpu_offload` and `vae_cpu_offload`. If OOM still happens, consider enabling `text_encoder_cpu_offload`.

### Multi-GPU Inference

We recommend enabling `use_fsdp_inference` and disabling both `dit_layerwise_offload` and `dit_cpu_offload`. If OOM happens, consider enabling `text_encoder_cpu_offload`, `image_encoder_cpu_offload`, and `vae_cpu_offload`. If OOM still happens, consider enabling `dit_cpu_offload`.

### When the Model Does Not Fit at Load Time

The offload options only help once loading has finished. If the run dies while
components are still being placed, or if the machine has unified memory so
there is no separate host pool to offload into, enable `lazy_module_load`.

## Examples

### Single GPU with Layerwise Offloading

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    # Recommended for single GPU
    dit_layerwise_offload=True,
    # Enable if OOM happens
    vae_cpu_offload=True,
    image_encoder_cpu_offload=True,
    text_encoder_cpu_offload=True,
    # Speeds up CPU-GPU transfer
    pin_cpu_memory=True,
)

prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers."
video = generator.generate_video(prompt, output_path="output/", save_video=True)
```

### Multi-GPU with FSDP

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=2,
    # Recommended for multi-GPU
    use_fsdp_inference=True,
    dit_layerwise_offload=False,
    dit_cpu_offload=False,
    # Enable if OOM happens
    vae_cpu_offload=True,
    image_encoder_cpu_offload=True,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=True,
)

prompt = "A majestic lion strides across the golden savanna."
video = generator.generate_video(prompt, output_path="output/", save_video=True)
```
