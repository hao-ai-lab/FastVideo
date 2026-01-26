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
```

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

## General Recommendations

### Single GPU Inference

We recommend enabling `dit_layerwise_offload`. If OOM happens, also enable `image_encoder_cpu_offload` and `vae_cpu_offload`. If OOM still happens, consider enabling `text_encoder_cpu_offload`.

### Multi-GPU Inference

We recommend enabling `use_fsdp_inference` and disabling both `dit_layerwise_offload` and `dit_cpu_offload`. If OOM happens, consider enabling `text_encoder_cpu_offload`, `image_encoder_cpu_offload`, and `vae_cpu_offload`. If OOM still happens, consider enabling `dit_cpu_offload`.

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
