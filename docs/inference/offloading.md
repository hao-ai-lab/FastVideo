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
> [!NOTE]
> For CLI usage, the underscore `_` should be replaced with a hyphen `-`.

### `use_fsdp_inference`:

Enable [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
for inference. The model weight will be sharded across multiple GPUs to reduce memory usage for each GPU, and weight will be broadcasted to all GPUs during inference layer by layer. When GPU interlink is fast,

#### Performance Impact
FSDP inference introduce negligible performance overhead because of weight prefetching. Performance overhead may be visible when GPU interlink is slow (e.g. multiple consumer-level GPUs connected by slow PCIe, and GPU P2P is not available).

#### Usage Recommendation
We recommend enabling this option when multiple GPUs are available.

### `dit_cpu_offload`:

Whether enabling CPU offloading for FSDP inference. When enabled, the model weight will be offloaded to CPU memory, and the weight of each layer will be moved to GPU memory only when that layer is being inferred.

#### Performance Impact
The implementation by pytorch does not overlap computation and data transfer, so enabling this option will harm performance.

#### Usage Recommendation
This option is not in effect when FSDP is disabled. We recommend use `dit_layerwise_offload` when single GPU is used. When FSDP is enabled and OOM still occurs, this option can be used to help.

### `dit_layerwise_offload`:
This option is similar to `dit_cpu_offload`, but 1) it overlaps computation and PCIe data transfer, and 2) it only works for single GPU inference.

#### Performance Impact
This option introduces negligible performance overhead.

#### Usage Recommendation
We recommend enabling this option when single GPU is used. This option is not compatible with FSDP.

### `text_encoder_cpu_offload`:
When this option is enabled, the text encoder model weight will be offloaded to CPU memory, and being computed on CPU.

#### Performance Impact
This option significantly slows down text encoding computation, but text encoding is usually not the bottleneck.

#### Usage Recommendation
We recommend enabling this option only when OOM happens.

### `image_encoder_cpu_offload` and `vae_cpu_offload`:
When these options are enabled, the weights are store in CPU memory, and moved to GPU memory when the corresponding module is being computed. After computation, the weights are moved back to CPU memory.

#### Performance Impact
These options introduce performance overhead due to PCIe data transfer.

#### Usage Recommendation
We recommend enabling these options when OOM happens.

## General Recommendations

### Single GPU Inference
We recommend enabling `dit_layerwise_offload`. If OOM happens, enable `image_encoder_cpu_offload` and `vae_cpu_offload` as well. When OOM still happens, consider enabling `text_encoder_cpu_offload`.

### Multi-GPU Inference
We recommend enabling `use_fsdp_inference`, disabling `dit_layerwise_offload` and `dit_cpu_offload`. If OOM happens, consider enabling `text_encoder_cpu_offload`, `image_encoder_cpu_offload` and `vae_cpu_offload`. When OOM still happens, consider enabling `dit_cpu_offload`.
