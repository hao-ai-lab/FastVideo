(support-matrix)=
# Support Matrix

| Model Name | HuggingFace Model ID | Resolutions | TeaCache| Sliding Tile Attn | Sage Attn |
|------------|----------------------|-------------|-------------|--------------------------|--|
| HunyuanVideo | hunyuanvideo-community/HunyuanVideo | |  | ✅ | ✅ |
| FastHunyuan | FastVideo/FastHunyuan-diffusers | | | ✅| ✅|
| Wan T2V 1.4B | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 480P | ✅ | ✅ | ✅ |
| Wan T2V 14B | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 480P, 720P | ✅| ✅| ✅|
| Wan I2V 480P | Wan-AI/Wan2.1-I2V-14B-480P-Diffusers | 480P | ✅| ✅| ✅|
| Wan T2V 720P | Wan-AI/Wan2.1-T2V-14B-Diffusers | 720P | ✅| ✅| ✅|
| StepVideo T2V | Coming soon! | 768px768px204f, 544px992px204f, 544px992px136f | | | |

## Special requirements

### StepVideo T2V
- The self-attention in text-encoder (step_llm) only supports CUDA capabilities sm_80 sm_86 and sm_90

### Sliding Tile Attention
- Currently only Hopper GPUs (H100s) are supported.
